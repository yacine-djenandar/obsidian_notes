## year: 2018

## Architecture

ShmCaffe is a distributed deep learning platform built on **BVLC Caffe (v1.0.0)** that introduces **remote shared memory** for parameter sharing. It supports both **asynchronous SGD (ASGD)** and a **hybrid training mode** combining _inter-node ASGD_ with _intra-node synchronous SGD_.

The system uses **MPI** for communication between distributed workers and integrates the **Soft Memory Box (SMB) Framework** to manage remote shared memory. Parameters are shared through a modified **Elastic Averaging SGD (EASGD)** algorithm adapted for shared memory — called **SEASGD (Shared-memory EASGD)**.

The **master MPI process (rank 0)** creates and manages the shared memory buffers, while **slave processes** attach to them for updates. ShmCaffe maintains compatibility with all Caffe hyperparameters and adds two new ones:

- **Update interval** – how often global weights are updated.
    
- **Moving rate** – scaling factor for averaging global and local weights.

## Virtual Shared Memory Framework

The **Soft Memory Box (SMB)** is a **virtual shared memory framework** designed to speed up communication between distributed deep learning workers. It enables **remote shared memory (RSM)** access across multiple nodes using **RDMA (Remote Direct Memory Access)** over **InfiniBand**.

Built mostly from scratch (except the InfiniBand module, adapted from Linux’s RDS), SMB lets processes read and write directly to remote memory buffers. It provides APIs for memory allocation, data transfer, accumulation, and update notifications.

In operation, the **master worker** creates shared memory buffers via the SMB API and broadcasts a **shared memory key (SHM key)** to other workers. The **slave workers** then use this key to request access from the SMB server, which grants an InfiniBand remote key enabling direct RDMA access. Once set up, all workers can efficiently share parameters through this shared memory space.

![[Pasted image 20251025233012.png]]


## Elastic Averaging SGD with Shared Memory

The **SEASGD** algorithm, introduced in **ShmCaffe**, is an asynchronous parameter update mechanism built on top of the **Soft Memory Box (SMB)** remote shared memory framework. It aims to accelerate distributed deep learning by enabling efficient weight sharing and synchronization among multiple workers.

Each **worker** (implemented as an MPI process or thread) trains a **local replica** of the neural network using its own mini-batches without data duplication. During training, workers compute gradients, update their local parameters, and synchronize with the **global parameters** stored in shared memory.

Unlike **Downpour SGD**, where the parameter server alone updates global weights, **EASGD** and its shared-memory extension **SEASGD** allow **both workers and the server** to update parameters. This design improves parallel efficiency and reduces communication bottlenecks.

---

### **1. Downpour SGD**

In Downpour SGD, the **parameter server** updates the global weights WgW_gWg​ using the gradients GxG_xGx​ sent by each worker xxx:

$Wg′​=Wg​−ηGx​$


where:

- $η$: learning rate
    
- $W_g$​: global weights
    
- $G_x$: gradient from worker xxx
    

---

### **2. Elastic Averaging SGD (EASGD)**

EASGD allows both workers and the parameter server to update weights cooperatively.

- **Local weight update from gradients:**
    

$Wx′​=Wx​−ηGx​$

- **Worker adjustment toward global weights:**
    

$Wx′′​=Wx′​−α(Wx​−Wg​)$

- **Global weight update (server side):**
    

$Wg′​=Wg​+α(Wx​−Wg​)$

where $\alpha$ is the **moving average rate**, which controls how strongly local and global weights are pulled toward each other.

---

### **3. Shared-Memory-Based EASGD (SEASGD)**

SEASGD replaces traditional network communication with **remote shared memory** using the SMB framework.  
Each worker computes and shares a **weight increment** $ΔWx$​:

$ΔWx​=α(Wx′​−Wg​)$

The updates are then applied locally and globally:

$Wx′′​=Wx′​−ΔWx​$
$Wg′​=Wg​+ΔWx​$

Unlike traditional parameter servers, the **SMB server** does not handle optimization logic — it only provides shared memory buffers and supports **accumulation operations**. This enables workers to collaboratively and asynchronously update shared parameters.

## Hybrid SGD

ShmCaffe enhances distributed deep learning efficiency by combining **intra-node synchronous SGD** with **inter-node asynchronous updates**, a method referred to as **Hybrid SGD (HSGD)**.

Within each node, workers (GPUs) are grouped together and use **NVIDIA’s NCCL `ncclAllReduce`** to aggregate their gradients and update a shared local model synchronously. Then, a **root worker** from each group performs asynchronous global parameter updates via the **SEASGD** algorithm using the **SMB server**. The updated global weights are broadcast back to the other workers in the group.

This hierarchical design reduces **inter-node communication traffic** and improves **training performance**. However, since ShmCaffe currently relies on a **single SMB server**, its communication bandwidth is limited, causing overhead to increase when training **large-scale models** with many workers.

![[Pasted image 20251025235036.png]]

## Aligning termination of all workers

ShmCaffe introduces mechanisms to better **synchronize the training progress** of distributed workers and **maximize GPU utilization** by aligning their training completion times.

In asynchronous SGD systems, even when all workers use identical GPUs, differences in computation time arise due to shared resources such as the system bus, file I/O, and network bandwidth. As BVLC Caffe terminates training after a fixed number of iterations rather than by accuracy or loss, faster workers must **wait idle** for slower ones to finish, wasting computational resources.

To address this, ShmCaffe replaces centralized coordination with a **shared-memory-based progress tracking system** using the **SMB framework**. Each worker writes its **current iteration count** to a shared memory buffer, allowing all workers to monitor collective progress in real time.

Based on this shared information, workers can **adjust their remaining iterations** so that all training processes finish roughly simultaneously. ShmCaffe supports several synchronization criteria:

1. All workers stop when the **master worker** finishes.
    
2. All workers stop when the **fastest worker** finishes.
    
3. All workers stop when the **average number of iterations** across workers reaches the target count.
    

This approach significantly reduces idle GPU time and improves overall **training efficiency** in distributed environments.

## Use of Shared Memory Buffer at SMB Server

In the **SEASGD** scheme, the shared memory for parameters is organized as shown in Fig. 5. A **global weight buffer** $Wg$​ is created on the **SMB server** and shared among all deep-learning workers.

Each worker allocates its own **shared memory buffer** $ΔWx$​ on the SMB server to store the **weight increment**, which represents the difference between its **local weight** and the **global weight**. These individual buffers $ΔWx$​ are **not shared** with other workers.

The values stored in each ΔWx\Delta W_xΔWx​ are **accumulated** into the global weight buffer $Wg$​. Since each worker can **read and write** exclusively to its own $ΔWx$​, parameters can be exchanged at **high speed**, leveraging the full physical bandwidth of the **InfiniBand** network.

![[Pasted image 20251025235608.png]]

## SEASGD Procedures in ShmCaffe

Figure below illustrates the **training procedure of SEASGD** as implemented in **ShmCaffe**.  
Each worker xxx launches two threads:

- a **main thread** that performs deep learning computations, and
- an **update thread** that updates the global weights in parallel.

The **update thread** remains **blocked** until the **main thread** wakes it up. At the start of each iteration, the main thread:

1. **Reads the global weights** WgW_gWg​ (T1),
2. **Updates the local weights** using the difference between the local and global weights (T2), and then
3. **Wakes up the update thread** (T3) to perform global updates asynchronously.

While the update thread communicates with the **SMB server**, the main thread continues training by processing a mini-batch, computing gradients (T4), and updating local weights (T5).

The update thread, once awakened, acquires a **lock**, writes the **weight increment** into the SMB shared memory (T.A1), and requests global weight accumulation from the SMB server (T.A2). The server processes these accumulation requests exclusively (T.A3) and notifies each worker upon completion (T.A4). After receiving the result, the update thread releases the lock and waits for the next wake-up signal.

If the **global weight update** (T.A1–T.A4) takes longer than the training and local update steps (T4–T5), the main thread will **pause** before (T2) until the update thread completes its work (T.A5).

ShmCaffe intentionally does **not overlap** the global weight reading phase with computation because doing so could cause **stale parameter issues** that degrade learning performance.

![[Pasted image 20251025235810.png]]

# Evaluation

## Read/Write Bandwidth in a SMB Server

In the experimental setup, the authors evaluated the **read/write performance** of the **SMB server’s shared memory** as the number of processes increased from 2 to 32. Each process allocated a **1 GB shared memory buffer** and performed **50% read and 50% write operations**, repeated across **10 trials**.

The results  indicate that the **aggregated bandwidth** reached **6.7 GB/s**, which is about **96% of the maximum 7 GB/s bandwidth** supported by the InfiniBand HCA.

This demonstrates that the **SMB server efficiently utilizes the available hardware bandwidth**, confirming that it is **well optimized for high-performance communication** in distributed deep learning.

![[Pasted image 20251026000232.png]]

## ShmCaffe vs. Distributed Deep Learning Frameworks

The authors evaluated the training performance of **four deep learning frameworks**—BVLC Caffe, Inspur Caffe-MPI, MPICaffe, and the proposed **ShmCaffe**—across **single-GPU, multi-GPU, and multi-node** environments.

#### **Frameworks compared**

1. **BVLC Caffe (v1.0.0)** – The original single-node framework using **SSGD (Synchronous SGD)** with **NCCL AllReduce** for multi-GPU synchronization.
2. **Caffe-MPI (v1.0)** – A distributed version by **Inspur** using **MPI Send/Recv** for gradient exchange in a **star topology**: the master gathers gradients from slaves, averages them, and redistributes updated weights.
3. **MPICaffe** – Implemented by the authors for comparison, based on BVLC Caffe but using **MPI AllReduce** for distributed gradient aggregation instead of NCCL.
4. **ShmCaffe (proposed)** – The authors’ system using **remote shared memory** and **hybrid SGD (HSGD)** combining intra-node synchronous and inter-node asynchronous updates.
#### **Experimental setup**

- **Dataset:** ILSVRC 2012 (ImageNet) – 1,331,167 images (1,281,167 for training, 50,000 for validation), converted to LMDB format (≈ 240 GB training, 10 GB validation).
- **Model:** **Inception v1**.
- **Mini-batch size:** 60 per worker (480 for 8 GPUs, 960 for 16 GPUs).
- **Learning parameters:**
    - Base learning rate η=0.1\eta = 0.1η=0.1
    - γ=0.1\gamma = 0.1γ=0.1
    - Momentum = 0.9
    - Step size = 4 epochs
    - Max iteration = 15 epochs
    - For ShmCaffe: moving rate = 0.2, update interval = 1

- **Hardware:** Distributed configurations shown in Table I; I/O bottleneck minimized using prefetching and an NFS server (1.5 GB/s over RAID0 SSDs).

All systems performed **gradient aggregation via backpropagation** across all layers, with no data augmentation, since the focus was on **speed**, not accuracy.

#### **Results**

- **Accuracy:**  
    ShmCaffe showed stable convergence comparable to Caffe, though slightly lower in final accuracy. However, it outperformed both **Caffe-MPI** and **MPICaffe** when scaling to **16 GPUs**.
    
- **Training time:**
    - **ShmCaffe** trained **10.1× faster** than BVLC Caffe.
    - **2.8× faster** than Caffe-MPI on 16 GPUs.
    - Its **communication time** per iteration was **5.3× faster** than Caffe-MPI.


![[Pasted image 20251026000648.png]]

![[Pasted image 20251026000705.png]]

![[Pasted image 20251026000723.png]]

## Asynchronous ShmCaffe vs. Hybrid ShmCaffe

This section compares the performance of two ShmCaffe variants — **asynchronous ShmCaffe (ShmCaffe-A)** and **hybrid ShmCaffe (ShmCaffe-H)** — across different hardware configurations.
#### **Asynchronous ShmCaffe (ShmCaffe-A)**

- Uses the **SEASGD** algorithm.
- As the number of GPUs increases, **accuracy gradually decreases**.
- Up to **8 GPUs**, convergence time remains similar to the 1-GPU case, but at **16 GPUs**, accuracy drops to **79.2%**, about **5.7% lower** than single-GPU performance.
- The results confirm that **asynchronous SGD (ASGD)** becomes less efficient as the number of workers grows, especially beyond 16 GPUs.
- These experiments were conducted with **moving rate = 0.2** and **update interval = 1**.

#### **Hybrid ShmCaffe (ShmCaffe-H)**

- Combines **intra-node SSGD (synchronous SGD)** and **inter-node SEASGD**.
- Tested configurations:
    - **4 GPUs** → 2 nodes (2 GPUs each)
    - **8 GPUs** → 2 nodes (4 GPUs each)
    - **16 GPUs** → 4 nodes (4 GPUs each)
- Achieved accuracies:
    - **84.0% (4 GPUs)**
    - **82.7% (8 GPUs)**
    - **83.5% (16 GPUs)**

These results are **very close to single-GPU Caffe accuracy** (only **0.9–2.2% difference**) and show minimal loss variation (**0.04–0.11**).

Figure 12 compares the **training times** of the two ShmCaffe modes — **asynchronous (ShmCaffe-A)** and **hybrid (ShmCaffe-H)** — as the number of GPUs increases (2, 4, 8, and 16).

- **ShmCaffe-A** achieves up to **11.5× faster training** with **16 GPUs** compared to **single-GPU Caffe**, which requires about **23 hours** to train the Inception v1 model for 15 epochs.
- **ShmCaffe-H** achieves up to **10.1× faster performance** under the same conditions.

The **speed advantage of ShmCaffe-A** comes from its **lack of synchronization** between GPUs, which avoids data I/O contention and synchronization overhead.  
However, **ShmCaffe-H**, which uses **synchronous SGD within nodes** and **SEASGD across nodes**, provides better balance and stability when each node contains multiple GPUs.

![[Pasted image 20251026001013.png]]


![[Pasted image 20251026001026.png]]

## Computation and Communication per 4 CNNs

The experiment analyzed the **computation** and **communication times** of **ShmCaffe-A** (asynchronous) and **ShmCaffe-H** (hybrid) during the training of **four CNN models** (listed in Table IV) under various configurations (Table III).

The training time for **1000 iterations** was measured and averaged to compute the **time per iteration**, separating computation and communication components.  
Communication time represents the part of training **not overlapped** with computation.

---

#### **Configuration Explanation**

In **ShmCaffe-H**, “S” denotes **synchronous SGD** (SSGD) and “A” denotes **asynchronous SGD** (SEASGD).  
Example:

- **8 (S4×A2)** → 8 GPUs organized into **2 groups** of 4 GPUs each.
    
    - Within each group: **SSGD** synchronizes local updates.
        
    - Across groups: **SEASGD** asynchronously updates global weights.
        

Table IV provides the **parameter size** and **forward/backward computation times** of the four CNNs, measured using **BVLC Caffe (v1.0.0)**.

---

#### **Results – ShmCaffe-A**

Figure 13 presents computation and communication times for the four CNN models, with **update interval = 1**.

- **Inception v1:**
    
    - Communication ratio = **16.3% (8 GPUs)** and **26% (16 GPUs)** → communication overhead grows but remains manageable due to small parameter size.
        
- **ResNet-50:**
    
    - Communication ratio increases to **30% (8 GPUs)** and **56% (16 GPUs)** → communication time exceeds computation time at large scale.
        
- **Inception-ResNet v2:**
    
    - Uses larger 320×320 images and has a larger model size.
        
    - Communication time grows sharply with more GPUs; for 16 GPUs, per-iteration communication volume reaches **6848 MB** (214 MB × 2 × 16).
        

---

#### **Training Time Formula**

The **training time per iteration** is defined as:

$Titer=Tcomp+Tcomm=max⁡[Tcomp,(Twwi+Tugw)]+Trgw+Tulw$

where:

- $T_{iter}$​: total training time per iteration
    
- $T_{comp}$​: total computation time (forward, backward, local weight update)
    
- $T_{comm}$​: total communication time
    
- $T_{wwi}$​: time for writing weight increment (Eq. 5)
    
- $T_{ugw}$​: time for updating global weight (Eq. 7)
    
- $T_{rgw}$​: time for reading global weight
    
- $T_{ulw}$​: time for updating local weight (Eq. 6)
    

Since ShmCaffe performs these steps **sequentially** each iteration, $T_{rgw}$​ and $T_{ulw}$​ cannot be merged into $T_{comp}$​.  
However, **weight writing and updating** $(T_{wwi} + Tugw​)$ may overlap with computation; if their combined duration is shorter than $T_{comp}$​, communication is effectively **hidden**.

---

#### **Observation – VGG16 Case**

For the **VGG16** model:

- With **2 GPUs**, per-iteration communication time = **727.7 ms**,  
    and total iteration time = **941.8 ms**.
    
- By contrast, one iteration with **1 GPU** takes **389.8 ms**.
    

Thus, for models with **very large parameters** and **short computation times** (like VGG16), distributing training across multiple nodes becomes inefficient — it’s **better to train on a single node** to avoid communication overhead.

![[Pasted image 20251026001830.png]]

This section compares the **computation** and **communication times** when training four CNN models using **ShmCaffe-H** (the hybrid version combining synchronous and asynchronous SGD).

In this context:

- Configurations like **4 (S4)** mean that **4 GPUs** are trained with **synchronous SGD (SSGD)** only, used as a **baseline** for comparison with ShmCaffe-H.
    
- The **x-axis** in Figure 14 represents different configurations mixing **SSGD** (within nodes) and **SEASGD** (across nodes).
    

---

### ⚙️ **Results Overview**

- For **Inception v1**, **ResNet-50**, and **Inception-ResNet v2**,  
    the **communication ratio** (i.e., proportion of time spent on communication vs. computation) generally stays **below 30%**, with a few exceptions.
    
- For smaller models like **Inception v1** and **ResNet-50**,  
    communication time **does not decrease much** compared to **ShmCaffe-A**,  
    because ShmCaffe-H adds extra **gradient aggregation steps** for SSGD within each group of workers.
    
- For **Inception-ResNet v2** (a much larger model):
    
    - When trained on **16 GPUs**, the communication ratio **drops from 65% to 30.7%**.
        
    - This happens because ShmCaffe-H reduces inter-node communication by **4×** compared to ShmCaffe-A, thanks to grouping and local synchronization.
        
- For **VGG16**, however:
    
    - Communication is still **very high** — about **35%** even with 4 GPUs on one machine.
        
    - It **rises to around 80%** when training with **16 GPUs** across 4 machines.
        
    - Therefore, **multi-node scaling** is **inefficient** for VGG16 due to excessive communication overhead.


	![[Pasted image 20251026002003.png]]


The **computation times** of **ShmCaffe-A** (asynchronous) and **ShmCaffe-H** (hybrid) are **nearly identical**.

However, as shown in **Figure 15**, differences emerge in **communication time** depending on model size and scale:

- For **smaller models** trained on **8 GPUs**,  
    both ShmCaffe-A and ShmCaffe-H show **similar communication performance**.
    
- As the **model size** (number of parameters) **increases** and the system **scales to more GPUs (e.g., 16)**,  
    **ShmCaffe-H** achieves **much lower communication time** than ShmCaffe-A.

![[Pasted image 20251026002150.png]]





