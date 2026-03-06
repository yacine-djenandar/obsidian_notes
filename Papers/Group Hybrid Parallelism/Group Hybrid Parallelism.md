
## Date 2021

## System Overview

GHP is an extension to Hybrid Parallelism, where the entire training model is divided into **K sequential submodels**, and **all of the workers are divided into R groups**, meaning that {w1, w2, .... wk} will train K submodels in a single group(on the same data), and R * wk are assigned to train submodel K,.

**Data parallelism is applied on groups, and model parallelism is applied within each group**.

when training is over, a synchronisation phase is done between all workers in all groups to update each one's model's parameters.

**MPI is used for scatter and gather operations within a single group**.

![[Pasted image 20251020002557.png]]

### Step 1: Model Partitioning

this step involves dividing the model into K submodels. this helps reducing synchronization times as sync times can be overlapped by different submodels. and memory requirements are relaxed since worker holds only a submodel, allowing for training on bigger batch sizes.

### Step 2: Worker Allocation Step

this step maps workers to submodels, that divinde ***NP*** workers into R equivalent groups, and assigns them to **K submodels** to determine worker allocation = {w1, w2, ...., wk}, which means
### $R * (w_1 + w_2 + ... + w_k) = NP$
	
### Training Time

training time per iteration is composed of: **Computation time, communication time, and synchronization time***
**The figure below shows each framework with its training time formula**

![[Pasted image 20251020003641.png]]

![[Pasted image 20251026115054.png]]
#### Computation Time

This section defines the **computation time** — the time each worker needs to perform training operations such as **feed-forward** and **backpropagation**.

It depends on how much computational work (measured in **FLOPs**, or floating-point operations) each worker performs and how powerful the worker is.

If the **batch size** is large enough (≥ 32), the workload is assumed to be **evenly distributed** among all CPU/GPU cores and threads, so each worker’s performance is roughly equal to its **computational capacity** (measured in FLOPs per second).

---

### ⚙️ **Formula Explanation**

The total computation time of worker group $G_k$​ is:

![[Pasted image 20251026115333.png]]

Where:

- $T_{comp_k}$​​: total computation time for group $GkG_kGk​$
    
- $T_{comp,k}^F$​: time for **feed-forward**
    
- $T_{comp,k}^B$​: time for **backpropagation**
    
- $C_k^F$​: number of FLOPs in feed-forward
    
- $C_k^B$​: number of FLOPs in backpropagation
    
- $w_k​$ number of workers in group $GkG_kGk​$
    
- $f$: FLOPs per second (computational performance of each worker)

### Communication Time

The **communication time** represents the delay caused by exchanging activation or gradient data between adjacent submodels. It includes the time for **scattering input data** to workers of a submodel and **exchanging activations or gradients** between submodels. This time depends on **network bandwidth** and **data size**.

For a submodel $G_k$​, input data of size $D_{k-1}/R$ is received by $xk=dwkPex_k = \frac{d w_k}{P_e}xk​=Pe​dwk​​$ compute nodes, producing activations of size $D_k/R$. When $w_k > 1$, additional delay occurs while **distributing input data** to workers and **gathering output** for the next submodel.

Communication time is divided into **inter-node** and **intra-node** parts.

- **Inter-node communication** occurs between nodes and follows **Hockney’s algorithm**, assuming all nodes in a group are interconnected. It uses **MPI Scatter** modeled by **MST** or **BKT** algorithms.
    
- **Intra-node communication** uses **PCIe links** for direct data transfer.
    

The total communication time for submodel $G_k$​ is given by:

![[Pasted image 20251026120022.png]]

where
## $\delta_k = D_{k-1} + D_k$





