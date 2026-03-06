
## Year 2022


## Teaching-Learning-based Optimization (TLBO)

**Teaching–Learning-Based Optimization (TLBO)** is a population-based metaheuristic designed for continuous large-scale optimization. It treats the objective function as a black box and is valued for being simple, effective, and easy to configure since it requires only two parameters: **population size** and **number of iterations**.

TLBO models the learning process of a class of students improving their skills through two main stages:

1. **Teacher Stage:**
    
    - The best solution acts as the _teacher (T)_.
        
    - The average of all solutions (_M_) is calculated.
        
    - Each solution (_S_) is updated toward the teacher using:
        
        ![[Pasted image 20251026163859.png]]
        
        where $r_i \in [0,1]$ and the _teaching factor_ $T_F$ is randomly 1 or 2.
        
    - New solutions replace their originals if they perform better.
        
2. **Learner Stage:**
    
    - Each solution interacts with another randomly chosen one (_PS_).
        
    - Depending on which is better, the solution shifts toward improvement using:
        
    ![[Pasted image 20251026164015.png]]
        
    - Improved candidates replace their predecessors.
        

After all cycles, the **best candidate** found is returned as the final solution.

## Representation of Candidate Solutions

The **Teaching–Learning-Based Optimization (TLBO)** algorithm is applied to optimize **neural network architectures**, which are represented as **vectors in** $\mathbb{R}^N$, where $N$ is the number of layers considered. The **objective function** $f: \mathbb{R}^N \to \mathbb{R}$ evaluates each candidate network.

Each vector component corresponds to a **layer**, encoded as a **real number** combining two parts:

- The **integer part** ($I_i$​) indicates the **type of layer** (e.g., convolutional, dense, dropout, etc.).
    
- The **decimal part** ($D_i$​) encodes the **configuration** of that layer (e.g., activation type, number of neurons, etc.).

The user defines the maximum number of layers $N$, and a special type can represent **disabled layers**, allowing TLBO to search architectures of varying depth instead of a fixed-length structure.

The **decimal encoding** maps the space of possible configurations to a value in $[0,1]$, inspired by how multidimensional matrices are flattened into one-dimensional vectors. This approach may involve rounding, but since TLBO is **derivative-free**, small plateaus or precision issues do not affect its performance significantly.

For example, a layer type with two parameters—activation function (4 options) and number of neurons (up to 500)—defines a configuration space of $4×500=2000$  possibilities. These can be represented as a single decimal value (e.g., 0.85) that indexes one configuration, possibly through a lookup table.

![[Pasted image 20251026165519.png]]

## Cost Function

The **cost (objective) function** in the TLBO framework must map candidate solutions from RN\mathbb{R}^NRN to a real value, f:RN→Rf : \mathbb{R}^N \to \mathbb{R}f:RN→R. Its role is to evaluate and rank neural network architectures based on their performance.

If **all architectures are feasible**, the cost corresponds to the **average validation accuracy (or error)** after training each network under the same dataset and training conditions.

If **some architectures are infeasible** (e.g., incompatible layer combinations such as applying a 7×7 convolution on a 5×5 feature map), they are penalized with a **high cost value (1000 − number of correct layers)**. This allows TLBO to not only distinguish between feasible and infeasible solutions but also **rank** the latter based on how close they are to being valid.

To reduce computational effort, it’s recommended to include **some feasible architectures** in the initial population.

Feasible architectures are **trained using backpropagation** with the **Adam optimizer** (batch size = 16, learning rate = 0.0004). Training stops **when the validation error doesn’t improve for five consecutive iterations**, preventing overfitting and adapting training time to each model’s quality.

The **objective to minimize** is the **cross-entropy loss**.

### $Cost = Categorical\_Crossentropy$

## Sequential evaluation of candidate solutions

The **initial population** in TLBO is defined by the user and includes:

- **2 predefined architectures** known to perform well (e.g., LeNet-CNN),
    
- **90% randomly generated feasible architectures**, and
    
- **a few completely random ones**, which may be **infeasible**.
    

To manage computation, **infeasible architectures** (which don’t require training) are evaluated on **CPUs**, while **feasible ones** are **trained and evaluated on GPUs**. Since feasibility is only known after evaluation, all candidates are first checked on the CPU. Feasible ones are then **queued for GPU processing**, and this CPU–GPU workflow is maintained throughout all optimization stages.

## Parallel evaluation of candidate solutions

The **evaluation process** in TLBO is **decoupled** from the optimizer and handled by an external module called the **oracle**, which evaluates candidate neural network architectures. Communication between the optimizer and oracle occurs via **text files**, and the oracle is invoked **synchronously** at each optimization stage.

The oracle operates in **two phases**:

1. **CPU Phase:**
    
    - Checks the **feasibility** of each candidate architecture.
        
    - **Infeasible solutions** receive a penalized cost and are finalized here.
        
    - The phase uses **OpenMP** for parallel processing, assigning one architecture per thread in a shared-memory model.
        
    - GPUs are not accessed in this phase to avoid conflicts among CPU threads.
        
2. **GPU Phase:**
    
    - Handles only **feasible architectures**, which require training and can vary greatly in complexity.
        
    - Uses **dynamic load balancing** through **MPI (Message Passing Interface)** to distribute workloads across multiple **GPU workers**, possibly located on different machines.
        
    - Each GPU runs as a **worker process**, while a **master process** coordinates task distribution and result collection.
        
    - The system runs under **Slurm**, which manages GPU nodes and queues.
        
    - Workers continually process new solutions until a termination signal is received.
        

This hybrid **OpenMP + MPI** approach allows efficient evaluation of large numbers of candidate neural networks across CPUs and distributed GPUs.

![[Pasted image 20251026165840.png]]

![[Pasted image 20251026170411.png]]

## Implementation details

The **TLBO implementation** was adapted from an existing **public C version** [21] and modified to **externalize the evaluation process** through the **oracle**, using **text files** for communication between the optimizer and evaluator.

The **oracle**, also written in **C**, follows an **MPI-based structure** with one **master process** and multiple **worker processes**:

- The **master process** handles input/output with the optimizer, performs the **first evaluation stage** on CPUs using **OpenMP** (for parallel feasibility checks), and delegates GPU-required evaluations to the workers.
    
- Each **worker process** is linked to a specific **GPU**, waits for assigned solutions, and evaluates them by running a **Python script** that uses **TensorFlow** to build, train, and assess the neural network.
    

This hybrid design efficiently combines **C (for control and communication)**, **OpenMP/MPI (for parallel and distributed execution)**, and **TensorFlow (for deep learning evaluation)**.

## HPC Infrastructure

The **experiments** were conducted on the **Bull cluster** of the Supercomputing-Algorithms research group at the **University of Almería (Spain)**. The setup used:

- **Two GPU nodes**, each equipped with **2 NVIDIA Tesla V100 GPUs**, **2 AMD EPYC 7302 CPUs (16 cores each)**, and **512 GB DDR4 RAM**, to serve as **GPU workers** for the oracle.
    
- **One additional node** with **2 AMD EPYC Rome 7642 CPUs (48 cores each)** and **512 GB DDR4 RAM** dedicated to running the **TLBO optimizer** and the **oracle’s master process**.
    

This configuration allowed efficient parallel and distributed evaluation of neural network architectures across high-performance computing resources.

## Evaluation of the Optimization method

The proposed TLBO-based optimizer was tested on the **CIFAR-10 dataset** to design a **compact neural network** with **fewer than 10 layers** and **under 150,000 parameters**, a key requirement for **IoT systems** with limited memory.

The optimizer was run with a **population size of 1000** and **50 cycles**. It produced a **7-layer architecture** containing **71,859 parameters** and achieving **65% top-1 accuracy**, outperforming the **human-designed baseline models** (55% and 61% accuracy).

The final architecture includes:

- 1 convolutional layer (61 filters, 5×6, stride 3×1, sigmoid activation),
    
- 1 dropout layer (62% rate),
    
- 1 average pooling layer (stride 2×4),
    
- 1 convolutional layer (82 filters, 3×2, stride 2×3, ReLU activation),
    
- 1 batch normalization layer,
    
- 2 fully connected layers (25 neurons with ReLU, 10 neurons with Softmax).
    

The **parallel and distributed evaluation scheme** is **transparent to TLBO**, enabling faster exploration of solutions and reducing design time—beneficial for this and other **population-based metaheuristics**.

## Evaluation of the performance

**Table 1** presents the execution times of three configurations used to evaluate the proposed TLBO-based optimization system:

1. **Sequential configuration:**
    
    - Uses **1 CPU thread and 1 GPU**.
        
    - Takes **6,770 seconds** to complete the first optimization cycle.
        
2. **Partially parallel configuration (OpenMP only):**
    
    - The **first stage** of the oracle (CPU feasibility check) is parallelized using **96 threads (OpenMP)**, while the **second stage** still uses **a single GPU**.
        
    - The CPU stage time drops from **630 seconds to 7 seconds**, achieving a **speedup of ~90×**, close to the ideal 96×.
        
    - The **overall speedup** (considering both CPU and GPU stages) is **1.10×**, consistent with **Amdahl’s law** due to the dominance of GPU computation time.
        
3. **Fully parallel configuration (OpenMP + MPI):**
    
    - Uses **96 CPU threads** for the first stage and **4 GPUs** for the second.
        
    - The GPU stage time decreases to **1,603 seconds**, yielding a **3.83× GPU speedup**, close to the ideal 4×.
        
    - The **overall system speedup** is **4.20×**, very close to the **ideal 4.39×**.
        

These results confirm the **high scalability and efficiency** of the proposed hybrid **OpenMP–MPI parallelization scheme**.

![[Pasted image 20251026170748.png]]

## Conclusion

This study presents a method that applies the **Teaching-Learning-Based Optimization (TLBO)** algorithm to **automatically design neural network architectures** for specific tasks.

#### **Main Contributions**

1. **New encoding scheme:**
    
    - Converts neural network architectures into **real-valued vectors**, compatible with continuous optimizers like TLBO.
        
    - Inspired by how **multi-dimensional arrays are flattened into one dimension** in computer science.
        
    - Includes a **“disabled layer” class** to flexibly represent networks of varying depth.
        
2. **Experimental validation:**
    
    - The method was tested by designing a **CNN for CIFAR-10 image classification**.
        
    - The automatically generated architectures **outperformed human-designed ones** under the same constraints.
        
    - This suggests the **potential to replace manual neural network design** with automated optimization.
        

#### **Parallelization and Performance**

- To handle the computational load, a **heterogeneous high-performance computing (HPC)** approach was used.
    
- A dedicated **oracle module** evaluates candidate architectures in two stages:
    
    1. **Stage 1 (CPU with OpenMP):** Filters feasible architectures, achieving a **90× speedup** using **96 CPU cores**.
        
    2. **Stage 2 (GPU with MPI):** Evaluates feasible networks on **4 GPUs**, achieving a **3.83× speedup**.
        
- The **overall system speedup** is **4.2×**, close to the theoretical **4.39×** limit.
    

#### **Future Work**

- Introduce **layer grouping** to simplify architectures and reduce the search space.
    
- **Use smaller datasets** during early evaluation to shorten computation time.
    
- **Compare other optimization algorithms** within the same encoding and oracle framework.


