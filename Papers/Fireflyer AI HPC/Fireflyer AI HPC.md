
### year: **2024**

# Fireflyer AI HPC Hardware architecture

## Single Node Architecture

Each node has 8 Nvidia PCIe A100 GPUs and one Infiniband NIC, like in the followingcapture

![[Pasted image 20251018183956.png]]

## Network Topology

The fat-tree architecture was used, specifically the double fat tree architecture(2 zones), each zone holds 600 GPU computre nodes, the overall system has 1250 GPU compute nodes and 200 storage nodes, with 10 000 A100 GPUs, the two zones are connected with 2 X 40 ports infiniband switches

![[Pasted image 20251018184425.png]]

## Cost Performance

The architecture achieved 83% of the performance of Nvidia DGX-A100, with only 60% of GPU energy consumption and costs. this is thanks to the double fat tree topology, DGX uses three layer fat tree architecture.

![[Pasted image 20251018184736.png]]

## HFReduce

HFReduce is an all reduce operation library that uses the following steps

1. **Intra Node Reduction**: Asynchronously transfer the gradients from GPUs to CPU memory*(GDRCopy for small data and MemCpyAsync for large data)*, then **preform reduction operation using CPU SIMD instructions**
2. **Inter Node Reduction**: Use the double binary tree algorithm for inter node allreduce, and **RDMA** for inter node data transfer
3. Finally, the CPU transfers the reduced gradients to GPUs through PCIe, using GDRCopy for 3X faster transfers than MemCpyAsync because it allows **caching data in the CPU cache for faster writes to the 4 grouped GPUs for each CPU.

![[Pasted image 20251018203344.png]]

### Advatages of HFReduce

1. **Reduced PCIe Bandiwdth Consumption**: HFReduce requires only one unit of PCIe bandidth to be used, compared to NCCL that requires using $\frac{2*n - 1}{n}$ units of PCIe bidirectionnal bandwidth.
2. **No GPU Kernel Overhead**: HFReduce uses the GPU Copy Engine instead of using a kernel like NCCL, which can affect computation kernels.

The figure below shows the results of reduction of 186 MB of data using NCCL and HFReduce.
**HFReduce provides great strong scalability**

![[Pasted image 20251018204734.png]]

**HFReduce can be improved using NVLink, that starts first by reducing gradients between GPUs interconnected using NVLInk(600 GB/s) before sending them to the CPU. Then, when the CPU returns the results, it splits the results between the NVLink interconnected GPUs, then all-gather is performed via NVLink between the two GPUs for full data gathering**. *HFReduce can Reach 10 GB/s Bandwidth*

![[Pasted image 20251018205823.png]]

### Bottlenecks of HFReduce

HFReduce requires 24X the amount of original data size in a single GPU:

1) *D2H Phase requires 8 write operations.* 
2) *Intra-node Reduce Add Phase involves 8 read operations and 1 write operation.* 
3) *Inter-node Allreduce Phase: IB send demands 2 read operations, while IB receive requires 2 write operations, along with 1 read operation for reduce add.* 
4) *H2D Phase Utilizing GDRCopy can reduce this to only 2 read operations, whereas MemCopy necessitates 8 read operations.*

## HaiScale

### HaiScale DDP

HaiScale DDP is a training tool that uses HFReduce for its backend operations instead of NCCL all reduce. **Allowing for asynchronous allreduce operation $=>$ allows overlapping communication with computation in backpropagation**.
Training VGG-16 with HFReduce took **half the time** compared to Torch DDP's NCCL, with nearly 88% parallel scalability from 32 to 512 GPUs.

![[Pasted image 20251018211136.png]]
**Weak Scalability**
### LLMs Training Optimization

- **NVLink Bridge enables Tensor Parallel between PCIe GPUs**: Using NVLink allows for 600 GB/s bandwidth between each pair of GPUs, allowing for better tensor parallelism
- **Pipeline Parallelism Optimization in PCIe Architecture**: Since there is one IB NIC for each 8 GPUs, which can lead to network bandwidth contention especially in pipeline parallelism, they introduced data parallelism rank, which makes GPUs in the same node belong to different DP rank which staggers the timing of PP parallelism for each rank. 
**The following figure shows that training time LLaMa-13B decreases from 64 to 9.7 seconds acheiving a parallel efficiency of 91% **

![[Pasted image 20251018212129.png]]
**Strong Scalability**

### Fully Sharded Data Parallel (FSDP)

It' an implementation of ZeRO Stage-3 algorithm, with some specific implementations, including:
- overlapping allgather and reduce-scatter communication with forward and backward computation.
- splitting the optimization during backward propagation for enhanced overlap.
**Training GPT-2 Medium achieved 95% scalability from 16 to 128 GPUs, and reduced training time by half compared to PyTorch's FSDP**

![[Pasted image 20251018212702.png]]
**GPT-2 Medium training / weak scalability**

## ADVANCED COST-EFFECTIVE AND CO-DESIGN OPTIMIZATIONS

### Divergence of Different traffics

4 different types of traffic exist in typical training(HFReduce, NCCL, 3FS storage, other), using infiniband SL(service Level) tech., we assign to each type of traffic its SL, each SL is then mapped to IB virtual lanes(VL) to ensure traffic isolation ==> preventing network congestion caused by Head of Line (HOL) blocking and traffic collisions.

### Topology Adjustment and Route Optimization

Static routing is used in the toplogy to avoid network congestion.

### NCCL Optimization


### Network Tuning in 3FS


