### Title: CL-SGD: Efficient Communication by Clustering and Local-SGD for Distributed Machine Learning


## CL-SGD DESIGN

#### Overview

![[Pasted image 20260307134355.png]]

The CL-SGD steps are summarized as follows:

1. The **PS(parameter server)** initializes initial parameters $w_0$ 
2. PS and workers start training the model locally for $T$ Iterations, when $T$ Iterations are done, the PS generates $g_t^*$ parameters and the worker $n$ generates $\large{g_t^n}$ parameters, for the $\large{t^{th}}$ iteration
3. Workers then Apply type-by-type gradient clustering for the  $\large{t^{th}}$ iteration $\large{\tilde{g_t}}$ and push them to the **PS**, then wait for the new updated weights from the PS $\large{w^*_{t+1}}$
4. **PS** receives and decodes the clustered gradients $\large{\tilde{g_t}}$ and use them to update the PS model parameters $\large{w^*_{t}}$ with combination of its locally generated gradients $\large{g_t^*}$
5. **PS** sends local the newly updated weights $\large{w^*_{t+1}}$ to the workers, these workers will replace their params with these ones, and resume local training
6. Repeat from 2 -> 6 until training finishes

#### CL-SGD Algorithm

![[Pasted image 20260307141423.png]]

#### Key System Design

Many types of gradients exist in neural networks depending on the architecture(***the downsized VGGNet-11 has 9 layers and 66 types of gradients, in which the first eight layers have 8 types of gradients, and the ninth layer has 2 types of gradients***). Clustering all these gradients at once can lead to significant information loss and therefore harder convergence, same type gradients are correlated based on the figure below, that is why the paper used type based gradient clustering.

![[Pasted image 20260307142843.png]]

For a type $\large{c}$ of gradients, we first initialize k cluster centers, then Minibatch-K Means is used to iterate over equation $J$ to minimize it, at the end, only cluster centers and ids are sent to the PS, equation $J$ is:
## $$J = \sum_{k=1}^K \sum_{m=1}^M \|u_m^c - r_{ck}\|_2^2$$

Where $\large{u_m^c}$ is $\large{m}$-th value of the $\large{c}$-th type of gradient, and $\large{r_{c_k}}$ is the centroid value of cluster $\large{c_k}$ which can usually be used to approximate all values in this cluster.

**The #silhouette_coefficient is used to determine the perfect value for $k$ which is the number of clusters**

![[Pasted image 20260307144636.png]]

CL-SGD uses the master-slave approach for these 3 reasons:

1. Remove the need of resource allocation for a whole other PS node
2. Remove the #gradient_staleness issue by using the actual **PS** for training too, because the PS has the same model version as the workers since it also was training the model for $T$ Iterations.
3. the **PS** has a complete, non compressed gradients, which helps reducing the accuracy damage caused by the loss of information from compression of workers.

The weights update for parameters is used as follows:

### For the PS
## $$
w_{t+1}^* = 
\begin{cases} 
w_t^* - \eta_t \left( g_t^* + \frac{\sum_{n=1}^{N} \tilde{g}_t^n}{N} \right) & \text{if } t\%T = 0 \\ 
w_t^* - \eta_t g_t^* & \text{else}
\end{cases}
$$
### For the workers
## $$
w_{t+1}^n = 
\begin{cases} 
\text{push } \tilde{g}_t^n \text{ and wait } w_{t+1}^* & \text{if } t\%T = 0 \\ 
w_t^n - \eta_t g_t^n & \text{else}
\end{cases}
$$
$\large{N}$ is the number of workers and $\large{\eta}$ is the learning rate

### Theoretical Analysis of CL-SGD Reducing Traffic

Without Compression, traffic required for communication is 
## $$M_0 = 32P$$
Where $P$ is the number of gradients.

With clustering, the traffic needed is
## $$M_{CL-SGD} = P+2*32*C$$
where $C$ is the number of gradient types

The compression Ratio is:

## $$
\Gamma_{CL-SGD} = \frac{32P}{P + 2 \cdot 32 \cdot C}
$$
which is $>> 1$ because $P >> C$ 


In comparison with **BSP(Bulk Synchronous Parallel)**, the forumla becomes the following: 
## $$
\Gamma = \frac{32 \cdot P \cdot \text{round}}{(P + 2 \cdot 32 \cdot C) \cdot \text{round}/T}
$$
where $round$ is the number of training iterations and $T$ is the number of iterations used by CL-SGD before communication

### Experiment

## Key Information Summary

### Architecture & Baseline

- **BSP (Bulk Synchronous Parallel)** without compression serves as the performance baseline

- **CL-SGD** uses a master-slave node architecture, though this architecture doesn't inherently provide communication compression

### Performance Comparison

CL-SGD is evaluated against communication compression methods:

- **Gradient compression**: ClusterGrad [10]

- **Local training**: Local-SGD [9] and STL-SGD [13]

### Experimental Settings

#### Datasets

- **CIFAR-10**: 50,000 training images, 10,000 test images

- **CIFAR-100**: 50,000 training images, 10,000 test images

- **Models trained**:

	- Downsized VGGNet-11 on both CIFAR-10 and CIFAR-100

    - Downsized ResNet-18 on CIFAR-10

#### Hardware Setup

- **OpenStack cluster** with three Sugon I840-G30 servers (8 cores, 2.1 GHz CPU each)
    
- **Node configuration**

	- Control node: 16GB RAM
	- Two compute nodes: 64GB RAM each

- **Network topology**: Virtualized k=4 fat-tree with 8 connected hosts

- **Host allocation**:

    - Parameter Server (PS): 1 host (17 VCPUs, 11GB RAM)
    - Workers: 7 hosts (4 VCPUs, 10.5GB RAM each)
    - Switches: Each with 1 VCPU, 512MB RAM
#### Hyper-parameters

- **Batch size**: b = 125 per iteration
    
- **Local iterations**: τ = (total data)/(batch size × number of clients)
    
- **Initial learning rate**: 0.9 for all algorithms
    
- **Other parameters**: Set according to original papers for BSP, Local-SGD, STL-SGD, and ClusterGrad
    
- **Training duration**: 200 epochs for all methods (based on BSP baseline convergence requirement)

## Experimental Results Summary

### Key Findings

- **CL-SGD achieves** considerable communication compression ratio (Γ) and reduced training time
    
- **Almost no accuracy loss** compared to baseline BSP across all datasets and models

### Performance Metrics

|Comparison|Compression Ratio|Training Time Improvement|
|---|---|---|
|vs BSP|1500×|51% faster|
|vs Local-SGD|50× (Local-SGD baseline)|—|
|vs STL-SGD|—|1.8% faster|
|vs ClusterGrad|—|33% faster|

---

## Three Reasons for CL-SGD's Excellent Performance

### 1. **Compression Ratio**

- **Mechanism**: Delayed communication + gradient clustering
    
- **Efficiency**: Only transmits important information for each gradient type
    
- **vs ClusterGrad**:
    
    - CL-SGD has higher per-iteration communication traffic but 20% shorter total training time
        
    - ClusterGrad suffers from high computational overhead (many cluster objects, long clustering time)
        
    - **CPU utilization**: CL-SGD (60-70%) vs ClusterGrad (75-90%) — ClusterGrad uses more complex computing
        

### 2. **Accuracy Preservation**

- **Finer clustering granularity**: Type-by-type gradient clustering reduces clustering errors
    
- **Master-slave architecture advantage**:
    
    - PS maintains complete local model parameters and uncompressed gradients
        
    - PS aggregates complete local gradients with compressed worker gradients
        
    - Updated parameters distributed to workers
        
- **Result**: High compression without accuracy loss through simultaneous training across workers
    

### 3. **Training Time Reduction**

- **Trade-off**: Extra clustering overhead offset by massive communication reduction
    
- **Quantified improvements**:
    
    - 51% faster than BSP
        
    - 1.8% faster than STL-SGD
        
    - 33% faster than ClusterGrad



![[Pasted image 20260307152643.png]]


![[Pasted image 20260307152623.png]]

![[Pasted image 20260307152812.png]]

## PRACTICAL ISSUES AND FUTURE WORK

### Selection of Hyperparameters of CL-SGD

- Cluster category(number of clusters) and $T$ are fixed, they require debugging to find their optimal values -> consider dynamic approaches to change them based on accuracy loss
### The Robustness of CL-SGD

In case of node failures, we risk having the #stragglers problem,  which results in increased training time, this can be solved by introducing asynchronous approaches as well as timeout mechanisms.

Also is new nodes join while training is running, they must have the same stage as the other nodes.



