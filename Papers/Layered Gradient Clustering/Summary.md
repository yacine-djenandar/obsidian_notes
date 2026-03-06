
## Title: Accelerating Distributed DNN Training Efficiency via Layered Gradient Clustering
## System overview

the system is based on Data parallelism + Parameter server architecture:

![[Pasted image 20260227161846.png]]

The steps are as follows:

1. The PS initializes the initial params $w_0$ and sends them to workers
2. Workers training for $T\ iterations$, after finishing, they generate and the gradient $g_t^n$ where $n$ is the worker and $t$ is the iteration
3. After the gradinents are generated, clustering begins, which is done per layer, and automatically determines the appropriate number of clusters needed, and generates $\tilde{g_t^n}$ which is the clustering results, containing **the cluster id and the centroid of each cluster**. this cluster is then sent to the PS and wait for it to perform the aggregation
4. The PS decodes $\tilde{g_t^n}$ and performs gradient aggregation, then the new generated parameters after gradient descent are sent to all workers.
5. After the new parameters get received by workers, they update their local models by the new paramters and proceed to perform another training iteration.
6. Repeat 2 - 5 until training is done

## Layer-Wise Gradient Clustering Method

Clustering all gradients of all layers may cause accuracy drop, severe info loss, because different layers have different sizes, and therefore different impact on clustering, so the proposition is using **MiniBatch K-Means** per layer:

### $$J = \sum_{i=1}^K \sum_{j=1}^G \|u_j^l - c_i\|_2^2$$

where $G$ is the number of gradients per layer, $K$ is the number of clusters found in that layer, $u_j^l$ is the $j^{th}$ gradient of the $l$ layer, and $c_i$ is the centroid of the $i^{th}$ cluster

This formula calculates the total distance between gradients of layer $l$ and the centers of clusters. **K Means aims at minimizing this result**

## Efficiency-Aware Tensor Compression Scheme

Clustering Time can become an overhead in some cases, that is why, we need to make sure that clustering time is not before performing it, meaning that clustering needs to satisfy this formula:
## $$T_{cluster}\ <\ T_{full} - T_{spr}$$

where $T_{cluster}$ is the clustering time, $T_{full}$ is the communication time without clustering, and $T_{spr}$ is the communication time with clustering.

The time required for normal gradients communication is:

$$T_{full} = \frac{M_0}{BW} = \frac{32PL}{BW}$$
where 32 is the size of the float, $P$ is the gradients per layer, $L$ is the number of layer, and $BW$ is the bandwidth.

Time required for LGC(layered gradient clustering) communication is:

$$T_{spr} = \frac{M_1}{BW} = \frac{L(P + 32K)}{BW}$$
where $K$ is the number of clusters.

### Layer-Wise Adaptive Gradient Sensing Method

Determining the best $K$ per Layer in Hierarchical gradients clustering is the main issue, since different layers dimensions require different $K$. The paper proposes **Layer - wise adaptive gradient sensing method (LAGS)**. in which sensitivity is measured per each layer using a proxy dataset $D$. and sensitive layers will use bigger values for $K$ or no clustering. and less sensitive layers can use a smaller $K$ and be clustered more aggressively.

$$S_l(K) = \sum_{x \in D} D_{KL} \left( O(x, g_l) \parallel O(x, g_l(K)) \right)$$
above is the formula to calculate the sensitivity of a layer $l$. $O$ is the output of that layer, and $D_{KL}$ is the **KL Divergence function**

After getting the sensitivity of each layer for different $K$ values, we need to find the $K$ per layer that minimizes the overall sensitivity

$$\min S = \sum_{l=1}^{L} S_l(K_l), \quad \forall K \in \mathbb{N}$$

### Pipeline of Clutering and Communication Method

![[Pasted image 20260227174618.png]]

We calculate the next optimal $K_{n+1}$ in the $S_n$ communication phase

### Experiment results

- **Hardware**: PS with NVIDIA GTX 1070 Ti; two servers each with NVIDIA A6000; 10Gbps bandwidth
- **Implementation**: LGC in PyTorch 2.2.1 and CUDA 12.2 with TCP communication
- **Models**: VGGNet-16, ResNet-18, ResNet-101
- **Datasets**: CIFAR10, CIFAR100, Tiny-ImageNet
- **Baselines**: 
  - BSP (no optimizations)
  - STL-SGD (stage-wise local SGD)
  - DGC and RedSync (gradient sparsification)
- **Training**: All methods trained for 100 epochs for fair comparison


![[Pasted image 20260227175440.png]]

![[Pasted image 20260227175510.png]]


- **Datasets**: CIFAR-10, CIFAR-100, Tiny-ImageNet
- **Methods compared**: LGC vs. BSP, DGC, STL-SGD, RedSync (results in Tables I and II)
- **Key observations**:
  - LGC significantly reduces communication traffic and training time
  - LGC maintains accuracy comparable to baseline BSP
- **Performance vs. BSP**:
  - ResNet-18/VGGNet-16 on CIFAR10/100: 40-72x less traffic, 56.54%-86.67% faster training
  - ResNet-101 on Tiny-ImageNet: 132x less traffic, 43.87% faster training
- **Performance vs. other methods** (DGC, STL-SGD, RedSync):
  - ResNet-18/VGGNet-16 on CIFAR10/100: 3-30x less traffic, 18.49%-60.68% faster training
  - ResNet-101 on Tiny-ImageNet: 20-105x less traffic, 29.57%-40.71% faster training
- **Why LGC works**:
  - Local training + hierarchical gradient clustering reduce communication frequency and traffic
  - Max compression ratio: 105x
  - Fine-grained clustering minimizes accuracy loss
  - Added computation time < saved communication time
