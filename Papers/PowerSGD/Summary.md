## PowerSGD - 2019

In this paper, authors proposed PowerSGD, a matrix approximiation-based compression that is compatible with all-reduce, the contribution is used **Data Parallelism framework**.

The main algorithm is as follows:

![[Pasted image 20260403121658.png]]

**Explanation**:

The main goal is to use a small $r$ such that, given a matrix of parameters $M^{n\ *\ m}$, we generate two matrices $P^{n×r}$ and $Q^{r×m}$ so that

## $$Numel(P) + Numel(Q) << Numel(M)$$

| Step | Operation                  | Description                                                                                               |
| ---- | -------------------------- | --------------------------------------------------------------------------------------------------------- |
| 1    | $P=MQ$                     | Multiply the gradient matrix by the previous $Q$. Result $P$ is $n×r$.                                    |
| 2    | $P←ALLREDUCE\ MEAN(P)$     | Average $P$ across all workers (each worker’s $P$ is summed then divided by number of workers).           |
| 3    | $\hat{P}=ORTHOGONALIZE(P)$ | Apply #Gram_Schmidt to make the columns of $P$ orthonormal. This stabilizes the power iteration.          |
| 4    | $Q=M^⊤\hat{P}$             | Multiply the transpose of the gradient by $\hat{P}$. Result $Q$ is m×rm×r.                                |
| 5    | $Q\ ←\ ALLREDUCE\ MEAN(Q)$ | Average $Q$ across all workers.                                                                           |
| 6    | Output $(\hat{P},Q)$       | The compressed representation. Total transmitted data per worker: $nr+mr$ floats (vs. $nm$ uncompressed). |

The **decompression** is simply $\hat{P}Q^T$,  which approximates the average gradient across workers.

### Example

## PowerSGD Compression Example (Rank‑2)
**Gradient matrix** \(M\) (4×3):
$$
M = \begin{bmatrix}
1.0 & 2.0 & 3.0 \\
4.0 & 5.0 & 6.0 \\
7.0 & 8.0 & 9.0 \\
1.5 & 2.5 & 3.5
\end{bmatrix}
$$
**Initial \(Q\)** (3×2) from previous iteration (warm‑start):
$$
Q = \begin{bmatrix}
0.5 & 0.2 \\
0.3 & 0.7 \\
0.1 & 0.4
\end{bmatrix}
$$
---
### Step 1: \(P = M Q\)
$$
P = \begin{bmatrix}
1.4 & 2.8 \\
4.1 & 6.7 \\
6.8 & 10.6 \\
1.85 & 3.45
\end{bmatrix}
$$
---
### Step 2: Orthogonalize $P → \hat{P}$
$$
\hat{P} = \begin{bmatrix}
0.1692 & 0.6856 \\
0.4957 & 0.1426 \\
0.8220 & -0.3967 \\
0.2237 & 0.5938
\end{bmatrix}
$$
---
### Step 3: $Q_{\text{new}} = M^{\top} \hat{P}$
$$
Q_{\text{new}} = \begin{bmatrix}
8.2416 & -0.6302 \\
9.9522 &  0.3951 \\
11.6628 & 1.4204
\end{bmatrix}
$$
---
### Step 4: Compressed representation $(\hat{P},\; Q_{\text{new}})$
---
### Decompression: \(\hat{M} = \hat{P} Q_{\text{new}}^{\top}\)
$$
\hat{M} = \begin{bmatrix}
0.962 & 1.955 & 2.947 \\
3.995 & 4.990 & 5.985 \\
7.023 & 8.022 & 9.021 \\
1.469 & 2.461 & 3.452
\end{bmatrix}
$$
---
### Comparison with original \(M\)
Original:
$$
M = \begin{bmatrix}
1.0 & 2.0 & 3.0 \\
4.0 & 5.0 & 6.0 \\
7.0 & 8.0 & 9.0 \\
1.5 & 2.5 & 3.5
\end{bmatrix}
$$
Reconstruction error is very small (≈0.04 per entry on average).

> [!note] Compression Effect
> The compression effect becomes noticeable when we apply it on big dimension matrices, for example:
> $$n=4096,\ m=4096,\ r=8$$
> $$4096×8\ +4096×8=65,536$$
> $$4096×4096=16,777,216$$
> We can notice that $$16,777,216 >> 65,536$$ 
> and the compression ratio is $$16,777,216/65,536≈256×$$

### Usage of warm start

The contribution uses warm start, which is using the previously calculated $Q$ of $M_{t-1}$
to approximate the matrix $M_t$:

>     A single step of subspace iteration yields a factorization $M ∼ P Q^T$ with the same performance as the best rank-r approximation from an expensive **Singular Value Decomposition.**

**In the paper, the rank $r$ is used as the number of columns for $P$ and the number of rows for $Q$ -> the bigger the rank -> the more values are kept -> less compression + better accuracy, and vice-versa**

## Efficient aggregation between workers

PowerSGD has the advantage of **linearity**, which means, when having $W$ Workers:
### $$
\text{decompress}\left( \text{aggregate}\left( \text{compress}(M_1), \dots, \text{compress}(M_W) \right) \right) = \text{decompress}\left( \text{compress}\left( \frac{1}{W}\sum_{w=1}^W M_w \right) \right)
$$
which means: **we can aggregate the compressed result then decompress it, instead of decompressing each workers compress result and then aggregate, which is a huge optimization in calculation**

The advantages of PowerSGD's linearity is that **it only needs all-reduce operation without all-gather** -> we do not need to have all compressed gradients in each of the nodes for the aggregation. which will allow us for a **divide and conquer approach of reduction, following these steps for 4 workers:**

> [!info] All‑reduce steps for \(W=4\) workers

1. **Pairwise sums**  
   - Worker 1 sends $M_1$ to Worker 2.  
   - Worker 3 sends $M_3$ to Worker 4.  
   - Worker 2 computes $M_1 + M_2$.  
   - Worker 4 computes $M_3 + M_4$.
2. **Reduce to one worker**  
   - Worker 2 sends its sum $(M_1 + M_2)$ to Worker 4.  
   - Worker 4 computes the total sum:  
     $$S = (M_1 + M_2) + (M_3 + M_4) = M_1 + M_2 + M_3 + M_4$$
3. **Broadcast** (optional, depends on all‑reduce variant)  
   - Worker 4 broadcasts $S$ back to Workers 1, 2, and 3 so every worker has the full sum.

This also avoids double compression compared to having a parameter server, which required *client -> server* + *server -> client* compression. 

> [!info] All‑reduce communication time scales as $O(log\ W)$ while the all-gather time scales as $O(W)$ where $W$ is the number of workers

### Error-feedback SGD

PowerSGD uses error feedback to compensate for the difference between compressed and original results, because the compression model is **biases**, they also use post-compression momentum in the algorithm

![[Pasted image 20260403161611.png]]
## Evaluation Results

![[Pasted image 20260403162753.png]]

## Cluster specifications

- 8 nodes total.

- Each node: 2× Nvidia GeForce GTX Titan X GPU (12 GB memory per GPU).

- GPUs connected via PCIe and SMP interconnect between NUMA nodes.

- CPU: Intel Xeon E5-2680 v3 @ 2.50 GHz, 48 cores per node.   

- System memory: 251 GiB per node.

- Network: 10 Gbit/s Ethernet (SFI/SFP+), fat tree topology.

- Software: PyTorch 1.1 on Anaconda Python 3.7.

### Comparison with other compressors

- Error feedback enables many compression schemes (including biased ones) to be used.

- Evaluation metrics: test accuracy and total time per mini‑batch (forward + backward + compression + decompression + communication).

- Two compression regimes studied: medium (≈32×) and high (128×).

- At 32× compression: all schemes except Random Block achieve accuracy close to full‑precision SGD.

- At 128× compression: **only PowerSGD** reaches the target test accuracy.

- In both regimes, the only schemes faster than full‑precision SGD are **PowerSGD** and **Random Block** (both are linear and support all‑reduce).

- Random K also supports all‑reduce, but its random memory access overhead makes it slower overall than SGD.

- On modern GPU infrastructure, PowerSGD (using matrix multiplication) is faster and much more accurate than the other compression schemes.

![[Pasted image 20260403163233.png]]

### Scalability of PowerSGD

- PowerSGD scales well with increasing number of workers; the study investigates expected performance with many workers and dependence on communication backend.
   
- Benchmarked against SGD and Signum (state‑of‑the‑art distributed algorithm).
   
- Forward/backward pass time is constant across all algorithms and worker counts.

- SGD and PowerSGD use all‑reduce → gradient communication time scales gracefully with more workers.

- Signum uses all‑gather instead of all‑reduce → communication time increases more steeply (similar to PowerSGD at 4 workers, but worse at 16 workers).

- All‑reduce enables simultaneous aggregation and communication; decompression cost is independent of number of workers.

- All‑gather forces each worker to receive and decompress W compressed gradients individually → decompression time scales linearly with workers.

- Importance of reduce operation for scalability is highlighted.

- With optimized NCCL backend, all three methods scale reasonably, but Signum shows sub‑linear scaling (slope <1 in log‑log plot).

- On slower GLOO backend, PowerSGD is the only method that retains excellent scaling due to its high compression rate.

![[Pasted image 20260403163625.png]]

### Other tasks and methods

- PowerSGD compared against Signum and Spectral Atomo (state‑of‑the‑art compressed algorithms).

- Spectral Atomo is impractical: full SVD per step is too slow, and it fails to match test accuracy of others.

- Signum gives a minor speedup over SGD.

- PowerSGD is the **fastest and most accurate** among the three.

- PowerSGD’s advantage is clearest on **very large models** where communication is the bottleneck.

- LSTM language model (much larger than ResNet‑18) used as test case.

- PowerSGD required rank‑4 to match full‑precision SGD’s test score.

- On LSTM: PowerSGD reduces communication by 90% and total running time by 55%.

- Signum becomes **slower than full‑precision SGD** and gets worse test score on LSTM.

- Convergence curves (Appendix C) show time‑to‑accuracy improvements for any target accuracy.

- Appendix D provides a case study: language modeling with transformers on WikiText‑2 using 32 workers on public cloud.

![[Pasted image 20260403163912.png]]


## Language Modeling with Transformers

- Case study to assess PowerSGD’s universality and ease of tuning.

- Implemented PowerSGD in Facebook AI Research’s fairseq library.

- Trained a transformer language model (Baevski & Auli, 2019) on Google public cloud.

- All settings were new: communication infrastructure, hardware, 32 workers, model architecture.

- Required higher rank (32 vs. previous 4) to match uncompressed SGD’s validation loss.

- Possible reason: differences in architecture or cosine learning rate schedule.

- Even at rank 32, achieved time‑to‑accuracy (to loss = 5) of about 1.5× faster than SGD.

- Compression ratio: 14×.

- Performance could likely improve by re‑tuning learning‑rate hyperparameters.

![[Pasted image 20260403172146.png]]

![[Pasted image 20260403172201.png]]