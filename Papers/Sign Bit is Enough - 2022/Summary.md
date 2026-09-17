The paper proposes a Multi Hop All reduce method for 1 bit signs without the need for extra bits like $\large \lceil log_2(M) \rceil$ bits per sign like in all reduce compatible SSDM approach. The paper uses an approach that does not required the flow of receive -> decompress -> aggregate -> recompress, which will cause the error to keep increasing from a node to another. It also leads to more time consumption because of the calculation overhead. as shown in the figure below:

![[Pasted image 20260917194334.png]]


**MARSIT stands for Multi-hop All-reduce using Sign-Bit**, and its main algorithms are below:

![[Pasted image 20260917194444.png]]

![[Pasted image 20260917194502.png]]

The parameters used in the algorithm are below:

Here's a consolidated table of every variable appearing in Algorithm 1 (Marsit, the one-bit synchronization primitive) and Algorithm 2 (Marsit-driven SGD, the outer training loop that calls it):

| Symbol                     | Meaning                                                                                                                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\large M$                 | Total number of workers (nodes/GPUs) in the multi-hop all-reduce ring                                                                             |
| $\large m$                 | Index of the current (local) worker, $\large m \in {1,\dots,M}$                                                                                   |
| $\large t$                 | Current synchronization / training round index                                                                                                    |
| $\large T$                 | Total number of global synchronization rounds (training horizon)                                                                                  |
| $\large K$                 | Interval, in rounds, between full-precision (32-bit) synchronizations                                                                             |
| $\large i$                 | Index of one of the $\large M$ segments a gradient vector is split into                                                                           |
| $\large \eta_l$            | Local stepsize applied to each worker's stochastic gradient                                                                                       |
| $\large \eta_s$            | Global stepsize applied to the aggregated sign vector                                                                                             |
| $\large \tilde{x}_t$       | Global model parameters at round $\large t$                                                                                                       |
| $\large \xi_k^{(m)}$       | A stochastic data sample drawn from worker $\large m$'s local distribution $\large \mathcal{D}_m$                                                 |
| $\large g_t^{(m)}$         | Worker $\large m$'s (local, stepsize-scaled) stochastic gradient at round $\large t$; in Alg. 1 it's overwritten to include the compensation term |
| $\large g_{t,i}^{(m)}$     | The $\large i$-th of the $\large M$ partitions of worker $\large m$'s gradient $\large g_t^{(m)}$                                                 |
| $\large c_t^{(m)}$         | Worker $\large m$'s local compensation (error-feedback) vector at round $\large t$                                                                |
| $\large \text{sgn}(\cdot)$ | Element-wise sign function                                                                                                                        |
| $\large v_i^{*}$           | Locally computed sign vector for segment $\large i$: $\large v_i^{*} = \text{sgn}\left(g_{t,i}^{(m)}\right)$                                      |
| $\large v_i$               | The received/in-transit sign vector for segment $\large i$, updated hop-by-hop as it circulates the ring                                          |
| $\large v$                 | Transient Bernoulli-sampled vector used to break ties when $\large v_i$ and $\large v_i^{*}$ disagree (defined in §4.1.1, feeds into line 6)      |
| $\large \odot$             | Marsit's bit-wise aggregation operator combining $\large v_i$ and $\large v_i^{*}$                                                                |
| $\large g_t$               | The aggregated global update at round $\large t$ — output of Alg. 1, consumed by Alg. 2's parameter update                                        |
| $\large \tilde{x}_T$       | Final trained model returned after $\large T$ rounds                                                                                              |

The paper uses error feedback ($\large c_t$) to compensate for error, and proposes the following bitwise operator as its main contribution:

$$\Large b_j = \begin{cases} (m-1)/m & v_{i,j}^{*} = 0 \\ 1/m & v_{i,j}^{*} = 1 \end{cases} \implies v_j = \begin{cases} 1 & pr = b_j \\ 0 & \text{Otherwise} \end{cases}$$
and can be expressed as:

$$\Large v_i \odot v_i^{*} = (v_i \text{ AND } v_i^{*}) \text{ OR } (v_i \text{ XOR } v_i^{*} \text{ AND } v)$$

The paper also uses the pricinple of flushing where after every $\large K$ iterations the model uses full precision 32 bit gradients and resets $\large c_t$ to $\large 0$.

![[Pasted image 20260917195503.png]]

# Experimental setup and results

## Experimental Hardware Setup

- Run on **Huawei Cloud**, on a cluster of **32 nodes**, each with **2 Nvidia T4 GPUs**
- Distributed training implemented with the **PyTorch distributed computing package**
- Two multi-hop all-reduce (MAR) topologies tested: **RAR** (ring all-reduce) and **TAR** (2D-torus all-reduce)
- Datasets/models: CIFAR-10 (AlexNet, ResNet-20), ImageNet (ResNet-18, ResNet-50), IMDb reviews (DistilBERT)
- Optimizer: **Momentum** for image classification, **Adam** for sentiment analysis (DistilBERT)
- Baselines compared against: PSGD (32-bit, uncompressed), signSGD (majority vote), EF-signSGD, SSDM, plus their own **Marsit-100** variant (K=100 full-precision resets) alongside plain **Marsit** (no periodic reset)

## Main Results

**Accuracy comparison (Table 2 — described in prose)**

- Compared to uncompressed PSGD, existing compression baselines show a noticeable accuracy drop across both image classification and sentiment analysis
- signSGD specifically loses up to ~5% accuracy vs. PSGD
- Marsit-100 and/or Marsit outperform the other compression baselines and land close to PSGD's accuracy
- On CIFAR-10, Marsit-100 (with periodic full-precision resets) beats plain Marsit (no resets); on ImageNet and IMDb the two don't differ much

![[Pasted image 20260917195556.png|700]]

**Time-to-accuracy (Figure 4a, ResNet-50/ImageNet)**

- PSGD (uncompressed) takes the longest to reach a given accuracy
- Marsit achieves a **~1.5x speedup** to reach comparable accuracy

**Communication efficiency (Figure 4b, ResNet-50/ImageNet)**

- Marsit needs **~90% less communication** than PSGD
- Marsit needs **~70% less communication** than existing signSGD-style baselines
- At matched communication budgets, Marsit and Marsit-100 consistently reach higher accuracy than the other baselines — the other signSGD methods plateau around **~50% accuracy** when Marsit/Marsit-100 have already converged

![[Pasted image 20260917200312.png]]

**Performance under different MAR topologies (Figure 5, AlexNet/CIFAR-10, TAR vs. RAR)**

- Per-round time is broken into computation, compression, and communication phases
- Marsit adds only minor compression overhead
- Marsit and/or Marsit-100 spend the least time in communication among all six methods
- Under TAR, every baseline's communication time is relatively small
- Under RAR, communication time dominates computation time, and this is exactly where Marsit's savings matter most — it needs less time between successive synchronizations than the baselines

![[Pasted image 20260917200320.png]]

**Overall (Conclusion)**

- Marsit matches non-compressed training's accuracy while cutting training time by **up to 35%**

