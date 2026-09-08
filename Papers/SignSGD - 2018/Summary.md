
### Architecture: Parameter Server

**SIGNSGD: Compressed Optimisation for Non-Convex Problems (Bernstein et al., ICML 2018)**

This paper proposes SIGNSGD, which compresses gradients by transmitting only their sign (~32x compression), and provides the first non-convex convergence proof showing it matches SGD's convergence rate despite using a biased gradient estimate. The analysis reveals that SIGNSGD works best when gradients are dense relative to noise and curvature (an ℓ1/ℓ2 geometric argument), which the authors empirically confirm holds in real deep networks. They extend the method to SIGNUM (a momentum variant related to ADAM) and to a distributed setting where a parameter server aggregates gradient signs via majority vote, proving this achieves the same variance reduction as full-precision distributed SGD while enabling 1-bit compression in both communication directions.

Experimentally, SIGNUM matches ADAM's accuracy and convergence speed on ResNet-50/ImageNet and ResNet-20/CIFAR-10, trailing well-tuned SGD by only ~2% top-1 accuracy, while requiring minimal extra hyperparameter tuning. Overall, the paper shows sign-based compression can achieve SGD-level performance with drastically reduced communication cost, supported by both theory and large-scale experiments.

![[Pasted image 20260908202116.png]]

![[Pasted image 20260908202141.png]]

## Convergence Analysis

Here's the Section 3 summary formatted for Obsidian:

---

## Section 3 — Convergence Analysis of SIGNSGD

**Goal:** Provide the first rigorous non-convex convergence guarantee for a sign-based gradient method, showing it matches SGD's convergence rate despite using a _biased_ gradient estimate.

### Assumptions

Fine-grained, coordinate-wise versions of standard SGD assumptions (recoverable as special cases of the usual scalar bounds):

- **Assumption 1 (Lower bound):** objective is bounded below by $f^*$
- **Assumption 2 (Smoothness):** coordinate-wise Lipschitz constants $\vec{L} := [L_1, ..., L_d]$
- **Assumption 3 (Variance bound):** coordinate-wise noise bound $\vec{\sigma} := [\sigma_1, ..., \sigma_d]$

### Main Result — Theorem 1

With learning rate $\delta_k = \dfrac{1}{\sqrt{|\vec{L}|_1 K}}$ and batch size $n_k = K$:

$$ \mathbb{E}\left[\frac{1}{K}\sum_{k=0}^{K-1}|g_k|_1\right]^2 \leq \frac{1}{\sqrt{N}}\left(\sqrt{|\vec{L}|_1}\left(f_0 - f^* + \frac{1}{2}\right) + 2|\vec{\sigma}|_1\right)^2 $$

— same $1/\sqrt{N}$ asymptotic rate as SGD ($N$ = cumulative stochastic gradient calls).

### Key Proof Idea

Bound the probability that a coordinate's sign is wrong:

$$ P[\text{sign}(\tilde{g}_{k,i}) \neq \text{sign}(g_{k,i})] \leq \frac{\sigma_{k,i}}{|g_{k,i}|} $$

- Large gradient relative to noise (high SNR) → low error probability, good progress
- Small gradient relative to noise → error probability can be high, but this only matters near a stationary point anyway
- Proof then follows a standard descent-lemma + telescoping-sum strategy, adapted to handle the bias introduced by the sign operation

### Why ℓ1 form matters — bridging to SGD comparison

Density measure used to translate between ℓ1 and ℓ2/ℓ∞ norms:

$$ \phi(\vec{v}) := \frac{|\vec{v}|_1^2}{d|\vec{v}|_2^2} $$

- $\phi(\vec{v}) = 1$ → fully dense vector
- $\phi(\vec{v}) \approx 1/d \approx 0$ → fully sparse vector

Yields two density ratios comparing SIGNSGD's bound to the standard SGD bound:

$$ R_1 := \frac{\sqrt{\phi(\vec{L})}}{\phi(g)} \qquad R_2 := \frac{\phi(\vec{\sigma})}{\phi(g)} $$

**Regimes:**

|Regime|Condition|Implication|
|---|---|---|
|(I)|$R_1 \gg 1$ and $R_2 \gg 1$|curvature & noise much denser than gradient → SGD favored|
|(II)|NOT $R_1 \gg 1$ and NOT $R_2 \gg 1$|SIGNSGD matches/beats SGD + gets compression|
|(III)|mixed (e.g. $R_1 \ll 1$, $R_2 \gg 1$)|indeterminate|

### Why it matters

Turns a heuristically popular algorithm (RPROP/ADAM lineage) into one with a **non-vacuous provable guarantee**, and analytically identifies _when_ SIGNSGD should match or beat SGD: when gradients are as dense or denser than curvature and noise. This motivates the empirical density measurements in **Figure 1** and **Figure A.2**, where real networks are shown to sit in regime (II)/(III).

## Majority Vote

**Goal:** Extend SIGNSGD to distributed training, where a parameter server aggregates gradient signs from multiple workers, enabling **1-bit compression in both directions** of communication (worker→server and server→worker).

### Majority Vote Scheme

Each worker sends only the sign of its local gradient; the server takes a majority vote and broadcasts just the 1-bit decision back:

$$ x_{k+1} = x_k - \delta,\text{sign}\left[\sum_{m=1}^{M}\text{sign}(\tilde{g}_m)\right] $$

### Main Result — Theorem 2

With $M$ workers:

- **(a)** Majority vote converges **at least as fast** as single-machine SIGNSGD.
- **(b)** Under the added assumption of unimodal, symmetric gradient noise (e.g. Gaussian), it converges at an **improved rate**, with the noise term shrinking by $1/\sqrt{M}$:

$$ \mathbb{E}\left[\frac{1}{K}\sum_{k=0}^{K-1}|g_k|_1\right]^2 \leq \frac{1}{\sqrt{N}}\left[\sqrt{|\vec{L}|_1}\left(f_0-f_*+\frac{1}{2}\right) + \frac{2}{\sqrt{M}}|\vec{\sigma}|_1\right]^2 $$

This matches the variance-reduction rate of full-precision distributed SGD.

### Why it matters

Majority vote is validated empirically (**Figure 2**) by showing real gradient noise in ResNet-20/CIFAR-10 and ResNet-50/ImageNet is close to unimodal and symmetric, supporting the theorem's key assumption. Overall, Section 4 shows sign-based majority voting can match distributed SGD's convergence behavior while cutting worker-server communication to 1 bit each way.

## 1. Gradient & Noise Density During Training

**Figure to check: Figure 1** (main text)

|Setup detail|Value|
|---|---|
|Model / dataset|ResNet-20 on CIFAR-10, trained 160 epochs|
|Algorithms compared|SGD, SIGNUM, ADAM|
|Method|Welford's algorithm; one full data pass **per epoch** to compute exact gradient mean $g$ and std vector $\sigma$ (160 extra passes total)|
|Repetitions|3 repeats per algorithm, error bars shown|
|**Result**|Gradient and noise densities are both consistently high (dense) and appear coupled throughout training; visible jumps at epochs 80 and 120 (learning rate decay steps). This places the problem in regime (II)/(III), as predicted by theory|
![[Pasted image 20260908214233.png]]

## 2. Gradient Density Across Architectures/Datasets

**Figure to check: Figure A.2** (supplementary)

| Setup detail    | Value                                                                                                                                                                             |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Init point      | Xavier initializer                                                                                                                                                                |
| Method          | Single full data pass to compute the exact full gradient at that point                                                                                                            |
| Models/datasets | MNIST (logistic regression), MNIST (LeNet variant), CIFAR-10 (ResNet-20), ImageNet (ResNet-50)                                                                                    |
| **Result**      | Gradients are remarkably dense across **all** tested datasets/architectures, regardless of model size — supports generality of the density finding beyond just ResNet-20/CIFAR-10 |

![[Pasted image 20260908214505.png]]

## 3. Shape of Stochastic Gradient Noise

**Figure to check: Figure 2** (main text)

|Setup detail|Value|
|---|---|
|Models/datasets|ResNet-20 on CIFAR-10 (epoch 50, batch size 128); ResNet-50 on ImageNet (epoch 50, batch size 256)|
|Algorithms compared|SGD, SIGNUM, ADAM|
|Sampling|One randomly chosen parameter per plot (not cherry-picked)|
|**Result**|All noise distributions are unimodal and approximately symmetric; at batch size 256 on ImageNet the distribution already looks Gaussian (CLT visibly kicking in) — empirically supports the unimodal/symmetric noise assumption behind Theorem 2(b)|

![[Pasted image 20260908214302.png]]

## 4. SIGNUM on ImageNet — Main Practical Benchmark

**Figure to check: Figure 3** (main text)

| Setup detail        | Value                                                                                                                                 |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Model / dataset     | ResNet-50 v2 on ImageNet                                                                                                              |
| Implementation base | Open-source code (github.com/tornadomeet)                                                                                             |
| Algorithms compared | SGD (tuned weight decay), SIGNUM, ADAM, SGD without weight decay                                                                      |
| Hyperparameters     | Learning rate & weight decay tuned on a held-out validation split; all other hyperparameters kept at SGD-community-favorable defaults |
| Note                | Data augmentation switched off at epoch 95 → visible accuracy jump                                                                    |
| **Result**          | SIGNUM ≈ ADAM in test accuracy; SIGNUM beats SGD without weight decay; SIGNUM is ~2% worse than well-tuned SGD                        |
**
![[Pasted image 20260908214350.png]]
## 5. SIGNUM Hyperparameter Sweep on CIFAR-10

**Figures to check: Figure A.3 (final runs) + Figure A.4 (full grid search)** (supplementary)

|Setup detail|Value|
|---|---|
|Model / dataset|ResNet-20 on CIFAR-10, split 45k/5k/10k (train/val/test)|
|Algorithms compared|SGD, SIGNUM, ADAM|
|Grid search|Initial learning rate × weight decay × momentum ∈ {0.0, 0.5, 0.9}; validation accuracy heatmaps in Fig. A.4; best config per algorithm re-run for final test in Fig. A.3|
|**Result**|All algorithms reach close to the 91.25% baseline (He et al., 2016a); SIGNUM's heatmap shape closely resembles ADAM's (algorithmic similarity); SGD has a larger region of very high-scoring configs, while SIGNUM/ADAM are more stable across a wider learning-rate range; little difference in final test accuracy across methods|
![[Pasted image 20260908214537.png]]