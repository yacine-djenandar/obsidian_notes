### Title: 1-Bit Stochastic Gradient Descent and its Application to Data-Parallel Distributed Training of Speech DNNs

### Release year: 2014

#### Data parallel distributed SGD

In typical minibatch form, Back Propagation can be written as

## $$
\begin{align}
\lambda(t + N) &= \lambda(t) + \epsilon(t) \cdot G(t) \tag{1} \\
G(t) &= \sum_{\tau=t}^{t+N-1} \left. \frac{\partial \mathcal{F}_\lambda(o(\tau))}{\partial \lambda} \right|_{\lambda=\lambda(t)} \tag{2}
\end{align}
$$

| Parameter | Meaning                                                                 |
| --------- | ----------------------------------------------------------------------- |
| `λ(t)`    | The model at "current" sample index `t`                                 |
| `t`       | Sample index (increases in steps of the minibatch size `N`)             |
| `N`       | Minibatch size (e.g., 1024)                                             |
| `F_λ`     | The partial gradient of the objective function for sample vector `o(τ)` |
| `o(τ)`    | Sample vector at time `τ`                                               |
| `τ`       | Index variable for summation over samples in the minibatch              |
| `ε(t)`    | Learning rate (variable as training progresses)                         |
Eq. (2) can be parallelized by splitting the sum over compute nodes: Each node processes part of each minibatch—data parallelism—and the sub-minibatch gradients are then summed up over all nodes. 

The optimal number of nodes $\hat{K}$ is the one that satisfies
### $$T_{calc}(\hat{K}) = T_{comm}(\hat{K})$$
This guarantess that we will not have any bottleneck between communication and computation

> [!NOTE]
> if $K$ is too big -> calculation time drops, but more data needs to be transferred
> if $K$ is too small -> communication time drops, but more calculations per node are required

the optimal number of $K$ can be calculated with this formula: 

## $$
\hat{K} = \frac{N/2 \cdot T_{\text{calc}}^{\text{frm}} + C \cdot T_{\text{calc}}^{\text{post}}}{\frac{1}{Z} \cdot T_{\text{comm}}^{\text{float}} - T_{\text{calc}}^{\text{upd}}}
$$
where

| Parameter                        | Meaning                                                           | Notes                                                         |
| -------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------- |
| $\hat{K}$                        | Optimal number of compute nodes (GPUs/servers)                    | Balances computation and communication                        |
| $N$                              | Minibatch size                                                    | Number of training samples per minibatch                      |
| $T_{\text{calc}}^{\text{frm}}$   | Computation time per **frame** (single sample)                    | Dominated by matrix multiplications; scales down with $K$     |
| $C$                              | Number of gradient post‑processing steps                          | e.g., momentum, AdaGrad accumulation (typical $C = 2$ or $3$) |
| $T_{\text{calc}}^{\text{post}}$  | Time for **one** component‑wise post‑processing step              | Memory‑bound, partially parallelizable                        |
| $Z$                              | Compression factor                                                | $Z = 32$ for 1‑bit quantization (32‑bit → 1‑bit)              |
| $T_{\text{comm}}^{\text{float}}$ | Communication time when exchanging **uncompressed** 32‑bit floats | Depends on network bandwidth and model size                   |
| $T_{\text{calc}}^{\text{upd}}$   | Fixed time to add aggregated gradient to model (model update)     | **Not** parallelizable; memory‑bound                          |
| $N/2$ factor                     | Half‑minibatch size due to **double buffering**                   | Allows overlapping computation and communication              |
| $1/Z$ factor                     | Communication time reduction from compression                     | Smaller communication = larger denominator                    |
### Double buffering

Double buffering allows to divide each mini batch to 2, where we communicate the computed gradients from a half and at the same time we compute the gradients of the second half, **which explains the $/2$  in the forumla of $\hat{K}$**

> [!NOTE]
> all of the formulas above apply to **Aynsc SGD(Hogwild methods)** too

# 1-Bit SGD with Error Feedback

The paper aims at reducing the communication cost of data exchange by proposing a **gradient quantization method**, and **a system to exchange gradients**. 

Since this type of quantization leads to error, error feedback was used, which was inspired from the **Sigma-Delta modulation method** 

The quantization formulas are as follows:

### $$
G_{ij\ell}^{quant}(t) = Q(G_{ij\ell}(t) + \Delta_{ij\ell}(t - N))
$$
### $$
\Delta_{ij\ell}(t) = G_{ij\ell}(t) - Q^{-1}(G_{ij\ell}^{quant}(t))
$$
where

| Parameter | Meaning | Notes |
|-----------|---------|-------|
| $G_{ij\ell}^{quant}(t)$ | Quantized gradient value at minibatch $t$ | Packed integer (1‑bit per value) |
| $Q(\cdot)$ | Quantization function | Maps real number to 1‑bit (threshold 0) |
| $G_{ij\ell}(t)$ | True gradient value (full precision) at minibatch $t$ | For weight parameter $(i,j,\ell)$ (layer $i$, row $j$, column $\ell$) |
| $\Delta_{ij\ell}(t - N)$ | Quantization error from **previous** minibatch (at index $t-N$) | Carried forward and added before quantization |
| $t$ | Current minibatch index | Increases by $N$ each step |
| $N$ | Minibatch size | Number of samples per minibatch (e.g., 1024) |
| $Q^{-1}(\cdot)$ | Dequantization function | Reconstructs approximate gradient from quantized bits using per‑column means |
| $\Delta_{ij\ell}(t)$ | New quantization error for current minibatch | Stored and added to next minibatch’s gradient |
> [!NOTE]
> A quantization threshold of 0 is used, which means if a $g>0$ -> $Q=1$ else $Q=0$
> for each weight matrix column $(j,l)$ is computed two values to represent that columns, these values are recalculated in a way that minimizes the square quantization error and are transmitted in each data exchange 

> [!WARNING]
> #### Proof that the values that minimize quadratic error are the positive and negative means
> We will apply it only for the positive values, and the same logic is used for the negative values:
> 
> **We need to find $x$ that satisfies**, with $k$ is the values in a column
> ##### $$Err=\sum_{i=0}^{k}(g_i - x)^2 \ is\ minimal$$
> 
> This means that
> $$Err^{\prime}(x) = 0$$
> $$-2*(\sum_{i=0}^{k}(g_i - x))=0$$
> $$\sum_{i=0}^{k}(g_i - x)=0$$
> $$\sum_{i=0}^{k}(g_i) - \sum_{i=0}^{k}(x)=0$$
> $$\sum_{i=0}^{k}(g_i) - k\cdot x = 0$$
> $$\sum_{i=0}^{k}(g_i)=k\cdot x$$
> $$x = \frac{1}{k}\cdot\sum_{i=0}^{k}(g_i)$$
> Which confirms that it is indeed the mean

##  System Description

The authors built a data‑parallel distributed SGD system combining **1‑bit quantization**, **double buffering**, **automatic minibatch‑size selection**, **AdaGrad**, and **model parallelism**.

### Three levers for parallelizability (from Eq. 4)

- **(a) Increase minibatch size \( N \)** – but \( N \) has an upper limit beyond which convergence degrades.  
  - Every 24 h of training data, they test ~45 min of data at different \( N \) and pick the largest that does **not** hurt training‑set frame accuracy.  
  - More mature models and smaller learning rates allow larger \( N \).  
  - A decaying learning rate profile is pre‑tuned on a cross‑validation set.

- **(b) Increase compression factor \( Z \)** – achieved by 1‑bit quantization (Section 3).

- **(c) Reduce fixed cost \( $T_{\text{calc}}^{\text{upd}}$\)** – addressed via model parallelism.

### AdaGrad placement

AdaGrad can be applied in **three places**:  
1. **Locally before quantization** – risks inconsistencies across nodes.  
2. **After aggregation during data exchange** – may interfere with quantization.  
3. **After momentum smoothing** – saves memory/fixed cost but reduces AdaGrad’s effect because momentum smooths out peaks.

**Best result**: applying AdaGrad to **quantized, unsmoothed gradients** (after aggregation).

### Model parallelism

- Distributes model parameters across multiple GPUs.  
- Perfectly parallelizes component‑wise operations → reduces fixed cost \( $T_{\text{calc}}^{\text{upd}}$ \).  
- Also reduces variable cost, though with suboptimal efficiency.  
- Particularly beneficial for dual‑GPU servers.

# Experimental Results Summary (Sections 5.1 & 5.2)

## Overall Setup
- **Task**: Switchboard speech‑to‑text (309‑hour SWBD‑I training set)
- **Model**: CD‑DNN‑HMM with:
  - 7 hidden layers × 2048 units
  - Output dimension: 9304 senones
  - **46M parameters** ($M = 46M$)
- **Test set**: Hub‑5’00 (1831 utterances)
- **Hardware**:
  - Server with 8 NVidia Tesla K20Xm GPUs
  - Server farm of 24 dual‑K20Xm servers (Infiniband)

---

## 5.1 Cost Measurements

- **Sub‑batch computation time** varies with sub‑batch size (Table 1):
  - 256 → 59 ms, 512 → 89 ms, 1024 → 143 ms, 2048 → 260 ms, 4096 → 490 ms, 8192 → 955 ms
  - Per‑frame cost $T_{\text{calc}}^{\text{frm}}$ decreases from 156 µs (256) to 114 µs (8192)
- **Fixed cost** (gradient post‑processing + model update):  
  $C \cdot T_{\text{calc}}^{\text{post}} + T_{\text{calc}}^{\text{upd}} = 18.2\,\text{ms}$ (measured with momentum + AdaGrad, no data parallelism)
- **Communication cost** (compressed, $Z=32$):  
  $\frac{1}{Z} \cdot T_{\text{comm}}^{\text{float}}$ fluctuates 3–10 ms over InfiniBand (slower than expected for 8 GB/s)
- **Quantized fixed cost** $T_{\text{calc}}^{\text{upd}} \approx 9\,\text{ms}$
- **Double buffering effect**: fixed cost per minibatch becomes $2 \times T_{\text{calc}}^{\text{upd}}$ (serialized otherwise)
- **Two‑way model parallelism** cuts fixed cost in half

---

## 5.2 Effect of 1‑Bit Quantization

- **Configurations tested**:
  1. Main (sigmoid) with 1‑bit quantization + data parallelism ($K = 4$ nodes)
  2. Alternate: ReLU instead of sigmoid
  3. Alternate: SVD‑based low‑rank + low‑latency front‑end (SVD‑LL)
- **First 24h of data**: no parallelism, no quantization (cold start)
- **Results** (Table 2):
  - 1‑bit quantization works well across all setups
  - Minor but consistent drop in training‑set frame accuracy
  - Main setup: WER actually improved slightly (probably noise)
  - **Error feedback is essential** – without it, training quickly diverges
- **Double buffering**:
  - Minor impact on accuracy
  - Undoes the small WER gain on the main setup
  - No change in WER for the ReLU setup

![[Pasted image 20260531235907.png]]

![[Pasted image 20260531235920.png]]


## 5.3 When to do AdaGrad?

- AdaGrad applied to **raw gradients before momentum** (not momentum‑smoothed) → higher training frame accuracy and **$0.3$ points better WER** (momentum smoothing reduces standard deviation, diminishing AdaGrad effect)
- **Data parallelism** over $K = 4$ nodes + **2‑GPU model parallelism** per node:
  - AdaGrad applied **locally before quantization** → small WER gain, but training frame accuracy drops a little
- **Shift AdaGrad to after data aggregation** (apply to quantized gradients) → **pleasant surprise**:
  - Frame accuracy jumps $1.7$ points
  - WER drops $0.3$ points
  - Explanation (speculative): quantization distributes outliers over multiple minibatches, affecting standard‑deviation estimate less
- **Best practice**: apply AdaGrad to quantized, unsmoothed gradients after aggregation
- **End‑to‑end training time reduction**: from $35$ h to **$8.1$ h** (first $24$ h of data used no parallelism/quantization – took $22$ min alone)

## 5.4 Impact of MB‑Size Selection and Double Buffering

- **Automatic minibatch‑size selection** changed initial time from $41$ h to $35$ h (selected larger $N$)
- Doing MB‑size selection only every $72$ h instead of $24$ h → time further reduced to **$7.3$ h**, losing $0.1$ point WER
- **Double buffering (DB)** alone gives no additional speed‑up; however, DB causes selection of smaller MB sizes
- **Disabling DB only for MB‑size selection** (then enabling DB) → runtime drops to **$6.3$ h**
  - This represents a **$5.5\times$ speed‑up** from using $4 \times 2 = 8$ GPUs instead of one
- Half‑batch DB is efficient and does not change convergence
- **Table 4**: DB up to **$21\%$ faster** at $N = 32$k

## 5.5 Combination with Model Parallelism

- **Two‑way model parallelism (MP)** tested on two systems: server farm (InfiniBand) and single 8‑GPU server
- Without data parallelism, a second GPU ($1 \times 2$) is **$55\%$ efficient**
- MP **only helps** in one configuration: communication‑bound minibatch size $N = 2880$ with $16$ GPUs
- In all other settings, MP is **less efficient due to caching**
- The paper used MP throughout because earlier, less optimized code gave a different balance

## 5.6 Training a Production‑Scale Model

- **Model**: $160$M parameters, trained on $3300$ h of multi‑bandwidth data (Switchboard/Fisher + $1300$ h wideband lectures)
- Test sets: Hub‑5’00, RT03S, internal tele‑conference, three IWSLT sets
- Two data passes (GMM and DNN alignments)
- **Double buffering not used** for this large model
- Model parallelism is about **$90\%$ efficient**
- **$1 \times 1$ setups** (no parallelism) use fixed $N = 1$k; ‘$1 \times 1$ realign’ started from earlier first‑pass model with MB‑size control → average $0.1$ worse WER
- **$10 \times 2$ setup** ($20$ GPUs, conservative MB‑size control) → **$6$–$7\times$ speed‑up**
- **$20 \times 2$ setup** (aggressive MB‑size control every $72$ h) → **over $10\times$ speed‑up**, but **notable accuracy loss**

![[Pasted image 20260601000003.png]]









  