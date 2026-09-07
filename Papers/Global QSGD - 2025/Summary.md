The primary contribution of this paper is the design, implementation, and theoretical formalization of **Global-QSGD**, the first gradient quantization method that seamlessly integrates with Allreduce without relying on heuristics. The authors deliver this through three main technical contributions: First, they introduce two quantization variants—Standard Dithering and Exponential Dithering, the latter of which includes a novel integer-based exponential reduce function to efficiently aggregate small gradient values. Second, they establish a rigorous mathematical framework by defining a new class of algorithms called Unbiased Distributed Mean Compressors ($\mathbb{U}^{n,d}(\theta)$), proving that Global-QSGD maintains bounded variance and guarantees stable convergence. Finally, they provide a theoretical performance model alongside empirical evaluations on GPU clusters, proving that their custom reduction operations result in up to a 3.51x training speedup across various network fabrics.

## Theoretical formulation

![[Pasted image 20260907001528.png]]

### Theoretical Formulation

| Concept | Formula | Explanation |
| :--- | :--- | :--- |
| **$l_p$-norm** | $\displaystyle \left\lVert y \right\rVert_p \overset{\text{def}}{=} \left( \sum_{i=1}^d \left\lvert y_i \right\rvert^p \right)^{1/p}$ | Calculates the standard local norm for a single vector[cite: 1]. For $p=\infty$, this simply represents the maximum absolute element in the vector[cite: 1]. |
| **(q, p)-mixed norm** | $\displaystyle \left\lVert x \right\rVert_{q,p} = \left( \sum_{i=1}^n \left\lVert x_i \right\rVert_q^p \right)^{1/p}$ | Defines the "Global Norm" calculated across all participating workers to ensure uniform scaling[cite: 1]. |
| **Normalized Gradient** | $\displaystyle y_i \overset{\text{def}}{=} \frac{\left\lvert x_i \right\rvert}{\left\lVert x \right\rVert_{q,p}}$ | Forces a worker's local absolute gradients into a $[0, 1]$ range by dividing them by the shared global norm[cite: 1]. |
| **Interval Bounding** | $\displaystyle l_{u_i^j} \le (y_i)_j \le l_{u_i^j+1}$ | The logical condition used to locate the exact interval bucket where a specific normalized gradient value sits[cite: 1]. |
| **Stochastic Rounding** | $\displaystyle \xi_i(y_i)_j = \begin{cases} l_{u_i^j+1} & \text{w.p. } \frac{(y_i)_j - l_{u_i^j}}{l_{u_i^j+1} - l_{u_i^j}} \\ l_{u_i^j} & \text{otherwise} \end{cases}$ | The random rounding operator[cite: 1]. It calculates the exact probability ("w.p.") of rounding a value up to the next boundary versus rounding it down[cite: 1]. |
| **Global-QSGD Operator** | $\displaystyle \text{Global-}Q_s^{q,p}(x) \overset{\text{def}}{=} \left\lVert x \right\rVert_{q,p} \frac{1}{n} \sum_{i=1}^n \text{sign}(x_i) \circ \xi_i(y_i)$ | The master equation[cite: 1]. It averages the signed, quantized gradients across all $n$ workers and multiplies the result by the global norm to restore the original scale[cite: 1]. |

#### Step-by-Step Example

Let's run a single training step using these formulas. Assume we have 2 workers ($n=2$) using an infinity norm ($p=q=\infty$) and Standard Dithering with 2 intervals ($s=2$), giving us boundaries of **0.0, 0.5, and 1.0**.

Worker 1's gradient ($x_1$): **3.0**

Worker 2's gradient ($x_2$): **-1.0**

1. **Global Norm ($\vert{}\vert{}x\vert{}\vert{}_{q,p}$):** Using the infinity norm across all workers, the maximum absolute value is **3.0**.

2. **Normalization ($y_i$):**
    - Worker 1: $\vert{}3.0\vert{} / 3.0 = \text{1.0}$.
    - Worker 2: $\vert{}-1.0\vert{} / 3.0 = \text{0.333}$.

3. **Interval Bounding & Stochastic Rounding ($\xi_i$):**
    - Worker 1's value (**1.0**) sits perfectly on a boundary, so it remains **1.0**.
    - Worker 2's value (**0.333**) is bounded by $l_{u_i^j}=0.0$ and $l_{u_i^j+1}=0.5$. The probability of rounding up is $\frac{0.333 - 0.0}{0.5 - 0.0} = 66.6\%$. Assuming the random roll falls within this probability, it rounds up to **0.5**.

4. **Sign Restoration ($\text{sign}(x_i) \circ \xi_i(y_i)$):**
    - Worker 1 becomes **1.0**.
    - Worker 2 becomes **-0.5**.

5. **Global Operator Aggregation:** The workers sum and average these values, then multiply by the global norm.
    
    - Average: $\frac{1}{2} (1.0 - 0.5) = \text{0.25}$.
    - Final Output: $3.0 \times 0.25 = \text{0.75}$.

### Algorithm Design

![[Pasted image 20260907003524.png]]

## The core problem it's solving

Standard quantization (like QSGD) has each worker rescale its own gradient using its own local norm before rounding to low precision. That's fine on its own, but it breaks Allreduce: if worker A quantizes relative to its norm and worker B quantizes relative to a different norm, you can't just sum the two quantized integer vectors — the scales don't match. You'd have to decompress, add as full-precision numbers, and re-quantize, which kills most of the speed benefit.

Global-QSGD's fix: compute **one shared global norm** across all workers first, then have everyone quantize relative to that same scale. Now the quantized values live on a common scale, so they _can_ be summed directly as integers via Allreduce.

## Algorithm 1, step by step

**Step 1 — Global normalization (lines 1–2)** Each worker $i$ holds its local gradient $x_i$. Instead of normalizing locally, every worker computes $|x_i|_p^p$ and calls an `AllreduceSUM` across all $n$ workers, then takes the $1/p$ root:

$$|x|_p = \left(\text{AllreduceSUM}{|x_i|_p^p}\right)^{1/p}$$

This is a cheap communication step (each worker only sends one scalar), and afterward every machine holds the _identical_ global norm — this is what guarantees consistent quantization scales across workers.

**Step 2 — Quantization (lines 3–5)** Now, in parallel and with no further communication, each worker:

- normalizes its gradient by the global norm to get $y_i = |x_i| / |x|$, which lands in $[0,1]$,
- computes $\text{sign}(x_i)$,
- applies the stochastic rounding operator $\xi_i(y_i)$ from Definition 3.1 (equation 2): each coordinate of $y_i$ falls between two adjacent quantization levels $l_u \le (y_i)_j \le l_{u+1}$, and it's randomly rounded to one of the two endpoints, with probability weighted by how close it is to each — this is what keeps the compressor **unbiased** (its expectation equals the true value).

**Step 3 — Aggregation (line 9)** Because all workers quantized against the same global norm, their quantized (sign, rounded-value) pairs are now on a common scale and can be combined with a normal `Allreduce(SUM)`, then divided by $n$ to get the mean:

$$\frac{1}{n}\text{Allreduce}\big(\text{SUM}{\text{sign}(x_i), \xi_i(y_i)}\big)$$

This is the payoff of the whole design: no decompress-aggregate-recompress cycle, quantized values go straight into a standard Allreduce.

**Step X — Sparsity branch (lines 6–8, optional)** If `sparse = True`, summing full quantized vectors would waste bandwidth on mostly-zero entries. Instead, each worker sends only its non-zero indices ($\text{nnz}_i$) and their sign/value, using `Allgather` rather than `Allreduce` (since you're concatenating each worker's non-zero list rather than summing dense vectors).

![[Pasted image 20260907005326.png]]

### Quantization Interval

![[Pasted image 20260907010226.png|700]]

**Dithering (general concept)**

- Stochastic rounding rule used to quantize a normalized gradient value to one of two neighboring levels.
- Rounding probability is linear interpolation between the two levels → guarantees the quantizer is **unbiased** (expected quantized value = true value), which is the property the whole convergence proof depends on.
- Standard and exponential dithering use the _same_ rounding rule — they only differ in **where the levels are placed**.

**Standard dithering ($Global-L^{q,p}_s$)**

- Levels spaced uniformly: $l_i = (s-i)/s$, i.e. $0, 1/s, 2/s, \dots, 1$.
- Natively Allreduce-compatible: quantized values are just integers, so summing them across workers = summing integers directly (no special logic needed).
- Weakness: as gradients shrink during training, most values collapse into the same low bin → loses precision exactly when it matters most (near convergence).

**Exponential dithering ($Global-E^{q,p}_s$)**

- Levels placed at powers of two: $1, 1/2, 1/4, \dots, 0$ (denser near zero, sparser near one) — inspired by Horváth et al.'s natural compression.
- Gives roughly constant _relative_ precision at every magnitude → better accuracy for small, late-training gradients.
- Verified against Figure 1's example: quantizing −0.3 gives 20%/80% split between −0.5 and −0.25, expectation = −0.3 exactly.
- Problem: quantized values are exponents, and $2^{-a}+2^{-b} \ne 2^{-(a+b)}$ → can't sum via plain integer Allreduce.

**Why exponential dithering needs Algorithm 2**

- A custom "reduce function" (based on the Cnat operator) stochastically rounds the sum of two (sign, exponent) pairs back into a single (sign, exponent) pair — unbiased, and still in compressed format, so it can be chained through the whole tree-Allreduce without ever decompressing to full floating point.
- Implemented with only integer comparisons/boolean masks (no floating-point branching) → efficient on GPU.
- Base 2 is used specifically so the exponent aligns with the IEEE-754 FP32 exponent field.

**Cost of exponential dithering**

- Each reduction step multiplies variance by 9/8; over a tree of depth $\log n$ this compounds to $(9/8)^{\log n} \approx n^{0.17}$ — small in practice (1.6× at n=16, 3.25× at n=1024).
- Real compute overhead too: ~79× the cost of a plain integer sum (vs ~1× for standard dithering) — this is the $\omega_E$ used later in the performance model (Section 5).

**Bonus advantage: bit-width scaling with number of workers**

- Standard dithering's running sum during Allreduce can hit $n \cdot s$ → needs $\approx 1+\log(s+1)+\log(n)$ bits; infeasible for e.g. n=16 with only 4-bit budget.
- Exponential dithering only communicates the exponent, whose max value during aggregation is $s+\log(n)$ → needs $\approx 1+\log(s+1+\log n)$ bits, satisfiable even for small $s$.
- Net effect: exponential dithering's bit requirement grows as $\log(\log n)$ rather than $\log(n)$ — matters at large scale independent of the accuracy argument.

**Net trade-off (ties to Section 7 results)**: standard dithering is cheaper/faster per step; exponential dithering costs more compute and some extra variance, but empirically converges closer to the no-quantization baseline.

![[Pasted image 20260907185418.png|700]]

## Convergence Analysis

Got it! The issue is likely that Obsidian tables break when you put block-level math (`$$`) or complex `\begin{array}` inside the cells. 

Here is the exact same summary, but rewritten with **only inline math** (`$...$`) inside the table. Copy and paste this directly into Obsidian—it will render perfectly.

### The Goal of the Convergence Analysis

**To mathematically prove** that Global-QSGD doesn't break training:
1. The compressed gradient is **unbiased** (correct direction on average).
2. The compression noise has **bounded variance** (won't cause divergence).
3. **Convergence is guaranteed** with a quantifiable worst-case slowdown (typically ~1.2x more iterations) for a massive drop in communication (4x).
4. It proves a dramatically better **theoretical compression ratio** than prior methods.

### Summary Table: Key Formulas Proved & Their Purpose

| **What they proved (Formula)**                                                                 | **What it proves**                                                                                 | **Why this matters**                                                                 |
| :---------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| **Unbiasedness** <br> $\mathbf{E}[\mathcal{G}(\mathbf{x})] = \bar{\mathbf{x}}$                 | The expected quantized gradient equals the true average gradient.                                 | **No systematic bias.** The optimizer isn't pulled toward the wrong solution.        |
| **Bounded Variance** <br> $\mathbf{E}[\|\mathcal{G}(\mathbf{x}) - \bar{\mathbf{x}}\|_2^2] \leq \frac{\theta}{n}\|\mathbf{x}\|_{2,2}^2$ | The noise is strictly controlled by constant $\theta$ and shrinks as $n$ (GPUs) increases.         | **Stable convergence.** Training won't oscillate or diverge due to noisy gradients.  |
| **Reduction Property** <br> $\text{Global-}Q(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n (Q_{single}(\mathbf{x}))_i$ | Global-QSGD on $n$ GPUs = standard QSGD on one giant concatenated vector.                         | **Reuses existing math.** Allows borrowing all known proofs from standard QSGD.      |
| **Variance: Standard Dithering** <br> $\theta = \frac{\sqrt{d}}{n s}$                          | Noise decreases linearly with $n$ (GPUs) and $s$ (quantization levels).                           | Safe, but variance decays slowly (just $1/s$).                                        |
| **Variance: Exponential Dithering** <br> $\theta = \frac{1}{8n} + \frac{\sqrt{d}}{n 2^{s-1}}$  | Noise decays **exponentially** with $s$. For 8-bit ($s=255$), the second term vanishes to zero.    | **Mathematically superior.** Ultra-high precision on small gradients with almost no extra variance. |
| **Sparsity / Compression Ratio** <br> $\text{Non-zeros} \leq 2^{2s-2} + \sqrt{nd}$             | Number of numbers sent is bounded by roughly $\sqrt{n \times d}$ (GPUs $\times$ model size).      | **Huge advantage.** Standard QSGD scales as $\sqrt{n}$; Global-QSGD scales as $\sqrt{nd}$—exponentially better as models grow. |
| **Iteration Slowdown** <br> $T_{comp} \leq (1 + \theta n) \cdot T_{uncomp}$                    | Quantization increases training steps by a factor of $1 + \theta n$.                              | **The "price" of compression.** For 16 GPUs, 8-bit: only ~1.2x more steps, but 75% less communication. Net win if network is slow. |
| **Adam Compatibility** <br> $\|\mathcal{G}\|_\infty \leq n^{1/p} d^{1/q} R$                    | Global-QSGD preserves the bounded $\ell_\infty$ norm required for Adam optimizers.                | **Works in the real world.** Proves the theory applies to modern LLM/Transformer training (Adam). |

### The Final Takeaway

The paper provides a **mathematical contract**: *"Use our compressor. Your training still converges. You take at most ~20% more iterations, but communicate 75% less data. If your network is slow, this is a pure win."*

## Performance Model

Here is a bullet-point summary of the **Performance Model** (Section 5 of the paper), breaking down *why* compression helps, *when* it helps, and the mathematical rule that decides whether you get a speedup.

---

### Performance Model Summary (Key Takeaways)

- **The Core Trade-off**: Gradient compression is only useful if the **time saved on communication** is larger than the **extra time spent on quantization/dequantization**. The paper builds a mathematical model to determine exactly when Global-QSGD speeds up training.

- **Allreduce Choice**: The analysis assumes a **Tree-based Allreduce** (rather than Ring-based), because it has fewer reduction steps ($2\log(n)$ depth). Fewer steps mean less accumulated noise and lower latency overhead.

- **The Baseline Allreduce Time** (without compression):
  - Modeled as: **$2\log(n)\alpha + 2\frac{\log(n)S}{\beta} + \frac{\log(n)S}{\gamma}$**
  - Where:
    - $\alpha$ = network propagation delay (latency).
    - $S$ = gradient size in bytes.
    - $\beta$ = network bandwidth (bytes/sec).
    - $\gamma$ = computation speed (bytes/sec).

- **What Changes with Global-QSGD**:
  - **Gradient size shrinks**: 32-bit → 8-bit means $\rho = 4\times$ compression ($\hat{S} = S/4$).
  - **Quantization overhead is negligible** ($\delta = 0$): The paper argues the quantization/dequantization happens only once per Allreduce call, so it barely adds to the runtime.
  - **Computation speed changes**: The custom reduction functions have a new speed ($\hat{\gamma}$), defined by an **overhead ratio** $\omega = \gamma / \hat{\gamma}$.

- **The Empirical Overhead Ratios ($\omega$)**:
  - **Standard Dithering** ($\mathcal{L}$): $\omega_S = 1$. (Native integer addition is just as fast as float32 addition).
  - **Exponential Dithering** ($\mathcal{E}$): $\omega_E = 79$. (The custom reduce function involves complex branching and exponent math, making it ~79x slower in computation per byte).

- **The "Make-or-Break" Speedup Condition (Equation 8)**:
  - After crunching the algebra, the paper derives a strict rule for when Global-QSGD beats uncompressed training:
    - **If $\omega < 4$** (Standard Dithering): Speedup is **always guaranteed** as long as bandwidth $\beta > 0$. (Because $\omega - 4$ is negative, making the inequality trivially true). 
    - **If $\omega > 4$** (Exponential Dithering): Speedup only happens if **$\beta < \frac{6\gamma}{(\omega - 4)}$**. 
  - *Translation*: Slow networks (low $\beta$) benefit from compression. Extremely fast networks (high $\beta$) might not benefit from heavy computation overhead.

- **Validating with Real Hardware (NVIDIA A100)**:
  - Computation speed: $\gamma = 2$ TB/s.
  - For Exponential Dithering ($\omega=79$), the speedup condition requires $\beta < 0.08 \times \gamma = 160$ GB/s.
  - **Measured bandwidths**:
    - NVLink (P2P): $\beta = 53.9$ GB/s ($< 160$ GB/s) → **Speedup achieved**.
    - PCIe (SHM): $\beta = 5.4$ GB/s ($< 160$ GB/s) → **Speedup achieved**.
  - *Conclusion*: Even on the fastest NVLink interconnects, exponential dithering still meets the condition for a net speedup.

---

**The Bottom Line**: The performance model proves that Standard Dithering is a "free lunch" (always speeds you up). Exponential Dithering is computationally heavier, but on *all real-world GPU interconnects* (NVLink and PCIe), the communication savings still outweigh the extra compute cost—guaranteeing a speedup in practice.

Here is a bullet-point summary of the **Implementation** and **Evaluation** sections, including the setup, benchmarks, baselines, speedup results, and generalization comparisons.

---

### Implementation Summary

- **Algorithm Variants**: Supports both **Standard Dithering** (`Global-$\mathcal{L}_s^{q,p}$`) and **Exponential Dithering** (`Global-$\mathcal{E}_s^{q,p}$`).
- **Default Configuration**: Uses the $\ell_\infty$-norm ($p=q=\infty$) and quantizes gradients to **8 bits** ($s=255$). (Theoretically compatible with any bit precision.)
- **Framework Integration**: Wrapped as a **custom hook** in PyTorch's Distributed Data Parallel (DDP) module.
  - Users simply call `model.register_comm_hook()` to enable it.
- **Granularity**: Operates at the **gradient bucket** level (default bucket size: ~25 MB).
- **Workflow per Bucket**: 3 steps — **Quantization** → **Allreduce** → **Dequantization**.
- **Custom Allreduce**: Implemented using **NCCL's point-to-point asynchronous communication API** (supports both ring and tree Allreduce).
- **Custom Reduction (Exponential Dithering)**: Uses a specialized reduce function (Algorithm 2) to handle exponential exponents, avoiding floating-point operations.
- **Performance Optimization**: Quantization, dequantization, and the custom reduction are implemented in **CUDA** to leverage GPU acceleration.
- **Sparse Support** (planned): Algorithm 1 includes an `Allgather` path to transmit only non-zero elements (not yet fully evaluated).

---

### Evaluation Setup

- **Small-Scale Hardware**:
  - **1 ASUS ESC N4A-E11 server**, Ubuntu 22.04, CUDA 11.6, PyTorch 1.13.0.
  - **4× NVIDIA A100 GPUs** (40 GB each).
- **Interconnects Tested**:
  - **P2P (NVLink)**: 4th-gen NVLink, GPU Direct, measured bandwidth = **53.9 GB/s**.
  - **SHM (PCIe)**: Host memory as middle buffer (via PCIe), measured bandwidth = **5.4 GB/s**.
- **Large-Scale Setup**:
  - **64 servers** on **Google Cloud Platform (GCP)**, each with 1 A100 GPU.
  - Network bandwidth fluctuates between **200 Mbps and 1.5 Gbps** (shared cloud network).
- **Baselines Compared**:
  - No quantization (baseline).
  - **QSGD** (standard quantization baseline).
  - **PowerSGD** (Allreduce-compatible baseline, rank=32).
  - **L-GreCo** (dynamic compression baseline).
  - *Note*: On the large-scale cloud setup, only PowerSGD was compared (Allgather-based methods like QSGD are too slow for 64 nodes).
- **Benchmark Models** (Table 3):

| Model | Dataset | Parameter Size | Training Epochs |
| :--- | :--- | :--- | :--- |
| DeepLight | Tiny Criteo | 607,959,381 | 10 |
| Wide ResNet-101-2 | MiniImageNet | 126,886,696 | 90 |
| TransformerXL | WikiText-103 | 191,950,298 | 20 |

---

### Speedup Results (Figure 4)

- **Cloud (GCP, 64 nodes)**: **Up to 3.17×** speedup over no quantization.
- **SHM (PCIe, slow interconnect)**: **Up to 3.51×** speedup.
- **P2P (NVLink, fastest interconnect)**: **Up to 1.38×** speedup.
- **Wide ResNet-101-2 with Exponential Dithering (P2P)**:
  - Only **< 2% overhead** to end-to-end training—proving Global-QSGD is robust even on models not dominated by communication.
- **Standard vs. Exponential Trade-off**: Standard dithering is faster runtime-wise; exponential dithering is slightly slower but converges closer to the no-quantization baseline (see below).

---

### Convergence & Generalization Results

- **Training Loss (Figure 3 & 5)**:
  - Global-QSGD achieves the **fastest wall-clock time** while preserving convergence.
  - **Standard dithering** achieves the quickest runtime.
  - **Exponential dithering** is slightly slower but achieves **convergence closest to no quantization**—better than QSGD and PowerSGD.
  - On GCP (Figure 5), compression delivers **greater time savings** because the network is a severe bottleneck.

- **Generalization (Validation Metrics) – Table 4**:

| Model / Metric | No Quant | Global-QSGD (Standard) | Global-QSGD (Exponential) | QSGD | PowerSGD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepLight (AUC)** | 6.75 × 10⁻¹⁸ | 6.95 × 10⁻¹⁸ | **6.87 × 10⁻¹⁸** | 6.78 × 10⁻¹⁸ | 6.76 × 10⁻¹⁸ |
| **Wide ResNet (Top-5)** | 7.06% | 8.87% | **8.87%** | 9.39% | 8.19% |
| **TransformerXL (PPL)** | 22.99 | 32.35 | **23.67** | 30.82 | 23.36 |

- **Key takeaway**: Exponential dithering **dramatically outperforms** standard dithering and other baselines in preserving model quality (e.g., TransformerXL perplexity drops from 32.35 → 23.67, nearly matching the uncompressed baseline of 22.99). All Global-QSGD variants maintain generalization comparable to no quantization.

![[Pasted image 20260907193227.png]]

### Current Limitations & Future Work (Acknowledged by the Authors)

- **Prototype overhead**: Built on NCCL's P2P API (less efficient than native NCCL). Plan to reimplement directly in CUDA with NVIDIA GPUDirect.
- **Scope**: Most evaluations are **single-node** (4 GPUs); multi-node clusters are the next target.
- **Cloud evaluation**: Did not test all baselines due to budget limits (only compared with PowerSGD).
- **Bit-width**: Only 8-bit tested; other bit configurations planned.
- **Sparsity**: The sparse Allgather path (Algorithm 1) is designed but not yet evaluated.
- **Pipelining**: Currently synchronous per step—no overlapping of compression and communication. Future optimization will overlap computation with communication to further improve efficiency.


