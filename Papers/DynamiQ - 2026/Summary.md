
### Paper Title: DynamiQ: Accelerating Gradient Synchronization using Compressed Multi-hop All-reduce 

The paper proposes a new multihop all reduce compatible gradient quantization scheme, which aims at dividing gradients by groupd and super groups, each super group consists of multiple groups, groups contain gradient coordinates that are closer in value.

### The DynamiQ framework

The figure summarizes the process of quantization,

![[Pasted image 20260913212551.png|700]]

The process of quantization in **DynamiQ** for a gradient is summarized as follows:

1. Each worker partitions the gradient into super-groups of $S$ entries, and computes each super-group's metadata, which is the mean of the super-group $\mu_{i,j}$ and the squared $\ell_2$ norm. for $X_{i, j}$ being the $j_{th}$ super group for the $i_{th}$ worker:
$$
\Large
\mu_{i,j} = \frac{\sum_{x \in X_{i,j}} x}{|X_{i,j}|}
$$
$$ \Large F_{i,j} = \sum_{x \in X_{i,j}} x^2 $$
2. An initial all reduce is ran on the metadata to aggregate it, considered lightweight because it contains only means and norms for super groups, the aggregation is done as follows:
$$ \Large \mu_j = \frac{1}{n}\sum_{i=1}^{n}\mu_{i,j}, \qquad F_j = \sum_{i=1}^{n}F_{i,j} $$

where $\Large n$ is the number of workers.

3. Per super-group normalization and variable bit width allocation per super-group, **gradient coordiantes are reordered such as super groups with the same bit width are near each other**. Then the main all reduce is performed

4. The last step reorders the super groups to their original ordering and using their mean to get the synced gradient.

### Determining super-group bitwidths

The paper uses more bits for super-groupe with bigger norms to reduce the quantization error, variable bit width pattern is used to allow for lower quantization errors and respect bandwidth contraints; **DynamiQ** accepts a parameters $\large \bar{b}$ which it the average per-coordinate budget, that contains all necessary data for the coordiante(metadata + bits...). The paper uses powers of two for quantization values(2, 4, 8, 16).

$W=\{1,2,4,8,16\}$ and $T_{a,b}$ are the thresholds where $a$ and $b$ are consecutive in $W$, therefore if $\large F_j \in [T_{a,b}, T_{b, c})$  then all entries in the super-group $j$ are will be quantized to $\large b$ bits.

For an encoded value with $\large a$ bits, using $\large b$ bits will decrease the $MSE$ by:


$$
\Large
T_{a,b} \cdot \frac{4^{b-a}-1}{4^b \cdot (b-a)}
$$
The paper aims at proposing a threshold that the benefit is the same for all the increases, meaning: 
$$
\Large
T_{1,2} = \frac{5}{32} \cdot T_{2,4},
\qquad
T_{2,4} = \frac{17}{512} \cdot T_{4,8},
\qquad
T_{4,8} = \frac{257}{2^{17}} \cdot T_{8,16}
	$$Should all be the same. Which means that we will need to choose the optimal value of $T_{1,2}$ in a way that satisfies the constraint of $\large \bar{b}$ .  The paper uses $\large W=\{2,4,8\}$ and proposes a fast binary-search-method to find the thresholds.

### DynamiQ quantization

#### Non-uniform quantization

using $b$ bits to represent a value, 1 bit is for the sign and $b-1$ bits for values, values are positive and are defined as:

$$
\Large
Q = \left\{ f(e,r) \mid r \in \{0,\ldots,2^{b-1}-1\} \right\}
$$
and 

$$
\Large
f(\epsilon,r) =
\frac{(1+2\epsilon^2)^r - 1}
{(1+2\epsilon^2)^{2^{b-1}-1} - 1}
$$
$Q \in [0,1]$  and, $\large \epsilon>0$ determines how non-uniform the values are, closer to 0 and the distribution is more uniform, and the bigger it is, the more values are close to 0 and less big values are. 

Before that each value of the gradient $G$ needs to be normalized as $Q \in [0, 1]$, by this formula
$$\Large \frac{x}{max|G|}$$
where 
$$\Large max|G|=max\{|x|\ ,x\ \in G\}$$

#### Hierarchical quantization

the sender represents each group with $\large sf$ (scaling factor) and the pair $\large (r, \zeta)$ where $r$ is the representation and $\zeta$ is the sign, the receiver then generates using the formula:

$$\Large \zeta \cdot f(\epsilon, r) \cdot sf$$

Instead of using the scaling factor $\large sf_G=max|G|$ which requires sending a 16-bit float per group(leading to big bandiwdth overhead if the group size $s$ is small). The paper proposes to use **hierarchichal quantization** that, given  $\large sf_{\mathcal{G}} = max|\mathcal{G}|$ where $\large \mathcal{G}$ is a super group, the formula for $sf_G$ is  

$$
\Large
sf_G \in \left\{ r_G \cdot \frac{sf_{\mathcal{G}}}{255} \;\middle|\; r_G \in \{0,\ldots,255\} \right\}
$$

if we decided to represent $sf_G$ as an *UINT8*.

Therefore, having

$$\large x^\prime = \frac{x}{max|G|}$$
then this value being quantized with the formula of $Q$
$$\large \hat{x^\prime} \in Q$$

then the final quantized $\large x$ becomes

$$\Large \hat{x}=\hat{x^\prime}\cdot sf_G$$
the paper proves that individual unbiasedness remains true

#### Correlated rounding

The paper uses negative correlation to minimize quantization error, in a way that if a worker quantizes a specific partial sum upwards, another one will do the opposite(downwards)

### Main All reduce

DynamiQ uses the typical all reduce, but with compressed data, composed of **reduce-scatter phase** and the **all-gather phase**. It uses 4 type of fused kernels to do so:
1. **Compress**
2. **Decompress-Accumulate**
3. **Decompress-Accumulate-Recompress**
4. **Decompress**

### Communication and runtime overhead

The paper uses 2 all reduce phases: 

1. For the metadata, which sends 2 Float 16 values: **the mean of the super-group $\large \mu_{\mathcal{G}}$  and the Squared Norm $\large F_{\mathcal{G}}$**, which means the total communicated is $\large 32$ bits total, if $\large C_s$ is the size of the super group, the the number of bits per coordiante will be $\large 32/C_s$

2. The main stage allreduce, which uses $\large \bar{b} - 32/C_s$ bits per coordiante as the total bit budget per coordiante all included is $\large \bar{b}$ . For each group of size $C$ we will need $8/C$ bits per coordiante + 16 bits for the super-group scale for the representation.

The reordering stage incurs no communication overhead as it is done locally

For the computation overhead(see paper for this one, not much important data mentionned).

The paper's implementation aims at reducing the global memory access overhead using **fused kernels**, fused kernels allow for multiple operations per single kernel and therefore less global memory access since data used is directly accessed from the super fast registers.  it also uses RDMA to overlap communication with computation.

## Evaluation


# Section 5 — Evaluation

**Scope** — end-to-end testbed evaluation, ring all-reduce isolated (5.1) and shared (5.2), then butterfly (5.3)

### Setup

**Testbed**

|Component|Spec|
|---|---|
|Cluster|4 servers (CentOS), 8 GPUs → 4 DDP workers|
|GPU|2× RTX A6000/server, 48GB GDDR6, 768GB/s, NVLink (NV4)|
|Network|2× 100Gb/s Mellanox ConnectX-6/server (1 used)|
|CPU|2× 16-core AMD EPYC 7313/server|
|Host memory|512GB/server|
|GPU↔host|PCIe 4.0 x16 (~32GB/s/direction)|

**Workloads**

|Model|Task|Dataset|
|---|---|---|
|BERT-large|Masked LM|WikiText-103|
|Gemma 1B|Instruction tuning (Chat)|UltraChat|
|LLaMA 1B|Chat|UltraChat|
|LLaMA 1B|Reasoning|MMLU|

**Table 1 — batch/LR configuration**

| Parameter                | BERT-large MaskedLM | LLaMA 1B Chat      | Gemma 1B Chat        | LLaMA 1B MMLU      |
| ------------------------ | ------------------- | ------------------ | -------------------- | ------------------ |
| Tokens per batch         | 2048                | 3000               | 3000                 | ~1600              |
| Batch size               | 1                   | 1                  | 1                    | 4                  |
| Initial LR               | $5 \times 10^{-5}$  | $2 \times 10^{-5}$ | $1.4 \times 10^{-5}$ | $6 \times 10^{-6}$ |
| Linear LR end factor     | 1/16                | 1/8                | 1/8                  | 1/8                |
| Linear-LR iters (epochs) | 15                  | 2                  | 2                    | 2                  |
| Total iters (epochs)     | 21                  | 3                  | 3                    | 3                  |

**Baselines and fair-comparison measures**

- THC: authors' own recommended settings (4-bit local compress, 8-bit aggregate) to avoid multi-hop overflow.
- OmniReduce: momentum-based heuristic (γ=0.8) so differing per-worker local Top-k sets still converge to the target global compression ratio.
- MXFP4/MXFP6: not natively supported by the testbed GPUs → accuracy (software) and timing (equivalent-traffic transmission, no compute) are decoupled, giving these baselines a best-case estimate rather than a penalized one.

**DynamiQ's configuration** 

- Group size s=16, super-group size S=256 (16 groups)
- Per-group scale → UINT8; per-super-group scale → BF16
- W={2,4,8}, bit allocation via the Appendix A approximation algorithm
- Default budget b̄=5 bits/coordinate — justified in the Fig. 7 ablation

**Metrics** : time-to-accuracy (TTA), final accuracy, throughput (rounds/sec), and vNMSE = E[‖X−X̂‖²]/‖X‖².

---

### 5.1 Ring all-reduce (isolated)

**Results narrative** :

- DynamiQ beats MXFP8 on every workload despite using fewer bits.
- Gemma 1B: target perplexity reached **18–28% faster** than MXFP8/MXFP6 respectively; THC/OR/MXFP4 converge slower than BF16 or miss the target entirely.
- LLaMA 1B MMLU: 99%-of-BF16 accuracy (72.38%) reached **~34.5% faster than MXFP8**, **~40.8% faster than BF16**.
- DynamiQ stays within **0.1%** of BF16's final accuracy everywhere; others degrade up to **2.5%**.

**Table 2 — extra DRAM traffic (bytes/coordinate)**, A_R=(n−1)/n

|Scheme|Formula|
|---|---|
|BF16|4 + 4·A_R|
|DynamiQ|22 + 11.875·A_R|
|MXFP8|18 + 13·A_R|
|THC|74 + 2·A_R|

**Table 3 — avg. vNMSE over training** 

|Method|BERT-large|LLaMA Chat|Gemma Chat|LLaMA MMLU|
|---|---|---|---|---|
|DynamiQ|0.00217|0.00149|0.00122|0.00096|
|MXFP8|0.00591|0.00320|0.00308|0.00299|
|MXFP6|0.02332|0.01350|0.01458|0.01298|
|MXFP4|0.12080|0.11059|0.11583|0.09039|
|OR|0.15499|0.08044|0.04676|0.04530|
|THC|0.00897|0.11978|0.15168|0.19599|

**Table 4 — DynamiQ bit-budget ablation** (LLaMA MMLU & Gemma Chat) 

|Config|MMLU vNMSE|MMLU Thp.|Gemma vNMSE|Gemma Thp.|
|---|---|---|---|---|
|DynamiQ 3b|0.01603|3.051|0.02334|1.440|
|DynamiQ 4b|0.00589|2.842|0.00831|1.397|
|DynamiQ 5b|0.00096|2.604|0.00122|1.353|
|DynamiQ 6b|0.00059|2.390|0.00053|1.306|
|MXFP8|0.00299|2.123|0.00308|1.246|

**Figures:**

| Tag          | Contents                                                                                           |
| ------------ | -------------------------------------------------------------------------------------------------- |
| **Figure 4** | Bar chart, time-to-target relative to BF16, 3 target strictness levels/workload                    |
| **Figure 5** | Zoomed-in TTA curves, all 4 workloads (full: Appendix Fig. 14; error-over-steps: Appendix Fig. 18) |
| **Figure 6** | Per-round time breakdown (compute/compression/exposed comm)                                        |
| **Figure 7** | DynamiQ bit-budget ablation TTA curves (3b–6b), LLaMA MMLU                                         |
![[Pasted image 20260915234005.png|700]]

![[Pasted image 20260915234233.png]]
### 5.2 Ring all-reduce, shared network

**Setup** : 3 extra DDP processes running ring all-reduce continuously, competing for bandwidth.

**Results**:

|Workload|Advantage vs. MXFP8, isolated|Advantage vs. MXFP8, shared|
|---|---|---|
|Gemma 1B Chat|16%|21.5%|
|LLaMA 1B MMLU|34.5%|40.2%|

Exposed communication under contention is less than 4× the isolated-case time (competing jobs' transmit windows only partially overlap). 

**Figure 8**: zoomed-in TTA under contention (Gemma Chat, LLaMA MMLU); full curves in Appendix Fig. 15. 

![[Pasted image 20260915234325.png]]

---

### 5.3 Butterfly all-reduce

**Results narrative**: DynamiQ reaches 72.38% accuracy (99% of BF16) **12.0% faster than MXFP8**; grows to **37.8%** faster at the 99.5%-of-BF16 target.

**Table 5 — final accuracy & vNMSE, LLaMA MMLU, butterfly**

|Method|Accuracy (%)|vNMSE|
|---|---|---|
|BF16|73.04|0|
|DynamiQ|73.04|0.00067|
|MXFP8|72.86|0.00203|
|MXFP6|72.46|0.02008|
|MXFP4|71.59|0.17058|

**Figure 9**: zoomed-in butterfly TTA (LLaMA MMLU); full version = Appendix Fig. 16. Note: OR/THC are absent from this comparison. 

![[Pasted image 20260915234416.png]]

---

# Section 6 — Simulation Studies

### 6.1 Scalability analysis

**Setup** : n from 2→64, split across LLaMA 1B MMLU (2–8 workers) and TinyBERT+GLUE (8–64 workers); ring all-reduce throughout; THC given 12 bits (not 8) for n>8 per the original authors' overflow-prevention recommendation.

**LLaMA 1B MMLU**: vNMSE and accuracy-gap both grow with n for every method, but DynamiQ scales best, nearing BF16 accuracy even at 8 workers.  
→ **Figure 10**: (a) vNMSE vs. # workers (2,4,8); (b) accuracy gap vs. BF16, vs. # workers. 

![[Pasted image 20260915234508.png]]

**TinyBERT + GLUE**: DynamiQ keeps the lowest vNMSE of all schemes up to 64 workers, and the CE loss closest to BF16. Noted exception (attributed to small-model training variance): DynamiQ trails BF16 at n=8 and trails MXFP8 at n=16,32.  
→ **Figure 11**: (a) vNMSE vs. # workers (8,16,32,64); (b) CE-loss gap vs. # workers 

![[Pasted image 20260915234534.png]]

THC/OR show slower-than-expected vNMSE growth: THC because of its 8→12 bit bump (effective only up to n=64); OR because its error is dominated by always discarding the bottom 50% of gradients, which doesn't get worse with scale. 

Theoretical backing: a worst-case MSE bound of O(εSM²n³) for ring vs. O(εSM²n²) for butterfly — butterfly's bound is a factor of n lower.

---

### 6.2 Large-scale simulation

**Setup** : n=8,192, 32,768 super-groups of 256 coordinates/worker; budgets b̄∈{5,6,8.5}, ring and butterfly; DynamiQ allowed W={2,4,8,16}; only MXFP8-e4m3 compared (others overflow at this scale, so excluded).

**Synthetic data method** : each super-group's magnitude drawn from a mixture of two LogNormal distributions, fitted to match the empirical Fig. 1(c) distribution (exact fit parameters are in Appendix E — not reproduced here since I couldn't re-verify the specific values). Coordinates within a super-group then drawn i.i.d. N(0, M_G²/256), preserving spatial locality.

**Table 7 — vNMSE by bit budget** 

|b̄|5|6|8.5|
|---|---|---|---|
|DynamiQ-ring|4.76|0.751|0.0105|
|MXFP8-ring|—|—|6.11|
|DynamiQ-butterfly|0.0336|3.23×10⁻³|2.75×10⁻⁵|
|MXFP8-butterfly|—|—|0.0353|

**Results narrative**: ring, b̄=8.5: DynamiQ ≈582× lower vNMSE than MXFP8. Butterfly, b̄=8.5: DynamiQ ≈1283× lower. Recommended budgets at this scale: b̄=8.5 for ring, b̄=6 for butterfly.

Two attributed reasons: (1) up to 16 bits can go to rare high-norm super-groups without taxing every coordinate; (2) the decompress-accumulate-recompress step keeps re-deriving group scales as partial sums evolve, vs. MXFP8-e4m3 spending 4 of 8 bits on the exponent unconditionally. 

---

### 6.3 Parametric study

**Setup**: group size 32, dropped to 16 specifically for the hierarchical-quantization row (INT8 scale params).

**Table 6 — cumulative vNMSE reduction** 

|Configuration|LLaMA 1B Chat|LLaMA 1B MMLU|
|---|---|---|
|Uniform quantization|0.1278|0.1207|
|Non-uniform quantization|0.0707|0.0664|
|+ Variable bitwidth allocation|0.0198|0.0130|
|+ Hierarchical quantization|0.0138|0.0092|
|+ Correlated rounding|0.0091|0.0059|

**Results narrative** : full stack = 14× (Chat) / 22× (MMLU) reduction vs. uniform. Variable bitwidth allocation is the primary driver (3.5–5.1×); non-uniform quantization ≈45%, hierarchical quantization ≈30%, correlated rounding ≈35% additional reduction.  
→ **Figure 12**: per-super-group vNMSE CDFs, non-uniform vs. uniform, at 2/4/8 bits, LLaMA MMLU and Gemma Chat. 

![[Pasted image 20260915234844.png]]

# DynamiQ Worked Example — 3 Workers, Every Step Shown

> [!info] What this note does Takes one gradient vector through **the entire DynamiQ pipeline** with numbers small enough to check by hand: §3.1 statistics → §3.2 bit allocation → §3.3 quantization → §3.4 reduce-scatter with recompression → all-gather → final reconstruction.
> 
> **Two deliberate simplifications** (both mechanisms were explained in full earlier; this note is about wiring the _whole pipeline_ together):
> 
> 1. **Group = super-group** (4 coordinates each), collapsing the two-level scale encoding into one scale.
> 2. **Rounding outcomes are stated directly** rather than re-deriving the correlated-rounding draw each time — correlated rounding's benefit is statistical across many coordinates, not visible in a single instance.

---

## 0. The setup

- $\large n = 3$ workers: **W1**, **W2**, **W3**
- 8 coordinates, split into **2 super-groups of 4**:
    - **SG1** = coordinates c1–c4 → deliberately **tiny** values
    - **SG2** = coordinates c5–c8 → deliberately **large** values
- Ring topology for this chunk: $\large \text{W1 (leaf)} \rightarrow \text{W2 (internal)} \rightarrow \text{W3 (sink)}$, then all-gather broadcasts the result back out.

### Raw local gradients

|Worker|c1|c2|c3|c4|c5|c6|c7|c8|
|---|---|---|---|---|---|---|---|---|
|**W1**|0.02|−0.02|0.03|−0.03|4.0|−4.0|2.0|−2.0|
|**W2**|0.01|−0.02|0.02|−0.01|3.0|−3.0|1.0|−1.0|
|**W3**|0.03|−0.01|0.01|−0.03|5.0|−5.0|3.0|−3.0|

### The ground truth we're trying to recover

Plain column sums — this is what a perfect, uncompressed all-reduce would produce:

||c1|c2|c3|c4|c5|c6|c7|c8|
|---|---|---|---|---|---|---|---|---|
|**True sum**|0.06|−0.05|0.06|−0.07|12|−12|6|−6|

---

## 1. Obtaining super-group statistics — §3.1

Each worker computes two scalars per super-group: the mean $\mu_{i,j}$ and the squared $\ell_2$ norm $F_{i,j}$.

**W1, SG1** — squaring each entry and summing:

$$\large F_{1,1} = 0.02^2 + 0.02^2 + 0.03^2 + 0.03^2$$ $$\large F_{1,1} = 0.0004 + 0.0004 + 0.0009 + 0.0009 = 0.0026$$

**W1, SG2** — same operation, much bigger numbers:

$$\large F_{1,2} = 4^2 + 4^2 + 2^2 + 2^2 = 16 + 16 + 4 + 4 = 40$$

Doing this for all three workers:

|Worker|$F_{i,1}$ (SG1)|$F_{i,2}$ (SG2)|$\mu_{i,1}$|$\mu_{i,2}$|
|---|---|---|---|---|
|W1|0.0026|40|0|0|
|W2|0.0010|20|0|0|
|W3|0.0020|68|0|0|

The **lightweight all-reduce** now sums these across workers:

$$\large F_1 = 0.0026 + 0.0010 + 0.0020 = \mathbf{0.0056}$$

$$\large F_2 = 40 + 20 + 68 = \mathbf{128}$$

$$\large \mu_1 = \mu_2 = 0$$

> [!tip] The skew, visible immediately $$\large \frac{F_2}{F_1} = \frac{128}{0.0056} \approx 22{,}857$$ SG2 carries essentially **all** the energy of this gradient. That ratio is the entire justification for spending different numbers of bits on the two super-groups.

Every worker now subtracts $\mu_j$ from its entries (zero-mean step). Here $\mu_j = 0$, so nothing changes — but this step matters in general.

---

## 2. Determining bitwidths — §3.2

Only two bitwidths are in play, so one threshold $T_{2,4}$ decides everything. Say this round's search landed on $\large T_{2,4} = 1$:

|Super-group|$F_j$|Compare to $T_{2,4}=1$|**Bits**|
|---|---|---|---|
|SG1|0.0056|$0.0056 < 1$ → below|**2 bits**|
|SG2|128|$128 > 1$ → above|**4 bits**|

$$\large \text{average} = \frac{2+4}{2} = 3 \text{ bits per coordinate}$$

### What each bitwidth buys you

**2 bits** = 1 sign bit + 1 magnitude bit → only 2 landing points:

$$\large Q_{2\text{bit}} = {0,\ 1}$$

**4 bits** = 1 sign bit + 3 magnitude bits → 8 landing points, placed **non-uniformly** using

$$\large f(\epsilon, r) = \frac{(1+2\epsilon^2)^r - 1}{(1+2\epsilon^2)^{,2^{b-1}-1} - 1}$$

With $b=4$ (so $2^{b-1}-1 = 7$) and choosing $\epsilon$ such that $1+2\epsilon^2 = 2$:

$$\large f(r) = \frac{2^r - 1}{2^7 - 1} = \frac{2^r - 1}{127}$$

|$r$|0|1|2|3|4|5|6|7|
|---|---|---|---|---|---|---|---|---|
|**$Q[r]$**|0|0.00787|0.02362|0.05512|0.11811|0.24409|**0.49606**|**1.00000**|
|gap to next|0.0079|0.0157|0.0315|0.0630|0.1260|0.2520|0.5039|—|

> [!note] Read that bottom row The gaps **double** each step: tight resolution near zero, coarse near the top. That's the "floating-point-like" spacing — and the gap from $r{=}6$ to $r{=}7$ (0.504 wide) is the single worst place a value can land. Keep this in mind for SG2 below.

---

## 3. SG1 (2 bits) — the full journey

### Step 3a — W1 (leaf) compresses and sends

Scale is the group max:

$$\large \text{sf}_{W1} = \max(|0.02|, |0.02|, |0.03|, |0.03|) = 0.03$$

Normalize each entry by dividing by the scale:

$$\large c_1: \frac{0.02}{0.03} = 0.667 \qquad c_3: \frac{0.03}{0.03} = 1.000$$

Then stochastically round onto ${0, 1}$ — probability of landing on 1 equals the normalized value itself:

|Coord|value|$x' = \frac{\lvert x \rvert}{0.03}$|$P(\text{up})$|outcome|$r$|
|---|---|---|---|---|---|
|c1|0.02|0.667|0.667|**up**|1|
|c2|−0.02|0.667|0.667|**down**|0|
|c3|0.03|1.000|1.000|up (certain)|1|
|c4|−0.03|1.000|1.000|up (certain)|1|

**Transmitted:** signs $(+,-,+,-)$, codes $r = (1, 0, 1, 1)$, scale $= 0.03$

**W2 decodes** each as $\large \hat{x} = \varsigma \cdot Q[r] \cdot \text{sf}$:

$$\large \hat{x}_{W1} = (,0.03,\ \ 0,\ \ 0.03,\ \ -0.03,)$$

### Step 3b — W2 (internal): decompress → accumulate → **recompress**

Add W2's own local values to what it just decoded:

$$\large \text{partial} = (0.03{+}0.01,\ \ 0{+}({-}0.02),\ \ 0.03{+}0.02,\ \ -0.03{+}({-}0.01))$$ $$\large \text{partial} = (,0.04,\ \ -0.02,\ \ 0.05,\ \ -0.04,)$$

> [!important] This is the multi-hop part that breaks other schemes The scale is now re-derived **from the partial sum**, not from any worker's original data: $$\large \text{sf}_{W2} = \max(0.04, 0.02, 0.05, 0.04) = 0.05$$ The quantization range tracks the data as it grows along the path. A fixed-format scheme can't do this — which is why partial sums overflow or lose precision at each hop.

|Coord|partial|$x' = \frac{\lvert x \rvert}{0.05}$|$P(\text{up})$|outcome|$r$|
|---|---|---|---|---|---|
|c1|0.04|0.8|0.8|**up**|1|
|c2|−0.02|0.4|0.4|**down**|0|
|c3|0.05|1.0|1.0|up (certain)|1|
|c4|−0.04|0.8|0.8|**up**|1|

**Transmitted to W3:** signs $(+,-,+,-)$, $r = (1,0,1,1)$, scale $= 0.05$ **W3 decodes:** $\large (,0.05,\ \ 0,\ \ 0.05,\ \ -0.05,)$

### Step 3c — W3 (sink): decompress → accumulate, reduce-scatter ends

$$\large \text{final} = (0.05{+}0.03,\ \ 0{+}({-}0.01),\ \ 0.05{+}0.01,\ \ -0.05{+}({-}0.03))$$ $$\large \text{final} = (,0.08,\ \ -0.01,\ \ 0.06,\ \ -0.08,)$$

### Step 3d — All-gather: sink compresses once more and broadcasts

$$\large \text{sf}_{W3} = \max(0.08, 0.01, 0.06, 0.08) = 0.08$$

|Coord|final sum|$x' = \frac{\lvert x \rvert}{0.08}$|$P(\text{up})$|outcome|$r$|
|---|---|---|---|---|---|
|c1|0.08|1.000|1.000|up (certain)|1|
|c2|−0.01|0.125|0.125|**down**|0|
|c3|0.06|0.750|0.750|**up**|1|
|c4|−0.08|1.000|1.000|up (certain)|1|

Everyone decodes, then adds $\mu_1 = 0$ back:

$$\large \hat{X}_{SG1} = (,0.08,\ \ 0,\ \ 0.08,\ \ -0.08,)$$

### Step 3e — SG1 scorecard

|Coord|True|Estimate|Error|Relative error|
|---|---|---|---|---|
|c1|0.06|0.08|+0.02|33%|
|c2|−0.05|**0.00**|+0.05|**100%**|
|c3|0.06|0.08|+0.02|33%|
|c4|−0.07|−0.08|−0.01|14%|

> [!warning] What 2 bits costs c2 collapsed to **exactly zero** — the information is gone entirely. With only ${0,1}$ available, a value at 0.125 of the scale almost always rounds to nothing.

---

## 4. SG2 (4 bits) — the full journey

### Step 4a — W1 (leaf) compresses

$$\large \text{sf}_{W1} = \max(4, 4, 2, 2) = 4.0$$

Normalized: $\large c_5, c_6 \rightarrow \frac{4}{4} = 1.000$ and $\large c_7, c_8 \rightarrow \frac{2}{4} = 0.500$

**c5, c6** land exactly on $Q[7] = 1.0$ → deterministic, $r=7$, zero error.

**c7, c8** at 0.500 fall in the widest gap, between $Q[6] = 0.49606$ and $Q[7] = 1.00000$:

$$\large P(\text{up}) = \frac{0.50000 - 0.49606}{1.00000 - 0.49606} = \frac{0.00394}{0.50394} \approx 0.0078$$

Less than a 1% chance of rounding up → **both round down** to $r = 6$.

**Transmitted:** signs $(+,-,+,-)$, $r = (7,7,6,6)$, scale $= 4.0$

**W2 decodes:**

$$\large \hat{x}_{c7} = Q[6] \times 4.0 = 0.49606 \times 4.0 = 1.98425$$ $$\large \hat{x}_{W1} = (,4.0,\ \ -4.0,\ \ 1.98425,\ \ -1.98425,)$$

> [!note] The trade-off in action Non-uniform quantization gave us fine resolution near zero — but 0.5 lands in the widest gap, so c7 (true value 2.0) came back as 1.984. That's the price paid for the precision packed near zero.

### Step 4b — W2 (internal): decompress → accumulate → recompress

$$\large \text{partial}_{c5} = 4.0 + 3.0 = 7.0$$ $$\large \text{partial}_{c7} = 1.98425 + 1.0 = 2.98425$$ $$\large \text{partial} = (,7.0,\ \ -7.0,\ \ 2.98425,\ \ -2.98425,)$$

New scale from the partial sum: $\large \text{sf}_{W2} = 7.0$

c5, c6 again land exactly on $Q[7]$. For c7:

$$\large x' = \frac{2.98425}{7.0} = 0.42632$$

This falls between $Q[5] = 0.24409$ and $Q[6] = 0.49606$:

$$\large P(\text{up}) = \frac{0.42632 - 0.24409}{0.49606 - 0.24409} = \frac{0.18223}{0.25197} \approx 0.723$$

72% chance → **rounds up** to $r = 6$.

**Transmitted to W3:** $r = (7,7,6,6)$, scale $= 7.0$

**W3 decodes:** $$\large \hat{x}_{c7} = 0.49606 \times 7.0 = 3.47244$$ $$\large (,7.0,\ \ -7.0,\ \ 3.47244,\ \ -3.47244,)$$

### Step 4c — W3 (sink): decompress → accumulate

$$\large \text{final}_{c5} = 7.0 + 5.0 = 12.0$$ $$\large \text{final}_{c7} = 3.47244 + 3.0 = 6.47244$$ $$\large \text{final} = (,12.0,\ \ -12.0,\ \ 6.47244,\ \ -6.47244,)$$

### Step 4d — All-gather: compress and broadcast

$$\large \text{sf}_{W3} = 12.0$$

c5, c6 at $12/12 = 1.0$ → exactly $Q[7]$ again, zero error. For c7:

$$\large x' = \frac{6.47244}{12.0} = 0.53937$$

Between $Q[6] = 0.49606$ and $Q[7] = 1.00000$:

$$\large P(\text{up}) = \frac{0.53937 - 0.49606}{1.00000 - 0.49606} = \frac{0.04331}{0.50394} \approx 0.086$$

**Rounds down** to $r = 6$:

$$\large \hat{x}_{c7} = 0.49606 \times 12.0 = 5.95276$$ $$\large \hat{X}_{SG2} = (,12.0,\ \ -12.0,\ \ 5.95276,\ \ -5.95276,)$$

### Step 4e — SG2 scorecard

|Coord|True|Estimate|Error|Relative error|
|---|---|---|---|---|
|c5|12|12.00000|**0**|**0%**|
|c6|−12|−12.00000|**0**|**0%**|
|c7|6|5.95276|−0.04724|0.79%|
|c8|−6|−5.95276|+0.04724|0.79%|

> [!tip] What 4 bits bought Worst relative error here is **0.79%** — versus **100%** in SG1. Twice the bits, but vastly more than twice the fidelity, because the extra levels let each hop's partial sum land close to a real quantization point.

---

## 5. Everything together

|Coord|SG|Bits|True sum|DynamiQ estimate|Error|
|---|---|---|---|---|---|
|c1|SG1|2|0.06|0.08|+0.02|
|c2|SG1|2|−0.05|0.00|+0.05|
|c3|SG1|2|0.06|0.08|+0.02|
|c4|SG1|2|−0.07|−0.08|−0.01|
|c5|SG2|4|12|12.00000|0|
|c6|SG2|4|−12|−12.00000|0|
|c7|SG2|4|6|5.95276|−0.04724|
|c8|SG2|4|−6|−5.95276|+0.04724|

### Computing vNMSE — the paper's own metric

$$\large \text{vNMSE} = \frac{\mathbb{E}\big[\lVert X - \hat{X} \rVert^2\big]}{\lVert X \rVert^2}$$

**Numerator** — sum of squared errors:

$$\large \lVert X - \hat{X} \rVert^2 = 0.02^2 + 0.05^2 + 0.02^2 + 0.01^2 + 0^2 + 0^2 + 0.04724^2 + 0.04724^2$$

$$\large = 0.0004 + 0.0025 + 0.0004 + 0.0001 + 0 + 0 + 0.00223 + 0.00223 = 0.00786$$

**Denominator** — squared norm of the true gradient sum:

$$\large \lVert X \rVert^2 = 0.06^2 + 0.05^2 + 0.06^2 + 0.07^2 + 12^2 + 12^2 + 6^2 + 6^2$$

$$\large = 0.0146 + 144 + 144 + 36 + 36 = 360.0146$$

**Result:**

$$\large \text{vNMSE} = \frac{0.00786}{360.0146} \approx 2.18 \times 10^{-5}$$

> [!example] The paper's whole thesis, in one number SG1's errors are catastrophic **per-coordinate** (up to 100%). Yet the overall vNMSE is $2.18 \times 10^{-5}$ — tiny.
> 
> Why? SG1's total contribution to $\lVert X \rVert^2$ is just $\large 0.0146$ out of $\large 360.0146$ — about **0.004%**. Wrecking SG1 barely registers. Meanwhile SG2, which holds **99.996%** of the energy, got the bits it needed and came back nearly exact.
> 
> That is exactly what $F_j$ was measuring back in §3.1, and exactly why using it to steer bit allocation works: **spend bits where the energy is.** At a modest 3-bit average budget, this vector came through with error five orders of magnitude below its own magnitude.

---

## 6. What was simplified here

| Simplified in this note                | The full mechanism                                                                                                                                                                                                                                                  |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Group = super-group (one shared scale) | §3.3 **hierarchical quantization** — per-super-group BF16 max, plus a per-group 8-bit _relative_ scale. Unbiased because the two roundings use independent randomness: $\mathbb{E}[\hat{x}] = \mathbb{E}[\hat{x}'] \cdot \mathbb{E}[\widehat{\text{sf}}] = x$       |
| Rounding outcomes stated directly      | §2.4/§3.3 **correlated rounding** — shared permutation $u_i = \frac{\pi_i + \gamma_i}{n}$ guarantees exactly one worker's draw lands in each interval $[\frac{k}{n}, \frac{k+1}{n})$, so up/down outcomes spread evenly instead of clustering by chance             |
| One threshold, chosen arbitrarily      | §3.2 **threshold derivation** — the ratio chain $T_{1,2} = \frac{5}{32} T_{2,4}$, $T_{2,4} = \frac{17}{512} T_{4,8}$, $T_{4,8} = \frac{257}{2^{17}} T_{8,16}$, derived by equalizing per-bit MSE payoff, then anchored by binary search to hit the target $\bar{b}$ |