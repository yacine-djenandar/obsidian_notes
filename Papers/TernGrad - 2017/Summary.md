TernGrad (Wen et al., 2017, NeurIPS) tackles the communication bottleneck in distributed SGD by quantizing gradients to just three values, {-1, 0, +1}, before sending them to a parameter server — instead of full 32-bit floats. Each worker computes a sign, a random Bernoulli mask (for unbiased stochastic rounding), and a max-magnitude scaler; a shared scaler across workers plus "parameter localization" (workers keep local parameter copies instead of pulling full-precision ones back) shrinks traffic in both directions. Layer-wise ternarizing and gradient clipping are added to preserve convergence, which the authors also prove theoretically. It uses a **parameter-server** setup, not all-reduce.

Results: no accuracy loss on AlexNet (sometimes even a small gain) and under ~2% loss on GoogLeNet (both on ImageNet), while cutting gradient communication up to ~20× — making it a simple but influential gradient-compression method for distributed deep learning.

Here's the same summary with Obsidian-safe math (using standard `$...$` inline and `$$...$$` block syntax, avoiding LaTeX macros Obsidian's MathJax often chokes on, and plain code formatting inside tables since tables don't reliably render inline LaTeX):

## Problem Formulation

- Cast as an **online learning system**: parameters `w` are updated from a stream of observations `z` (mini-batches), with loss $Q(z,w)$.
- Minimization target:

$$C(w) := \mathbb{E}{Q(z,w)}$$

- Standard update rule (GOGA — General Online Gradient Algorithm):

$$w_{t+1} = w_t - \gamma_t g_t, \quad g := \nabla_w Q(z,w)$$

- TernGrad replaces the true gradient with a **ternarized, stochastic version**:

$$w_{t+1} = w_t - \gamma_t \left( s_t \cdot \text{sign}(g_t) \circ b_t \right)$$

- Key property they must establish: this noisy, 3-level update still converges to the same minimum as full-precision SGD.

## Key Variables

|Symbol|Meaning|
|---|---|
|$w$, $w^*$|current parameters / true minimum|
|$g_t$|true (floating-point) gradient at step $t$|
|$s_t = \max(\lvert g_t \rvert)$|scaler (max absolute value of gradient)|
|$b_t$|random binary mask, $b_{tk} \sim \text{Bernoulli}(\lvert g_{tk} \rvert / s_t)$|
|$\tilde{g}_t = s_t \cdot \text{sign}(g_t) \circ b_t$|ternarized gradient sent to server|
|$\gamma_t$|learning rate at step $t$|
|$h_t = \lVert w_t - w^* \rVert^2$|squared distance to optimum|
|$A, B$|positive constants bounding the gradient|

The rest of the summary (prose + block equations) stays as-is from the previous message.

## Convergence Analysis — Main Idea

**Step 1 — Unbiasedness.** Since $\mathbb{E}{b_t \mid z_t, w_t} = |g_t| / s_t$, they show:

$$\mathbb{E}{s_t \cdot \text{sign}(g_t) \circ b_t} = \mathbb{E}{g_t} = \nabla_w C(w_t)$$

→ **the ternary gradient is an unbiased estimator** of the true gradient. This is the crux that lets convergence proofs for standard SGD carry over.

**Step 2 — Borrow classic convergence machinery (Bottou's Lemma).** Under two standard assumptions —

- _Assumption 1:_ $C(w)$ has a single minimum and $-\nabla_w C(w)$ always points toward it
- _Assumption 2:_ learning rate satisfies $\sum \gamma_t^2 < \infty$, $\sum \gamma_t = \infty$ (decays "just right")

a Lyapunov/quasi-martingale argument (Lemma 1) says: if a bound of the form

$$\mathbb{E}{h_{t+1} - (1+\gamma_t^2 B)h_t \mid X_t} \le -2\gamma_t (w_t - w^*)^T \nabla_w C(w_t) + \gamma_t^2 A$$

holds, then $w_t \to w^*$ almost surely.

**Step 3 — New gradient bound for TernGrad (Assumption 3).** To use that lemma, they need a bound on the gradient itself:

$$\mathbb{E}{\max(|g|) \cdot |g|_1} \le A + B |w - w^*|^2$$

(stronger than the standard SGD bound $\mathbb{E}{|g|^2} \le A + B|w-w^*|^2$, since $\max(|g|)\cdot|g|_1 \ge |g|^2$).

**Step 4 — Main result (Theorem 1).** Combining Assumption 3 with Lemma 1: TernGrad's update converges almost surely, i.e.

$$P\left(\lim_{t \to \infty} w_t = w^*\right) = 1$$

**Purpose of it all:**

- Proves TernGrad is theoretically safe to use (not just an empirical hack).
- Because the TernGrad bound is _stronger_ (harder to satisfy) than standard SGD's, it explains **why** vanilla ternarization can hurt large networks — `max(|g|)` can be an outlier compared to most gradient entries.
- This directly **motivates the two practical fixes**: layer-wise ternarizing and gradient clipping — both aimed at shrinking `max(|g|)` so the TernGrad bound gets closer to the standard SGD bound, which is what actually makes training stable at ImageNet scale.

## Example

Here's the example redone precisely against the paper's exact equations (Wen et al., 2017, Section 3.1, Algorithm 1), with explicit tracking of when each element becomes $-1$, $0$, or $+1$.

## Setup — Algorithm 1, Worker steps 1–2

Each worker $i \in {1,...,N}$, $N=3$, computes its local gradient $g_t^{(i)}$ on its data shard $z_t^{(i)}$:

|Worker|$g_t^{(i)}$|
|---|---|
|1|$[0.8,\ -0.3,\ 0.1,\ -0.05]$|
|2|$[0.6,\ -0.9,\ 0.2,\ 0.05]$|
|3|$[0.3,\ -0.2,\ -0.7,\ 0.4]$|

## Step 1 — Local scaler, Eq. (2)

$$s_t^{(i)} := \max\left(\text{abs}(g_t^{(i)})\right)$$

$s^{(1)}=0.8$, $s^{(2)}=0.9$, $s^{(3)}=0.7$.

## Step 2 — Scaler sharing, Eq. (4)

The paper notes (Section 3.1, paragraph immediately after Eq. 3) that if each worker ternarizes with its _own_ scaler, the server-side average is no longer ternary. To fix this it defines a **shared scaler**:

$$s_t = \max({s_t^{(i)}} : i=1...N) = \max(0.8,0.9,0.7) = 0.9$$

This $s_t=0.9$ replaces each worker's local $s_t^{(i)}$ everywhere it appears in Eq. (1) and Eq. (3).

## Step 3 — Exactly where $-1$, $0$, $+1$ come from: Eq. (1) and Eq. (3)

Eq. (1) defines the ternarized gradient elementwise as:

$$\tilde{g}_t = s_t \cdot \text{sign}(g_t) \circ b_t$$

where $\text{sign}(\cdot)$ and $\text{abs}(\cdot)$ act elementwise (defined in text right after Eq. 2), and $\circ$ is the Hadamard product. For a single coordinate $k$, this is:

$$\tilde{g}_{tk} = s_t \cdot \text{sign}(g_{tk}) \cdot b_{tk}$$

with $b_{tk}$ sampled per Eq. (3):

$$P(b_{tk}=1) = \lvert g_{tk} \rvert / s_t, \qquad P(b_{tk}=0) = 1 - \lvert g_{tk} \rvert / s_t$$

**The rule (this is the whole ternarization logic):**

- If $b_{tk}=0$ (the coin flip fails) → the ternary code is **0**, regardless of the gradient's sign. Small-magnitude gradients (relative to $s_t$) have low $P(b_{tk}=1)$, so they're _likely_ to be zeroed — this is the compression.
- If $b_{tk}=1$ and $g_{tk}>0$ → the ternary code is **$+1$**.
- If $b_{tk}=1$ and $g_{tk}<0$ → the ternary code is **$-1$**.

Applying this with shared scaler $s_t=0.9$ and one sampled realization of $b_t$ (sampling is inherently random — Eq. 3 — this is just one draw):

**Worker 1** ($\text{sign}=[+1,-1,+1,-1]$):

|$g_{1k}$|$\lvert g_{1k}\rvert/s_t$ (Eq. 3)|sampled $b_{1k}$|ternary code|why|
|---|---|---|---|---|
|$0.8$|$0.889$|$1$|$+1$|sign$+$, $b=1$|
|$-0.3$|$0.333$|$1$|$-1$|sign$-$, $b=1$|
|$0.1$|$0.111$|$0$|$0$|$b=0$ (dropped)|
|$-0.05$|$0.056$|$0$|$0$|$b=0$ (dropped)|

**Worker 2** ($\text{sign}=[+1,-1,+1,+1]$):

|$g_{2k}$|$\lvert g_{2k}\rvert/s_t$|sampled $b_{2k}$|ternary code|why|
|---|---|---|---|---|
|$0.6$|$0.667$|$1$|$+1$|sign$+$, $b=1$|
|$-0.9$|$1.0$|$1$|$-1$|sign$-$, $b=1$ (certain, since $\lvert g\rvert=s_t$)|
|$0.2$|$0.222$|$1$|$+1$|sign$+$, $b=1$|
|$0.05$|$0.056$|$0$|$0$|$b=0$ (dropped)|

**Worker 3** ($\text{sign}=[+1,-1,-1,+1]$):

|$g_{3k}$|$\lvert g_{3k}\rvert/s_t$|sampled $b_{3k}$|ternary code|why|
|---|---|---|---|---|
|$0.3$|$0.333$|$0$|$0$|$b=0$ (dropped)|
|$-0.2$|$0.222$|$0$|$0$|$b=0$ (dropped)|
|$-0.7$|$0.778$|$1$|$-1$|sign$-$, $b=1$|
|$0.4$|$0.444$|$1$|$+1$|sign$+$, $b=1$|

## Step 4 — Scaled ternary gradient sent, Eq. (1)

Multiplying the ternary codes by $s_t=0.9$ gives what's actually reconstructed/used at the server:

|Worker|$\tilde{g}_t^{(i)} = s_t\cdot\text{sign}(g_t^{(i)})\circ b_t^{(i)}$|
|---|---|
|1|$[0.9,\ -0.9,\ 0,\ 0]$|
|2|$[0.9,\ -0.9,\ 0.9,\ 0]$|
|3|$[0,\ 0,\ -0.9,\ 0.9]$|

(Only the ternary code ${-1,0,+1}$ plus one shared float $s_t$ needs to travel over the network — _Algorithm 1, Worker step 4_.)

## Step 5 — Server average, Algorithm 1 (Server, step 7)

$$g_t = \sum_i \tilde{g}_t^{(i)} / N = [0.6,\ -0.6,\ 0,\ 0.3]$$

Every entry is a multiple of $s_t/N=0.3$, confirming the paper's claim (text right after Eq. 4) that scaler sharing bounds the averaged gradient to at most $2N+1 = 7$ levels: ${-0.9,-0.6,-0.3,0,0.3,0.6,0.9}$.

## Step 6 — Pull & update, Algorithm 1 (Worker steps 5–6)

Workers pull this quantized $g_t$ (not full-precision parameters — "parameter localization," Section 3.1) and update their local parameter copy:

$$w_{t+1} = w_t - \eta , g_t$$

With $w_t=[1.0,2.0,-1.0,0.5]$, $\eta=0.1$: $w_{t+1} = [0.94,\ 2.06,\ -1.0,\ 0.47]$.

## Communication savings (Section 3.1)

- Worker→server: $32/\log_2(3) \approx 20.18\times$ (stated right after Eq. 3), or $16\times$ with a practical 2-bit code.
- Server→worker: $32/\log_2(1+2N) \approx 32/\log_2(7) \approx 11.4\times$ for $N=3$ (stated right after Eq. 4).

You're right — I sprinkled unnecessary `$...$` math delimiters around plain numbers (like `$10\text{K}$`, `$N=64$`) that don't need LaTeX rendering at all and can break in Obsidian. Here's the same summary with all of that stripped to plain text/Markdown:

## Common Experimental Setup (Section 4, intro)

- **Framework**: TensorFlow
- **Parameter averaging**: exponential moving average with decay 0.9999; accuracy evaluated on the averaged parameters
- **Optimizers tested**: vanilla SGD, momentum SGD (momentum = 0.9), Adam
- **LR schedule**: polynomial decay, power = 0.5 (base LR → 0)
- **Gradient clipping constant**: c = 2.5, cross-validated once on CIFAR-10 and reused across _all_ datasets/models
- **Fairness control**: in each floating-vs-ternary pair, all other hyperparameters are held identical unless stated
- **Repetitions**: the paper does **not** report repeated runs, multiple seeds, or error bars — each configuration in every table/figure is a single training run to convergence

## Experiments and results
## Experiment 1 — LeNet on MNIST (Section 4.1, Figure 3)

|Setup detail|Value|
|---|---|
|Model / dataset|LeNet / MNIST (no data augmentation)|
|Optimizers|momentum SGD (base LR 0.01) and vanilla SGD (base LR 0.1)|
|Weight decay|0.0005|
|Total mini-batch size|64 (fixed)|
|Max iterations|10K|
|Workers tested|N = 2, 4, 8, 16, 32, 64|
|Reference figure|**Figure 3**|

![[Pasted image 20260908223452.png]]

**Result**: TernGrad tracks baseline accuracy closely at every worker count for both optimizers — max accuracy **gain** 0.15%, max **loss** 0.22%. No degradation even at N = 64 (one sample/worker/iteration), showing TernGrad scales to many workers without breaking convergence.

## Experiment 2 — CifarNet on CIFAR-10 (Section 4.1, Table 1)

|Setup detail|Value|
|---|---|
|Model / dataset|CifarNet (ConvNet) / CIFAR-10, with random 24x24 crop, mirroring, brightness/contrast jitter (center crop at test)|
|Optimizer|Adam, base LR 0.0002|
|Batch scheme|per-worker batch size fixed → total batch grows with N|

|Workers|Total batch|Iterations|Floating acc.|TernGrad acc.|Gap|
|---|---|---|---|---|---|
|2|128|300K|86.56%|85.64%|−0.92%|
|16|2048|18.75K|83.19%|82.80%|−0.39%|

**Result**: less than 1% accuracy degradation in both cases; the gap actually _shrinks_ at large batch size, which the authors attribute to TernGrad's inherent noise helping escape sharp minima.

## Experiment 3 — AlexNet on ImageNet (Section 4.2, Table 2, Figure 4)

|Setup detail|Value|
|---|---|
|Model / dataset|AlexNet / ImageNet|
|Per-worker mini-batch|128 (fixed)|
|Optimizer|momentum SGD, Batch Normalization on conv layers|
|Modifications for TernGrad|smaller dropout ratio, smaller weight decay, last (classification) layer kept in full precision|
|Reference figures/tables|**Table 2** (accuracy), **Figure 4** (accuracy/loss/sparsity curves)|

|Base LR|Total batch|Workers|Iterations|Gradients|Top-1|Top-5|
|---|---|---|---|---|---|---|
|0.01|256|2|370K|floating|57.33%|80.56%|
|0.01|256|2|370K|TernGrad|57.61%|80.47%|
|0.01|256|2|370K|TernGrad, no clip|54.63%|78.16%|
|0.02|512|4|185K|floating|57.32%|80.73%|
|0.02|512|4|185K|TernGrad|57.28%|80.23%|
|0.04|1024|8|92.5K|floating|56.62%|80.28%|
|0.04|1024|8|92.5K|TernGrad|57.54%|80.25%|

**Result**: TernGrad matches or _beats_ full-precision accuracy at every batch size — notably **+0.92% top-1 at batch 1024**, where floating-point actually drops. Removing gradient clipping costs **~3% top-1**, isolating its importance. Figure 4 (4 workers, batch 512) shows matching accuracy/loss curves and that **71.32%** of gradients in layer fc6 are ternarized to zero on average.

## Experiment 4 — GoogLeNet on ImageNet (Section 4.2, Table 3)

|Setup detail|Value|
|---|---|
|Model / dataset|GoogLeNet (no auxiliary classifiers) / ImageNet|
|Optimizer|polynomial LR decay, per [41]|
|Reference table|**Table 3**|

|Base LR|Total batch|Workers|Iterations|Gradients|Top-5|
|---|---|---|---|---|---|
|0.04|128|2|600K|floating|88.30%|
|0.04|128|2|600K|TernGrad|86.77%|
|0.08|256|4|300K|floating|87.82%|
|0.08|256|4|300K|TernGrad|85.96%|
|0.10|512|8|300K|floating|89.00%|
|0.10|512|8|300K|TernGrad|86.47%|

**Result**: average accuracy loss **under 2%**, using hyperparameters tuned for the floating-point baseline (not re-tuned for TernGrad) — suggesting further gains are possible with tuning.

## Experiment 5 — Performance/Throughput Model (Section 5, Figure 5)

Since the physical cluster was limited to 4 machines / 8 workers, the authors validated a performance model on that hardware, then used it to project throughput up to 512 GPUs.

|Setup detail|Value|
|---|---|
|Validation cluster|4 machines x 4 GTX 1080 GPUs, Mellanox InfiniBand|
|Projected clusters|(a) 128 nodes x 4 GTX 1080, 1Gbps Ethernet + PCI switch; (b) 128 nodes x 4 Tesla P100, 100Gbps InfiniBand + NVLink|
|Models tested|AlexNet, GoogLeNet, VggNet-A|
|Per-GPU batch size|AlexNet 128, GoogLeNet 64, VggNet-A 32|
|Reference figure|**Figure 5** (a & b)|

![[Pasted image 20260908223633.png]]

**Result**: TernGrad increases throughput for all three models; the speedup scales with the model's communication-to-computation ratio — AlexNet and VggNet-A (communication-heavy) benefit more than GoogLeNet. Standout numbers:

- **3.04x speedup** for AlexNet on 8 GPUs over low-bandwidth Ethernet/PCI
- Still **~2x speedup** for VggNet-A on 128 nodes even on the high-end InfiniBand+NVLink cluster

## Take-away pattern across all experiments

- **Small-scale (LeNet, CifarNet)**: near-zero accuracy cost, validates convergence theory broadly across optimizers and worker counts.
- **Large-scale (AlexNet, GoogLeNet)**: small accuracy trade-off (0 to ~2%), requiring the layer-wise ternarizing + clipping tricks to hold.
- **Throughput**: the real payoff — up to ~3x real speedup on bandwidth-constrained clusters, confirming the compression ratio math translates into wall-clock gains.