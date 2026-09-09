## Summary: EF21 — A New, Simpler, Theoretically Better, and Practically Faster Error Feedback

**The problem.** In distributed machine learning, many workers each hold local data and compute gradients that must be sent to a central server. Since full gradients are expensive to communicate, they're typically _compressed_ (e.g., keeping only the top-k largest coordinates) before sending. But naive compressed gradient descent can diverge, because biased compressors introduce systematic errors. **Error feedback (EF)**, first proposed in 2014, is the standard fix: each worker tracks the error it introduced by compressing (the difference between what it wanted to send and what it actually sent), and adds this error back into the next round's gradient before compressing again. Despite EF's practical popularity, its theoretical understanding remained weak — existing analyses either only worked for a single machine, needed unrealistic assumptions (like globally bounded gradients, which fails even for simple quadratics), or required adding extra unbiased compressors that increase communication cost.

**The proposed method and its theory.** The paper introduces **EF21**, a redesigned error-feedback mechanism that fixes these gaps. Instead of feeding back the compression error of the _raw gradient_, EF21 has each worker maintain a running local estimate of its own gradient and repeatedly compresses the _difference_ between the new true gradient and its previous estimate, then updates the estimate by adding that compressed difference back in. This subtle change of what gets compressed (a gradient-estimate correction rather than a stepsize-scaled gradient) makes the method's analysis far more tractable. The authors prove EF21 converges under only two standard, easily verifiable assumptions — smoothness of each worker's local function and a lower bound on the global objective — with no bounded-gradient or bounded-dissimilarity requirement, and it works in the realistic **heterogeneous** (non-i.i.d.) distributed data setting. Under these mild assumptions EF21 achieves the optimal $O(1/T)$ convergence rate for smooth nonconvex problems, improving on the previous best known $O(1/T^{2/3})$ rate (which itself needed the strong bounded-gradients assumption). Under the additional Polyak-Łojasiewicz condition, EF21 attains a fast **linear** convergence rate — the first such result for a biased-compressor error-feedback method that doesn't rely on extra unbiased compression. The authors also show a more aggressive variant, EF21+, and an extension to stochastic gradients, and prove that under restrictive conditions (deterministic, positively homogeneous, additive compressors) classical EF and EF21 actually coincide — explaining the name (an "error feedback mechanism from the year 2021").

**Results.** Beyond the theoretical gains, extensive experiments (logistic regression with a nonconvex regularizer on LibSVM datasets, plus least-squares and deep-learning benchmarks) show EF21 consistently and substantially outperforms classical EF in practice, largely because it tolerates much larger, more aggressive learning rates without diverging. Overall, the paper's contribution is a conceptually simple change to a widely used heuristic that simultaneously simplifies the theory, weakens the assumptions needed, strengthens the guaranteed convergence rate, and improves empirical performance — making EF21 a practical drop-in replacement for EF in communication-efficient distributed training.

Here's a structured breakdown of EF21's core math, algorithms, and a fully worked numerical example.

## 1. Problem Setup

The paper considers the standard **distributed finite-sum minimization** problem:

$$\min_{x\in\mathbb{R}^d} f(x), \quad \text{where } f(x) = \frac{1}{n}\sum_{i=1}^n f_i(x) \tag{1}$$

|Symbol|Meaning|
|---|---|
|$n$|number of worker nodes|
|$d$|dimension of the model $x$|
|$f_i$|local (private) loss function held by node $i$|
|$x^t$|shared model parameter at iteration $t$|
|$\mathcal{C}$|a compression operator applied to gradients before communication|

## 2. Compression Operators

Two classes of compressors are defined:

|Class|Condition|Notation|
|---|---|---|
|**Unbiased**|$\mathbb{E}[\mathcal{C}(x)]=x,\ \ \mathbb{E}\left[\|\mathcal{C}(x)-x\|^2\right]\le \omega\|x\|^2$|$\mathcal{C}\in \mathcal{U}(\omega)$|
|**Biased / contractive**|$\mathbb{E}\left[\|\mathcal{C}(x)-x\|^2\right]\le (1-\alpha)\|x\|^2,\quad 0<\alpha\le1$|$\mathcal{C}\in \mathcal{B}(\alpha)$|

EF21 is built for the **biased** class $\mathcal{B}(\alpha)$. Canonical example: **Top-$k$** (keep the $k$ largest-magnitude coordinates, zero the rest) satisfies $\mathcal{C}\in\mathcal{B}(\alpha)$ with $\alpha = k/d$.

**Why not just compress the gradient directly (DCGD)?** $$x^{t+1} = x^t - \frac{1}{n}\sum_{i=1}^n \mathcal{C}(\nabla f_i(x^t))$$ This is _distributed compressed gradient descent_, and it is known to **diverge** — biased compression introduces a persistent bias that plain averaging never corrects.

## 3. The Core Idea: the Markov Compressor

Instead of compressing the gradient itself, EF21 maintains a **running local gradient estimate** $g^t$ and compresses only the _change_ needed to correct it:

$$\mathcal{M}(v^0) := \mathcal{C}(v^0) \tag{9}$$ $$\mathcal{M}(v^{t+1}) := \mathcal{M}(v^t) + \mathcal{C}\left(v^{t+1}-\mathcal{M}(v^t)\right) \tag{10}$$

Writing $g^t := \mathcal{M}(\nabla f(x^t))$, this becomes self-referential: each new estimate is the old estimate plus a compressed correction. The key structural gain, formalized later, is that the **distortion** $|g^t-\nabla f(x^t)|^2$ obeys a _contraction recursion_ across iterations — unlike a raw compressor, whose distortion bound is static and doesn't shrink.

## 4. The Algorithms

**Algorithm 1 — EF21 (single node)**

```
Input: x⁰ ∈ ℝᵈ, learning rate γ > 0, g⁰ = C(∇f(x⁰))
for t = 0, 1, 2, ..., T-1:
    x^(t+1) = x^t - γ g^t
    g^(t+1) = g^t + C(∇f(x^(t+1)) - g^t)
```

**Algorithm 2 — EF21 (multiple nodes, the paper's main method)**

```
Input: x⁰; g_i⁰ = C(∇f_i(x⁰)) for i=1,...,n; γ>0; g⁰ = (1/n)Σ g_i⁰
for t = 0, 1, 2, ..., T-1:
    Master computes x^(t+1) = x^t - γ g^t, broadcasts x^(t+1)
    for each node i in parallel:
        c_i^t = C(∇f_i(x^(t+1)) - g_i^t)      # only this is COMMUNICATED
        g_i^(t+1) = g_i^t + c_i^t              # local state update
    Master computes g^(t+1) = g^t + (1/n)Σ c_i^t
```

_Only the compressed correction $c_i^t$ is sent — never the raw gradient._

**Algorithm 3 — EF21+ (hybrid, more aggressive)** Each node computes **both** a plain biased-compressed gradient $b_i^{t+1}=\mathcal{C}(\nabla f_i(x^{t+1}))$ and the Markov-compressed one $m_i^{t+1}=g_i^t+\mathcal{C}(\nabla f_i(x^{t+1})-g_i^t)$, measures which has smaller distortion from the true gradient, and keeps the better one as $g_i^{t+1}$.

## 5. Assumptions and Main Theorems

| #   | Assumption               | Statement                                                                                                          |
| --- | ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| 1   | Smoothness + lower bound | Each $f_i$ is $L_i$-smooth: $\|\nabla f_i(x)-\nabla f_i(y)\|\le L_i\|x-y\|$, and $f^{\inf}:=\inf_x f(x) > -\infty$ |
| 2   | Polyak–Łojasiewicz (PL)  | $f(x)-f(x^\star) \le \frac{1}{2\mu}\|\nabla f(x)\|^2$ for some $\mu>0$                                             |

Two derived constants govern everything, given $\mathcal{C}\in\mathcal{B}(\alpha)$: $$\theta = 1-\sqrt{1-\alpha}, \qquad \beta = \frac{1-\alpha}{1-\sqrt{1-\alpha}}$$ and the average distortion $G^t := \frac{1}{n}\sum_{i=1}^n |g_i^t-\nabla f_i(x^t)|^2$.

| | Theorem 1 (nonconvex) | Theorem 2 (PL, linear rate) |
|---|---|---|
| Extra assumption | none beyond Assumption 1 | Assumption 2 (PL) |
| Stepsize | $0<\gamma\le \left(L+\tilde{L}\sqrt{\beta/\theta}\right)^{-1}$ | $0<\gamma\le\min\left(\left(L+\tilde{L}\sqrt{2\beta/\theta}\right)^{-1},\ \theta/(2\mu)\right)$ |
| Guarantee | $\mathbb{E}\left[\lVert \nabla f(\hat{x}^T) \rVert^2\right] \le \dfrac{2(f(x^0)-f^{\inf})}{\gamma T} + \dfrac{\mathbb{E}[G^0]}{\theta T}$ | $\mathbb{E}[\Psi^T] \le (1-\gamma\mu)^T \mathbb{E}[\Psi^0]$, where $\Psi^t := f(x^t)-f(x^\star)+\dfrac{\gamma}{\theta}G^t$ |
| Rate | $O(1/T)$ | linear (geometric) |

Here $L \le \frac{1}{n}\sum_i L_i$ and $\tilde{L} := \left(\frac{1}{n}\sum_i L_i^2\right)^{1/2}$. Theorem 1 beats the prior $O(1/T^{2/3})$ EF rate while dropping the unrealistic bounded-gradient assumption; Theorem 2 is the first linear-rate result for biased-compressor EF without extra unbiased compression.

**For Top-$k$** ($\alpha=k/d$): $\sqrt{\beta/\theta} = \dfrac{1+\sqrt{1-k/d}}{k/d}-1$ — this shrinks as $k$ grows, so less aggressive compression permits a larger stepsize, exactly as intuition suggests.

**Bonus result (Theorem 3):** if $\mathcal{C}$ is deterministic, positively homogeneous, and additive, classical EF and EF21 produce _identical_ iterate sequences — explaining the name ("error feedback, 2021 edition").

## 6. Worked Numerical Example

**Setup** — $n=2$ nodes, $d=2$, quadratics with a shared minimizer: $$f_1(x)=\tfrac12|x-a_1|^2,\ a_1=(4,0), \qquad f_2(x)=\tfrac12|x-a_2|^2,\ a_2=(0,3)$$ $$\nabla f_i(x)=x-a_i,\quad L_1=L_2=1 \Rightarrow L=\tilde L=1,\qquad f(x)=x-(2,1.5)\text{-minimizer: } x^\star=(2,1.5)$$

Compressor: **Top-1** on $\mathbb{R}^2$ ($k=1,d=2\Rightarrow\alpha=0.5$). Then $\theta\approx0.293$, $\beta\approx1.707$, $\sqrt{\beta/\theta}\approx2.414$.

- Theorem 1 bound: $\gamma \le 1/(1+2.414)\approx 0.293$
- Theorem 2 bound (also PL, $\mu=1$): $\gamma \le \min(0.227,,0.146) = 0.146$

We pick $\gamma = 0.1$ (safely satisfies both), start at $x^0=(0,0)$.

**Iteration table** (Algorithm 2 in action):

|$t$|$x^t$|$\nabla f_1(x^t)$|$\nabla f_2(x^t)$|$g_1^t$|$g_2^t$|$g^t$|
|---|---|---|---|---|---|---|
|0|$(0, 0)$|$(-4, 0)$|$(0, -3)$|$(-4, 0)$|$(0, -3)$|$(-2, -1.5)$|
|1|$(0.20, 0.15)$|$(-3.8, 0.15)$|$(0.2, -2.85)$|$(-3.8, 0)$|$(0.2, -3)$|$(-1.8, -1.5)$|
|2|$(0.38, 0.30)$|$(-3.62, 0.30)$|$(0.38, -2.70)$|$(-3.8, 0.30)$|$(0.2, -2.70)$|$(-1.8, -1.2)$|
|3|$(0.56, 0.42)$|...|...|...|...|...|

**Step-by-step for $t=0\to1$:**

- $g_1^0=\text{Top-1}(-4,0)=(-4,0)$ (larger-magnitude coordinate kept), $g_2^0=\text{Top-1}(0,-3)=(0,-3)$
- Master step: $x^1 = (0,0) - 0.1\cdot(-2,-1.5) = (0.2, 0.15)$
- Correction node 1: $\nabla f_1(x^1)-g_1^0=(0.2, 0.15)$; Top-1 keeps first coord: $c_1^0=(0.2,0)$, so $g_1^1=(-3.8,0)$
- Correction node 2: $\nabla f_2(x^1)-g_2^0=(0.2, 0.15)$; Top-1 keeps first coord: $c_2^0=(0.2,0)$, so $g_2^1=(0.2,-3)$
- **Only 2 scalars (+indices) per node were ever sent**, not the full 2D gradient.

**Why this beats naive compression:** watch how the _distortion_ $|g_i^t-\nabla f_i(x^t)|$ behaves instead of staying stuck:

|$t$|$\|g_1^t - \nabla f_1(x^t)\|$|$\|g_2^t - \nabla f_2(x^t)\|$|
|---|---|---|
|0|$0$|$0$|
|1|$0.15$|$0.15$|
|2|$0.18$|$0.18$|

The coordinate that Top-1 "missed" at one step gets picked up at the next (it alternates which coordinate is transmitted as the gradient shifts), so the estimate keeps self-correcting — this is exactly the contraction behavior of Lemma 2, $\mathbb{E}[G^{t+1}]\le(1-\theta)G^t+\beta|\nabla f(x^{t+1})-\nabla f(x^t)|^2$, rather than a fixed, un-shrinking compression bias. Meanwhile $x^t$ steadily marches toward $x^\star=(2,1.5)$: $(0,0)\to(0.2,0.15)\to(0.38,0.30)\to(0.56,0.42)\to\dots$

## Experiments in the EF21 Paper

### Overall Goal

The experiments test two things: (1) how much **larger a stepsize** EF21/EF21+ can tolerate compared to classical EF before diverging/stalling, and (2) how much **less communication** (bits sent) EF21 needs to reach a target accuracy, across convex, nonconvex, and deep-learning settings.

### Setup, Hardware, and Datasets

|Experiment family|Model / objective|Datasets|Hardware|Code|
|---|---|---|---|---|
|Main paper (§5) & Appendix A.1|Nonconvex logistic regression: $f(x)=\frac{1}{n}\sum_i \log(1+e^{-y_i a_i^\top x}) + \lambda\sum_j \frac{x_j^2}{1+x_j^2}$, $\lambda=0.1$|LibSVM: **phishing, mushrooms, a9a, w8a**|3 CPU node types (AMD EPYC 7702 64-Core; Intel Xeon Gold 6148; Intel Xeon Gold 6248)|Python 3.8|
|Appendix A.2|Least squares $f(x)=\frac{1}{n}\sum_i(a_i^\top x-b_i)^2$ (not strongly convex, but satisfies PL)|Same 4 LibSVM datasets|Same CPU nodes|Python 3.8|
|Appendix A.3|Deep learning: **ResNet18** and **VGG11** image classifiers|**CIFAR-10** (50,000 train / 10,000 test)|3 GPU types (NVIDIA GTX 1080 Ti, RTX 2080 Ti, Tesla V100)|PyTorch|

**Distributed split:** $n=20$ clients for the convex/nonconvex experiments (data split into equal parts, heterogeneous regime), $n=5$ workers for the deep-learning experiments (batches of size $\tau\in{128,1024}$).

|Dataset|$n$|$N$ (total points)|$d$ (features)|$N_i$ per client|
|---|---|---|---|---|
|phishing|20|11,055|68|552|
|mushrooms|20|8,120|112|406|
|a9a|20|32,560|123|1,628|
|w8a|20|49,749|300|2,487|

**Repetitions:** the paper does not report averaging over multiple random seeds/trials — each configuration (dataset × $k$ × stepsize multiplier) is a single run, which is reasonable since Top-$k$ is deterministic.

### Key Experiments and Results

**1. Stepsize tolerance (Figure 1, main text; extended in Figures 3–6, Appendix A.1.1)**

- Setup: Top-1 compressor, stepsize swept as multiples (1×, 8×, 16×...128×) of the theoretical bound from Theorem 1, on dataset a9a (and phishing/mushrooms/w8a in the appendix).
- Result: classical EF plateaus at a fixed error level for _every_ stepsize tested (it never reaches low $|\nabla f(x^t)|^2$). EF21 and EF21+ keep improving as the stepsize grows and tolerate far larger multiples (EF21+ remained stable even at 2048× the baseline) before eventually diverging.

**2. Fine-tuned communication efficiency (Figure 2, main text; extended Figure 7, Appendix A.1.2)**

- Setup: for each method, both $k$ and $\gamma$ are individually fine-tuned; plain distributed gradient descent (GD, i.e., $k=d$, no compression) is added as a baseline.
- Result: on all four datasets, EF21 and EF21+ reach the target gradient-norm accuracy using far fewer bits/client than EF (which stalls), and GD is the least communication-efficient of all. The best $k$ found across methods was small (1, 2, or 4).

**3. Least-squares / PL-condition experiments (Appendix A.2, Figure 8)**

- Same qualitative pattern as the logistic regression case: EF stalls under large stepsizes while EF21/EF21+ continue converging — empirically confirming the paper's Theorem 2 linear-rate guarantee on a PL (but not strongly convex) objective.

**4. Deep learning: ResNet18 & VGG11 on CIFAR-10 (Appendix A.3, Figures 13–15)**

- Setup: full gradients replaced by minibatch stochastic gradients; $k\approx 0.05D$ (D = number of model parameters: 11.5M for ResNet18, 132.9M for VGG11); compared against EF, EF21, EF21+, and vanilla SGD, all with individually tuned stepsizes.
- Result (Fig. 13, ResNet18 / Fig. 14, VGG11): EF21 and EF21+ track EF closely during training but reach **better test accuracy** for both architectures at comparable communication budgets. Figure 15 additionally shows that shrinking $k$ (0.01D vs 0.05D vs 0.25D) makes EF21 more communication-efficient and reach higher test accuracy sooner.

### Conclusion

Across every regime tested — nonconvex logistic regression, least-squares (PL) problems, and deep neural network training — the experiments consistently support the paper's two central theoretical claims: EF21 tolerates substantially larger learning rates than classical EF without getting stuck at a residual error floor, and it reaches a target accuracy while transmitting far fewer bits per client. The hybrid EF21+ variant pushes this further, tolerating even more aggressive stepsizes by adaptively falling back to plain compression when it beats the Markov-compressor estimate. Because these gains hold on both convex-type benchmarks and real deep-learning workloads (ResNet18/VGG11 on CIFAR-10), the experiments support the paper's claim that EF21 is not just a theoretical refinement but a practically superior drop-in replacement for classical error feedback in communication-constrained distributed training.