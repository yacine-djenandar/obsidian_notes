
### Title: Stochastic Sign Descent Methods: New Algorithms and Better Theory

The paper introduces new sign based gradient quantization techniques.

### Single Node contributions

The paper uses the traditional method of weight update: 

$$\Large x_{k+1} = x_k - \gamma_k \operatorname{sign} \hat g(x_k)$$

This is Option 1 of signSGD — the plain deterministic update (no function-value comparison, unlike the arg min version from before).

The paper proposes a new method for updating weights in gradients, which is 


$$\Large x_{k+1} = \arg\min\{f(x_k),\, f(x_k - \gamma_k \operatorname{sign} \hat g(x_k))\}.$$

where:

| Symbol                               | Meaning                                                         |
| ------------------------------------ | --------------------------------------------------------------- |
| $\large{x_k}$                        | Model parameters at iteration $k$                               |
| $\large{x_{k+1}}$                    | Model parameters at the next iteration                          |
| $\large{f(\cdot)}$                   | The objective function                                          |
| $\large{\gamma_k}$                   | Step size at iteration $k$                                      |
| $\large{\hat g(x_k)}$                | Stochastic gradient at $x_k$, and unbiased estimation for $g_k$ |
| $\large{\operatorname{sign}(\cdot)}$ | Component-wise sign function                                    |
| $\large{\arg\min\{\cdot,\cdot\}}$    | Picks whichever of the two candidate points has lower $f$-value |
| $\large g_i(x_k)$                    | The actual gradient coordiante at $x_k$                         |
For both methods to converge, the paper proposes a condition called success probability bound(SPB) which is: 
$$\Large \rho_i(x):=\text{Prob}(\operatorname{sign} \hat g_i(x_k) = \operatorname{sign} g_i(x_k)) > \frac{1}{2}$$
### Distributed Setup

The paper proposes a new sign-based method, called **Stochastic Sign Descent with Momentum (SSDM)**, that uses a new sign stochastic sign operator defined as:


$$
\Large \left(\widetilde{\operatorname{sign}}\, g\right)_i =
\begin{cases}
+1, & \text{with probability } \dfrac{1}{2} + \dfrac{1}{2}\dfrac{g_i}{\|g\|} \\[6pt]
-1, & \text{with probability } \dfrac{1}{2} - \dfrac{1}{2}\dfrac{g_i}{\|g\|}
\end{cases}
$$


and  $\large \widetilde{\operatorname{sign}}(0) = 0$ with probability = 1

The two main algorithms in this paper are

#### For single nodes:

![[Pasted image 20260917012029.png]]

#### For Multiple nodes, the SSDM version:

![[Pasted image 20260917012232.png|700]]


One of the characteristics of SSDM is that it is all reduce compatible, no need for the decompress, add, recompress flow in this case

> [!info] Important Note 
> The paper does not provide any experimentation on SSDM, only experiments on the original signSGD and signSGD with majority vote under

