## Title: Scalable Distributed DNN Training Using Commodity GPU Cloud Computing

## Year: 2015

### Method

The approach used in the paper is data parallelism, where each worker has a full replica of the model and generates gradient for its own minibatch

Authors mentioned 2 observations:

1. **Many techniques for speeding up SGD can be formulated as variants of delaying weight updates**:  such as double buffering, async sgd, waiting for a full mini batch calculation to calculate gradients instead of per sample calculations.

2. **Sub gradients are very sparse(which means most of gradients are near or equal to 0)**: most weights in a fully connected DNN have near zero values, which means that only a small fraction of gradients need to be communicated instead of all of the gradients, since most of them are almost null, send only the most significant ones, other values can be delayed.
### Compaction and dead reckoning

the paper considers communicating only sub-gradients which absolute value exceed a threshold $\tau$ , the resulting sparse gradient is a key-value map, where key -> indices of gradients, and values -> gradient value, these maps are the messages communicated in SGD. the calculation of this map is done in the GPU for better efficiency, and is under the umbrella of **string compaction** problem class.

There is no synchronization needed, the paper uses **dead reckoning** in which they send the deltas of the weights instead of gradients, these deltas need to be the same in all nodes, and can be received in different orders since the update is commutative.

### Gradient residual

We cannot simply ignore the small gradient updates, since different gradient elements have different  ranges, so in each node, after each mini-batch, gradients are aggregated as **gradient residuals**, and now residual values that exceed $\tau$ are communicated, and then substracted from the residuals.

### Quantization and compression

- Naive implementation -> (integer for the index, float for gradient value)
- Can be improved by combining the quantized gradient and the index in a single 32-bit integer
- Quantization error is added to gradient residuals.
- 1 bit is found sufficient to represent quantized value($+\tau$ and $-\tau$ ), and 31 bits are more than enough to represent index.
- Messages can be further compressed using entropy encoding, which can be used on the GAPS between indices instead of absolute values, Golomb Rice encoding reduces the size from 32 -> 10 or 11 bits => which is **3 times better**. but was not used in this paper as it caused some transmission delay because of extra computation.

### Pseudo Code

1. Receive and uncompress any weight update messages from other compute nodes and apply them to the local replica of the DNN

2. Load feature vectors and supervision targets for a mini-batch

3. Compute a sub-gradient $\mathbf{G}^{(s)}$ by Back-Propagation

4. Aggregate the sub-gradient in the gradient residual

   $$
   \mathbf{G}^{(r)} = \mathbf{G}^{(r)} + \mathbf{G}^{(s)}
   $$

5. Reset the message map $\mathbf{M}$

6. For each element $g_i^{(r)}$ of $\mathbf{G}^{(r)}$:

7. If $g_i^{(r)} > \tau$ then

   - Push the pair $\{i,+\tau\}$ to the message $\mathbf{M}$
   - Subtract $\tau$ from residual:

     $$
     g_i^{(r)} = g_i^{(r)} - \tau
     $$

   Else if $g_i^{(r)} < -\tau$ then

   - Push the pair $\{i,-\tau\}$ to the message $\mathbf{M}$
   - Add $\tau$ to the residual:

     $$
     g_i^{(r)} = g_i^{(r)} + \tau
     $$

8. Compress $\mathbf{M}$ and send to all other compute nodes

9. Apply $\mathbf{M}$ to the local replica of the DNN

> [!NOTE]
> - Asynchronous communication (as in “Hogwild”) is straightforward to implement with this method.  
> - Empirically, async operation is **significantly faster** when scaling beyond **10 compute nodes**.  
> - For tasks with **variable batch size** (e.g., sequence‑based BMMI training), synchronous training is highly inefficient.  
> - Therefore, the paper **focuses on asynchronous operation** for distributed training.

##  Experiments

### Hardware and infrastructure
- **Cloud platform**: Commodity AWS infrastructure
- **Compute nodes**: EC2 G2 instances with NVIDIA GRID K520 GPU (1,536 CUDA cores, 4 GB device memory)
- **Networking**: Standard AWS networking (no special high‑speed interconnects)
- **Data storage**: AWS S3; feature vectors and supervision targets fetched on‑demand

### Task and data
- **Task**: Acoustic modelling for automatic speech recognition (ASR)
- **ASR system**: Hybrid DNN‑HMM (DNN outputs posterior probabilities of phonetic states)
- **Evaluation metric**: Word Error Rate (WER) – absolute WER not important, only relative degradation vs. single‑node baseline
- **Training data**: 1,000‑hour subset of transcribed in‑house Amazon speech data
- **Features**: 20 log Mel filter bank coefficients, force‑aligned with existing ASR system
- **Frame segmentation**: 25 ms analysis window, 10 ms step size → **368 million** training feature vectors
- **Test set**: Separate set with 185,000 words

### DNN architecture and training recipe
- **Initialisation**: Supervised layer‑by‑layer pre‑training (single‑threaded SGD, small subset of data)
- **Final DNN structure**:
  - Fully connected layers
  - Input: window of centre frame ±8 frames (i.e., 17 frames)
  - Hidden layers: 5 layers, each affine transform + sigmoid non‑linearity
  - Output layer: affine + softmax over phonetic classes
  - **Total trainable parameters**: 14.6 million
- **Distributed training**: 10 epochs of distributed SGD (cross‑entropy; method also works for BMMI)
- **Learning rate schedule**: Halved after each epoch
- **Early‑training stabilisation** (to avoid divergence):
  - First 1/6 of epoch 1: smaller mini‑batch size and fewer compute nodes
  - Afterwards: normal batch size and full parallelism (per Table 1)

![[Pasted image 20260601015309.png]]

### Results

- **Baseline**:
  - Single‑threaded SGD on same hardware/infrastructure with identical dimensions and hyper‑parameters
  - Used to measure relative WER reduction for all other results
  - Total elapsed time for epochs 1–10: 102 hours (~10,000 frames/second)
  - Subsequent results only report on epochs 2–10

- **Scaling the number of compute nodes**:
  - Number of nodes varied from 5 to 80 during epochs 2–10 (other hyper‑parameters constant)
  - First epoch used a fixed 10 nodes (elapsed time: 84 minutes)
  - Method scales well up to at least 80 GPU nodes
  - Distributed training showed slightly faster convergence per epoch and marginally better asymptotic accuracy (no definitive explanation, but no evidence of degradation in convergence speed or accuracy)

![[Pasted image 20260601015432.png]]

![[Pasted image 20260601015458.png]]

![[Pasted image 20260601015530.png]]

**Effect of varying the threshold τ**

- Baseline results in Table 2 used τ = 4
- To study effect of τ, reran epochs 2–10 with 40 compute nodes across different τ values
- Too high τ → convergence affected, WER degraded
- Too low τ → weight update messages too large, communication bottleneck slows training
- However, a wide interval of τ exists that yields both fast convergence and small enough update messages for optimal training speed

![[Pasted image 20260601015624.png]]

**Scaling the model size**

- Larger models → more computation and larger weight updates, but also sparser update distributions → lower communication‑to‑computation ratio
- Tested DNNs with double and triple the hidden layer size, plus an 8‑hidden‑layer DNN (see Table 4)
- WER significantly better than baseline
- Crucially, weight update size does **not** grow with hidden layer size, and grows slower than number of hidden layers
- Therefore, the method becomes **more efficient** as model size increases

**Comparison with previous results**

- No prior results for fully‑connected DNN architectures competitive with the reported scaling
- Early CPU‑based example: 42M weight ASR DNN, only 2.2× speedup, plateaued at 8 nodes (20 CPU cores each)
- More recent work: scaled to at least 4 GPU nodes, but scaling beyond 16 nodes not demonstrated
- Several results limited to GPUs in a single server — low single‑digit speedups
- Closest comparable results use InfiniBand clusters:
  - One study: impressive speedups with model‑parallel CNN training up to 64 GPU nodes
  - Another study: for fully‑connected DNNs, 10× speedup using model‑ + data‑parallelism across 20 GPU nodes

**Discussion**

- The method practically solves distributed DNN training for current model sizes and datasets: very large models train in days/hours instead of months
- The method becomes *more* efficient as model size increases, enabling future scaling to even larger models
- However, scaling the number of compute nodes is not limitless

**First limitation: effective batch size**
- More nodes → larger effective batch size (samples aggregated per weight update)
- Early in training, longer batch sizes degrade convergence rate
- Mitigations: tune hyper‑parameters or use automatic mini‑batch size selection
- Ultimately, this is a fundamental trade‑off of SGD

**Second limitation: peer‑to‑peer gradient communication**
- Beyond a task‑ and system‑dependent node count, training becomes communication‑bound again
- Potential mitigation: hierarchical aggregation of messages
- However, because messages are sparse and quantized, aggregation may not substantially reduce total size

- The experiments used the most challenging configuration with respect to both limitations, with moderately sized DNNs
- The method scales better for larger models
- Therefore, despite theoretical limits, in practice the method may scale to considerably more nodes than shown in this report