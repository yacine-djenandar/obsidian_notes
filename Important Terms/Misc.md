#iteration

Training Iteration if a full feed-forward + back propagation step in model training.



#stragglers

**Stragglers** are slow-working nodes in a distributed system that delay the entire training process. In distributed training with a Parameter Server (PS), stragglers are workers that take significantly longer than others to complete their computation or communication tasks.
### The Core Problem: Synchronization Barrier

In **synchronous training** (most common with PS):

> **The PS must wait for ALL workers to finish before proceeding to the next iteration**

This means:

- Fast workers finish → sit idle waiting
    
- One slow worker → **entire system slows down** to its pace
    
- Training time = time of the **slowest worker** × number of iterations