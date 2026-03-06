## Year 2024

## DESIGN OF FAULT-TOLERANT DL CACHE

### Addressing Failure via IO Redirection to PFS

To ensure fault tolerance in distributed caching systems for deep learning, failed node requests can be redirected to the Parallel File System (PFS), which stores the original training data. The approach involves two stages: detecting node failures (via timeout-based monitoring) and redirecting I/O operations to the PFS. Each client independently detects failures and reroutes requests, maintaining data availability without extra inter-node communication.

While effective for small datasets or late-epoch failures (where PFS accesses are infrequent), this method becomes problematic with large datasets or early-epoch failures. Frequent PFS access increases latency, causing:

1. **Straggler problems** – slower nodes delay synchronized batch processing, harming parallelism and scalability.
    
2. **Job time limit violations** – longer runtimes can exceed HPC job allocations, leading to premature termination and wasted resources.
    

Thus, PFS redirection offers simplicity and resilience but significantly impacts performance and runtime predictability in large-scale or early-stage training scenarios.

![[Pasted image 20251026225533.png]]

### Elastic Recaching with Hash Ring

In the original HVAC caching system, static hash partitioning evenly distributes data across nodes using a hash of the data path. However, when a node fails, recalculating hashes for N−1N-1N−1 nodes causes massive data redistribution, moving even well-cached data and reducing efficiency. Alternatives like multiple hash functions or range partitioning still suffer from scalability or load-balancing issues.

To address this, the authors implemented **consistent hashing (hash ring)** for fault tolerance. Both data and nodes are mapped to positions on a circular hash ring, and each data item is stored on the next clockwise node. When a node fails, only its data is reassigned to the next node, minimizing data movement.

To avoid load imbalance (when one node gets too much data), **virtual nodes** are introduced—each physical node appears multiple times on the ring, leading to more even data distribution. However, increasing virtual nodes raises resource costs, requiring balance between fairness and efficiency.

The resulting **elastic recaching approach** rebuilds the hash ring dynamically after a failure, redirecting requests to new nodes. Lost data is fetched once from the PFS, served to the client, and cached locally. This mechanism minimizes PFS accesses, limits data movement to only lost items, and adapts quickly using efficient map-based data structures with logarithmic time complexity—providing fault tolerance with minimal performance impact.

![[Pasted image 20251026225706.png]]

## Evaluation

### Experimental Setup

The proposed fault-tolerant HVAC system was evaluated on the **Frontier supercomputer**, where each node includes **two 1.9 TB Samsung PM9A3 NVMe SSDs** (aggregated into a **3.5 TB RAID0 XFS volume**) with up to **8 GB/s read** and **4 GB/s write** bandwidth. Frontier also uses the **Orion Lustre parallel file system** for shared storage.

**Implementation:**  
The team extended the original **C++ HVAC system**, built on the **Mercury RPC library**, to include fault tolerance through a **consistent hashing (hash ring)** mechanism implemented using C++’s `std::map`. Around **1,000 lines of code** were added or modified.

**Application & Dataset:**  
Experiments used **CosmoFlow**, a 3D CNN from **MLPerf HPC v0.5**, trained on **1.3 TB of cosmology simulation data** (524K training and 65K validation samples). CosmoFlow runs on **Horovod** with **elastic training**, allowing it to continue training after node failures by restarting the failed epoch.

**Failure Injection:**  
Random node failures were simulated after the **first epoch** (once data was cached) using **SLURM commands** to drain nodes dynamically during runtime. This mimicked real HPC failures while avoiding bias by randomizing the failure timing and nodes.

**Compared Systems:**

1. **NoFT:** Original HVAC without fault tolerance.
    
2. **FT w/ PFS:** Redirects missing I/O requests to the parallel file system (PFS).
    
3. **FT w/ NVMe:** The proposed **hash ring–based elastic recaching** system, with **100 virtual nodes per physical node**, which re-caches lost files from NVMe storage after failure.

### Evaluation Results

1)- **Overall Performance** Figure 5(a) shows that under **non-failure conditions**, all configurations (NoFT, FT w/ PFS, and FT w/ NVMe) scale efficiently as the number of nodes increases. The **NoFT version** performs best due to the absence of overhead from fault-tolerance mechanisms such as conditional checks, timeout monitoring, and mutex locks, which slightly slow the fault-tolerant versions.

When **node failures** are introduced (Figure 5b), single-node failures occur randomly after the first epoch to simulate cache loss.

- The **NoFT baseline** crashes immediately upon failure.
    
- **FT w/ PFS** suffers the **highest slowdown** — up to **68.7% longer runtime at 1024 nodes** — because it continuously accesses the **Parallel File System (PFS)** for lost data.
    
- **FT w/ NVMe**, which re-caches lost files only once, performs significantly better, with **12.5–26.7% overhead**, outperforming FT w/ PFS by up to **25%**.
    

As the system scales, relative overhead grows because the **fixed rollback time** of Horovod’s elastic training becomes more prominent when per-epoch training time decreases. However, the gap between FT w/ NVMe and FT w/ PFS remains large since **PFS access creates straggler nodes** that slow down overall progress—highlighting the critical need to minimize PFS usage in large-scale, fault-tolerant DL systems.

Figure 6(a) further confirms that **epochs with NVMe recaching** recover much faster than those using **PFS redirection**, achieving near **no-failure performance** as node count increases—demonstrating **better scalability and resilience** of the NVMe-based fault-tolerant design.

2)- **Load Distribution Analysis**: 

When a node fails in a distributed system, its data must be redistributed to other nodes. The system can use _virtual nodes (vNodes)_ — logical subdivisions of physical nodes — to help balance this redistribution. The experiment simulates 1024 physical nodes, each configured with varying numbers of virtual nodes, repeated 500 times to compute averages.
### 🔸 **Key Observations:**

1. **Increasing virtual nodes improves load distribution:**
    
    - As the number of virtual nodes per physical node grows, more nodes participate in receiving redistributed data after a failure.
        
    - **Example:**
    
        - With **10 vNodes per node**, only **~3 nodes** receive the redistributed data.
            
        - With **1000 vNodes per node**, about **300 nodes** share the load.
    
    - This means the load becomes **more evenly spread** across the cluster.
    
2. **Files per node decrease with more vNodes:**
    
    - When more nodes receive data, each one receives fewer files.
        
    - This leads to **better balance** and **less overload** per node.
        
    - The decreasing standard deviation of this metric confirms that the file distribution becomes more uniform.
    
3. **Diminishing returns beyond ~500 vNodes:**
    
    - The improvement in the number of receiver nodes slows after around **500 vNodes per physical node**.
        
    - Beyond this point, the number of receiver nodes stabilizes at **≈ 350**, with little further gain.
        
    - Hence, the relationship between vNodes and receiver nodes is **non-linear** — adding more vNodes doesn’t proportionally increase load balance.
    
4. **Trade-offs and optimal configuration:**
    
    - Increasing vNodes enlarges the **hash table**, which increases **memory usage** and **computation time**.
        
    - Therefore, more vNodes = better balancing but **higher overhead**.
        
    - The authors identify an **optimal value** of around **100 virtual nodes per physical node** for their system, balancing performance and resource efficiency.


