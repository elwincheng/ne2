# 7.3.x Scalability & Performance Module (Author: Elwin Cheng)

## 7.3.x.1 Module Overview

The Scalability & Performance module addresses the computational limitations of the baseline resilient Nash equilibrium (NE) seeking algorithm. The existing Python implementation, while functionally correct, exhibited two critical scalability bottlenecks: (1) excessive memory consumption that limited the simulation to approximately 150 agents before exceeding a 32 GB memory constraint, and (2) a runtime-dominant filtering subroutine that consumed over 77% of total execution time. This module documents the final design of the optimizations that resolved both issues, enabling the algorithm to scale beyond 1,000 agents with a 4x overall runtime speedup and a 94x speedup in the computational bottleneck.

The design was guided by three principles: minimize memory footprint growth as the number of agents $N$ increases, exploit the inherent parallelism in the per-agent filtering computation, and preserve the algorithmic correctness of the original implementation. The resulting system integrates three layers of optimization: a memory-efficient gradient computation that reduces storage complexity from $O(N^4)$ to $O(N^3)$, a C++ native extension via pybind11 that eliminates Python interpreter overhead in the inner loop, and OpenMP thread-level parallelism that distributes work across CPU cores.

In addition to the runtime and memory optimizations, this module introduces an adaptive step size framework and Nesterov-accelerated gradient method (Accelerated GRANE) that improve the convergence rate of the algorithm, reducing the number of iterations required to reach a given error tolerance.

## 7.3.x.2 System Architecture

Figure 1 shows the high-level architecture of the scalability module and how it integrates with the baseline algorithm.

```
+----------------------------------------------------------------------+
|                     Resilient NE Simulation                          |
|                        (resilient.py)                                |
|                                                                      |
|  +------------------+     +-------------------+    +---------------+ |
|  | Adversarial      |     | Message Filtering |    | Gradient      | |
|  | Communication    |---->| (C++ / OpenMP)    |--->| Update        | |
|  | (Python/NumPy)   |     | remove_d.cpp      |    | (Python/NumPy)| |
|  +------------------+     +-------------------+    +-------+-------+ |
|                                                            |         |
|                                                    +-------v-------+ |
|                                                    | Step Size     | |
|                                                    | Controller    | |
|                                                    | adaptive_     | |
|                                                    | step.py       | |
|                                                    +---------------+ |
+----------------------------------------------------------------------+
         |                        |                        |
    Python layer            C++ native ext.          Python layer
    (unchanged)          (new: pybind11+OpenMP)   (new: adaptive LR)
```

**Figure 1.** System architecture showing the three-layer optimization. The message filtering hotspot was ported to C++ with OpenMP parallelism. The gradient update was refactored for memory efficiency, and a new adaptive step size controller module was added.

The algorithm executes in three phases per iteration:

1. **Adversarial Communication**: Each agent broadcasts its state estimate to neighbors. Adversarial agents inject noise or send constant values. This phase is matrix-based (NumPy) and was not a bottleneck.

2. **Message Filtering**: Each agent filters received messages by removing the $D$ most extreme values above and below its own estimate, then averages the remaining values. This is the computational bottleneck ($O(N^2)$ per-agent, $O(N^3)$ total), ported to C++.

3. **Gradient Update**: The filtered state is used to compute a gradient step. The step size is controlled either by a fixed constant or by an adaptive method (AdaGrad, RMSProp, Adam, AMSGrad, NAdam), or by Nesterov-accelerated extrapolation.

## 7.3.x.3 Memory Optimization: Gradient Computation Refactoring

### Problem

The baseline implementation precomputed a combined matrix `RF = R^T * F` of dimensions $(N \cdot d) \times (N^2 \cdot d)$, where $d = 2$ is the per-agent action dimension. For $N$ agents, this matrix requires $O(N^4)$ memory. At $N = 150$ agents, `RF` alone occupies approximately 30 GB, approaching the 32 GB system limit.

The gradient update in the baseline was computed as:

$$x_{k+1} = v_k - \alpha \cdot (\texttt{RF} \cdot v_k + R^T b)$$

where $\texttt{RF} = R^T F$ is the precomputed $(Nd) \times (N^2 d)$ matrix.

### Solution

The key observation is that the matrix-vector product $R^T F v$ can be decomposed into two sequential multiplications:

$$R^T F v = R^T \cdot (F \cdot v)$$

The intermediate result $F \cdot v$ has dimension $(Nd) \times 1$, and $F$ itself is block-diagonal with dimensions $(Nd) \times (N^2 d)$. While $F$ is the same size as `RF`, it is constructed from `block_diag` (a sparse structure) that SciPy handles efficiently. The critical difference is that we no longer store the dense product $R^T F$, which has $O(N^4)$ entries. Instead, the two separate multiplications each involve matrices of $O(N^3)$ or fewer non-zero entries.

```python
# Baseline (O(N^4) memory):
self.RF = self.R.transpose().dot(self.F)      # Precompute once
state_x = state_v - step * (self.RF.dot(state_v) + self.RB)

# Optimized (O(N^3) memory):
temp1 = self.F.dot(state_v)                   # (Nd x 1), using block-diagonal F
temp2 = self.R.transpose().dot(temp1)          # (N^2*d x 1) -> (Nd x 1)
state_x = state_v - step * (temp2 + self.RB)
```

This refactoring reduces memory from $O(N^4)$ to $O(N^3)$, enabling the simulation to scale from approximately 150 agents to over 1,000 agents within the same 32 GB memory budget.

## 7.3.x.4 Runtime Optimization: C++ Native Extension with OpenMP

### Profiling and Bottleneck Identification

Using Python's `cProfile` module on a 15x15 grid (221 agents, 1,000 iterations), the profiling results revealed that `filter_communicated_message` consumed 77% of total runtime (approximately 146 out of 333 seconds for the unoptimized version). Within this function, the inner call to `remove_extreme_D_average` accounted for 44% of total runtime.

The filtering function has a triply-nested loop structure (over agents, states, and action components) that cannot be vectorized as a matrix operation because each agent performs a sort-and-trim operation on a different set of neighbor messages. The Python implementation incurs significant interpreter overhead on every iteration of these inner loops.

### C++ Port via pybind11

The entire `filter_communicated_message` function, including the inner `remove_extreme_D_average` call, was ported to C++ and exposed to Python via pybind11. The C++ implementation preserves the exact same algorithmic logic while eliminating Python interpreter overhead.

Key design decisions in the C++ implementation:

**Compressed Sparse Row (CSR) format for the adjacency list.** The Python implementation stores the adjacency list as a list of sets. To pass this to C++, we convert it to CSR format (an `offsets` array of length $N+1$ and a flat `neighbors` array of length $E$, where $E$ is the total number of edges). This is both memory-efficient and cache-friendly for sequential neighbor access.

```python
def adjlist_to_csr(adj_list_gc, N):
    offsets = np.zeros(N + 1, dtype=np.int64)
    neighbors = []
    for i in range(N):
        offsets[i+1] = offsets[i] + len(adj_list_gc[i])
        neighbors.extend(adj_list_gc[i])
    return offsets, np.asarray(neighbors, dtype=np.int64)
```

**Direct array indexing.** The C++ code indexes into the flattened NumPy arrays using row-major layout (`Y[row * N + col]`), avoiding Python object creation and reference counting for each element access.

**Input validation.** The C++ function validates array dimensions and CSR structure before processing, providing clear error messages for shape mismatches.

The C++ function signature:

```cpp
py::array_t<double> filter_communicated_message_cpp(
    py::array state_y,       // (dim_state, N) communicated messages
    py::array Go,            // (N, N) observation graph
    py::array offsets,       // (N+1,) CSR row offsets
    py::array neighbors,     // (E,) CSR neighbor list
    int D                    // robustness parameter
);
```

### OpenMP Parallelization

The filtering computation for each agent is independent: agent $i$'s filtered state depends only on the messages it receives from its neighbors, not on the filtered output of any other agent. This makes the outer loop over agents embarrassingly parallel.

A single OpenMP pragma parallelizes the outer loop:

```cpp
#pragma omp parallel for schedule(static)
for (int64_t agent_i = 0; agent_i < N; ++agent_i) {
    // Each agent's filtering is independent
    ...
}
```

The `schedule(static)` directive evenly distributes agents across threads, which is appropriate because all agents perform approximately the same amount of work (same number of neighbors in the grid topology). The number of OpenMP threads is configured via the `OMP_NUM_THREADS` environment variable, set to 8 by default.

**Platform-specific compilation.** The build system (`setup.py`) detects the operating system and applies appropriate compiler flags:

| Platform | Compiler | Optimization | OpenMP |
|----------|----------|-------------|--------|
| Linux    | GCC      | `-O3`       | `-fopenmp` |
| macOS    | Clang    | `-O3`       | Not enabled (requires `libomp`) |
| Windows  | MSVC     | `/O2`       | `/openmp` |

### Results

The C++ port with OpenMP achieved a **94x speedup** in the `filter_communicated_message` function and a **4x overall runtime reduction** on the 15x15 grid benchmark. The speedup comes from three sources: elimination of Python interpreter overhead (estimated 10-20x), compiler optimizations at `-O3` (estimated 2-3x), and OpenMP parallelism across 8 cores (up to 8x on Linux).

## 7.3.x.5 Convergence Optimization: Adaptive Step Sizes and Nesterov Acceleration

### Motivation

The baseline algorithm uses a fixed step size $\alpha = 1/40$ for all agents and all iterations. In adversarial settings, different agents experience different gradient magnitudes depending on their proximity to adversarial nodes and their position in the network. A fixed step size must be conservative enough for the worst-case agent, which slows convergence for all other agents.

Adaptive step size methods from the machine learning optimization literature address this by maintaining per-component learning rates that adjust based on gradient history.

### Adaptive Step Size Controller (`adaptive_step.py`)

A new module `adaptive_step.py` implements a unified `AdaptiveStepController` class supporting six methods:

| Method | Description | Key Property |
|--------|-------------|-------------|
| Constant | Fixed step size $\alpha$ | Baseline |
| AdaGrad | $\alpha / (\sqrt{\sum g_t^2} + \epsilon)$ | Adapts to gradient magnitude; decaying LR |
| RMSProp | $\alpha / (\sqrt{E[g^2]_t} + \epsilon)$ | Non-accumulating via exponential moving average |
| Adam | Momentum + adaptive LR with bias correction | Best general-purpose; combines benefits of momentum and RMSProp |
| AMSGrad | Adam with $\hat{v}_{\max} = \max(\hat{v}_{\max}, \hat{v}_t)$ | Non-increasing step sizes; stronger convergence guarantees |
| NAdam | Adam with Nesterov look-ahead momentum | Often faster convergence than Adam |

The controller exposes a unified `get_update(gradients)` interface that returns the full update vector to subtract from the state, handling the differences between methods (momentum-based vs. step-size-only) internally.

### Accelerated GRANE (Nesterov Extrapolation)

In addition to adaptive step sizes applied to the gradient, we implemented an accelerated variant of the GRANE algorithm using Nesterov-style momentum applied directly to the filtered state:

$$y_k = v_k + \beta (v_k - v_{k-1})$$
$$x_{k+1} = y_k - \alpha \nabla J(y_k)$$

where $v_k$ is the filtered state at iteration $k$, $\beta \in [0, 1)$ is the momentum coefficient (default 0.9), and $\alpha$ is the step size. The key idea is that instead of computing the gradient at the current filtered state $v_k$, we extrapolate forward using the momentum term and compute the gradient at the extrapolated point $y_k$. This look-ahead mechanism can accelerate convergence, particularly in settings where the objective is well-conditioned.

The Nesterov acceleration is applied at the state level (after filtering), not at the gradient level, which means it does not require additional communication rounds and has negligible computational overhead.

### Integration with the Main Algorithm

The step size method is configured via command-line arguments and can be changed dynamically at runtime:

```
python resilient.py --mode adaptive --step-method adam --base-lr 0.1
python resilient.py --mode adaptive --step-method accelerated --base-lr 0.05 --momentum 0.9
python resilient.py --mode compare  # Benchmarks all methods side-by-side
```

The `compare` mode runs all specified methods from the same initial state and produces convergence plots for direct comparison.

## 7.3.x.6 Visualization

An animation framework was added to visualize agent trajectories over time. The `animate_position_comparison` function produces side-by-side animations comparing constant step size vs. Nesterov-accelerated convergence, exported as GIF files via matplotlib's `PillowWriter`. Each frame shows the 2D positions of all agents (honest and adversarial) along with the Nash equilibrium targets, with a real-time error readout in the title.

## 7.3.x.7 Summary of Changes from Baseline

The table below summarizes all modifications to the baseline implementation:

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `resilient.py` | Modified | 755 (was 324) | CSR conversion, C++ binding integration, adaptive step controller integration, Nesterov acceleration, animation, CLI |
| `adaptive_step.py` | New | 311 | Adaptive step size controller (AdaGrad, RMSProp, Adam, AMSGrad, NAdam) |
| `remove_d.cpp` | New | 180 | C++ implementation of message filtering with OpenMP |
| `setup.py` | New | 40 | Build configuration for C++ extension with platform-specific flags |
| `info_robust_graph.py` | Unchanged | 295 | Graph theory utilities (baseline code) |

Total new code: approximately 1,000 lines across 4 files (3 new, 1 modified).

| Metric | Baseline | Final Design | Improvement |
|--------|----------|-------------|-------------|
| Max agents (32 GB) | ~150 | 1,000+ | ~7x |
| Memory complexity | $O(N^4)$ | $O(N^3)$ | Order reduction |
| Filter runtime (221 agents) | 146 s | 1.6 s | 94x speedup |
| Overall runtime (221 agents) | 333 s | ~83 s | 4x speedup |
| Step size methods | 1 (constant) | 7 (constant, AdaGrad, RMSProp, Adam, AMSGrad, NAdam, Nesterov) | 6 new methods |
| Parallelism | None | OpenMP (8 threads) | Multi-core |
| Language | Python only | Python + C++ | Native extension |
