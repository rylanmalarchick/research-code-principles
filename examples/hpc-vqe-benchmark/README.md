# HPC VQE Benchmark Example

Demonstrates the **117x speedup** achieved in the QuantumVQE project through systematic optimization.

## The Optimization Journey

| Optimization | Time | Speedup | Cumulative |
|--------------|------|---------|------------|
| Baseline (Serial Python) | 593.95s | 1x | 1x |
| + JAX JIT Compilation | 27.32s | 21.7x | 21.7x |
| + GPU Backend (Lightning) | 9.78s | 2.8x | 60.7x |
| + MPI Parallelization (4 ranks) | 5.04s | 1.9x | **117.8x** |

## Key Techniques

### 1. JIT Compilation (21.7x)

```python
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(1,))
def compute_energy(params: jnp.ndarray, n_qubits: int) -> float:
    """JIT-compiled energy computation.
    
    The @jax.jit decorator compiles this function to XLA,
    eliminating Python interpreter overhead.
    
    static_argnums=(1,) tells JAX that n_qubits won't change,
    allowing more aggressive optimization.
    """
    # ... quantum circuit evaluation
    return energy
```

### 2. GPU Backend (2.8x additional)

```python
import pennylane as qml

# CPU backend (baseline)
dev_cpu = qml.device("lightning.qubit", wires=n_qubits)

# GPU backend (2.8x faster for 4+ qubits)
dev_gpu = qml.device("lightning.gpu", wires=n_qubits)
```

### 3. MPI Parallelization (1.9x additional)

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Distribute bond lengths across MPI ranks
all_bonds = np.linspace(0.1, 3.0, 100)
my_bonds = np.array_split(all_bonds, size)[rank]

# Each rank computes its portion
my_results = [compute_energy(bond) for bond in my_bonds]

# Gather results
all_results = comm.gather(my_results, root=0)
```

## Files

| File | Purpose |
|------|---------|
| `vqe_benchmark.py` | Complete benchmark implementation |
| `run_benchmark.sh` | PBS job script for HPC cluster |
| `results/` | Benchmark results and plots |

## Running the Benchmark

### Local (CPU only)
```bash
python vqe_benchmark.py --backend cpu --n-bonds 10
```

### HPC Cluster (GPU + MPI)
```bash
qsub run_benchmark.sh
```

## Results

The benchmark demonstrates that research code can achieve HPC-grade performance
while maintaining the quality standards defined in this repository:

- **800+ tests** with 74% coverage
- **Type hints** throughout
- **CI/CD pipeline** with automated testing
- **Reproducible** with pinned dependencies and fixed seeds

## Citation

If you use this benchmark, please cite:

```bibtex
@software{malarchick2025vqe,
  author = {Malarchick, Rylan},
  title = {QuantumVQE: GPU-Accelerated Variational Quantum Eigensolver},
  year = {2025},
  url = {https://github.com/rylanmalarchick/QuantumVQE}
}
```
