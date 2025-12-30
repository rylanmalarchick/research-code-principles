# C++/HPC Research Starter Template

Production-grade C++17/CUDA project scaffold for research software.

## Quick Start

```bash
# Copy this template to your project
cp -r templates/cpp_hpc ~/my-project
cd ~/my-project

# Configure (without CUDA)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Configure (with CUDA)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON

# Build
cmake --build build -j$(nproc)

# Run tests
ctest --test-dir build --output-on-failure

# Run main executable
./build/my_research_project
```

## What's Included

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Modern CMake with zero-warning policy, GoogleTest, optional CUDA |
| `include/CudaMemory.hpp` | RAII wrapper for CUDA memory - zero manual cudaFree |
| `include/core.hpp` | Physical validation functions (unitarity, normalization) |
| `tests/test_*.cpp` | Example tests demonstrating specification-first approach |

## Configuration Highlights

### Zero-Warning Policy

The CMakeLists.txt enables strict warnings:
```cmake
-Wall -Wextra -Wpedantic -Wshadow -Wconversion ...
```

In CI (when `CI` environment variable is set), warnings are treated as errors (`-Werror`).

### RAII for CUDA Memory

The `CudaMemory<T>` class ensures GPU memory is automatically freed:

```cpp
#include "CudaMemory.hpp"

void simulate() {
    // Allocate 2^20 complex numbers on GPU
    research::CudaMemory<cuDoubleComplex> d_state(1 << 20);
    
    // Use d_state.get() for raw pointer in kernels
    myKernel<<<blocks, threads>>>(d_state.get());
    
    // Memory automatically freed when d_state goes out of scope
    // No manual cudaFree needed!
}
```

### GoogleTest Integration

Tests are automatically fetched and built:

```cpp
TEST(MyTest, SomethingWorks) {
    EXPECT_EQ(compute(), expectedValue);
}
```

Run with: `ctest --test-dir build`

## Physical Validation

The `core.hpp` module provides:

```cpp
#include "core.hpp"

// Validate quantum operations
research::checkUnitarity(gateMatrix, dimension);  // U†U = I
research::checkNormalized(stateVector);            // ⟨ψ|ψ⟩ = 1
research::checkHermitian(hamiltonian, dimension);  // H = H†
```

## Next Steps

1. Update project name in `CMakeLists.txt`
2. Add source files to `src/`, headers to `include/`
3. Write tests in `tests/` (specification first!)
4. Add CUDA kernels to `src/cuda/` if using GPU
5. Configure CI (see `.github/workflows/ci.yml` in parent)

## Directory Structure

```
cpp_hpc/
├── CMakeLists.txt          # Build configuration
├── include/
│   ├── CudaMemory.hpp      # RAII CUDA wrapper
│   └── core.hpp            # Validation utilities
├── src/
│   ├── core.cpp            # Validation implementations
│   └── main.cpp            # Entry point
└── tests/
    ├── test_main.cpp       # GoogleTest main
    ├── test_core.cpp       # Validation tests
    └── test_cuda_memory.cpp # CUDA tests
```

## License

MIT
