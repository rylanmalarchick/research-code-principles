# CUDA Kernel Optimization Prompt

Use this when writing GPU-accelerated code.

---

## RAII for GPU Memory (CRITICAL)

Never use raw cudaMalloc/cudaFree. Use the CudaMemory wrapper:

```cpp
#include "CudaMemory.hpp"

void simulate() {
    // Automatic cleanup when out of scope
    research::CudaMemory<cuDoubleComplex> d_state(1 << num_qubits);

    // Use d_state.get() for raw pointer
    myKernel<<<blocks, threads>>>(d_state.get());
}  // Memory freed automatically
```

---

## Error Checking (ALWAYS)

```cpp
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            throw std::runtime_error(                                   \
                std::string("CUDA error at ") + __FILE__ + ":" +       \
                std::to_string(__LINE__) + ": " +                      \
                cudaGetErrorString(err));                               \
        }                                                               \
    } while (0)

// Usage
CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
```

---

## Kernel Launch Best Practices

```cpp
// Calculate grid dimensions
const int threads_per_block = 256;  // Typically 128-512
const int num_elements = 1 << num_qubits;
const int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

// Launch with error checking
myKernel<<<blocks, threads_per_block>>>(d_state.get(), num_elements);
CUDA_CHECK(cudaGetLastError());      // Check launch errors
CUDA_CHECK(cudaDeviceSynchronize()); // Wait and check execution errors
```

---

## Memory Coalescing

```cpp
// GOOD: Coalesced access (threads access consecutive memory)
__global__ void goodKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;  // Thread i accesses element i
    }
}

// BAD: Strided access
__global__ void badKernel(float* data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * stride < n) {
        data[idx * stride] *= 2.0f;  // Non-coalesced!
    }
}
```

---

## Shared Memory Tiling

```cpp
__global__ void tiledKernel(float* data, int n) {
    __shared__ float tile[TILE_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load to shared memory
    if (idx < n) {
        tile[threadIdx.x] = data[idx];
    }
    __syncthreads();

    // Compute using shared memory (much faster)
    // ...
}
```

---

## Device Pointer Naming Convention

```cpp
float* d_input;   // d_ prefix for device pointers
float* h_output;  // h_ prefix for host pointers (pinned)
float* input;     // Regular host pointer
```
