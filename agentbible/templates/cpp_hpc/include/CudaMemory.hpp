// SPDX-License-Identifier: MIT
// Copyright (c) {{YEAR}} {{AUTHOR_NAME}}

/**
 * @file CudaMemory.hpp
 * @brief RAII wrapper for CUDA device memory - zero manual cudaFree calls
 * @author {{AUTHOR_NAME}}
 * @date {{YEAR}}
 *
 * This class ensures CUDA memory is automatically freed when the object
 * goes out of scope, preventing memory leaks and ensuring exception safety.
 *
 * Usage:
 * @code
 * CudaMemory<cuDoubleComplex> d_state(1 << num_qubits);
 * // Use d_state.get() for raw pointer
 * // Memory automatically freed on destruction
 * @endcode
 */
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>

/**
 * @brief Check CUDA call and throw on error with file/line info.
 *
 * @throws std::runtime_error on CUDA failure
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error in ") + __FILE__ +\
                                     ":" + std::to_string(__LINE__) + ": " +   \
                                     cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

namespace research {

/**
 * @brief RAII wrapper for CUDA device memory.
 *
 * Features:
 * - Automatic memory cleanup on destruction
 * - Move semantics (no copy)
 * - Safe default construction (nullptr)
 * - Exception-safe allocation
 *
 * @tparam T Element type (e.g., float, cuDoubleComplex)
 *
 * Thread Safety: Not thread-safe. Each CudaMemory object should be owned
 * by a single thread.
 */
template <typename T>
class CudaMemory {
public:
    /**
     * @brief Default constructor - creates empty (nullptr) memory.
     */
    CudaMemory() noexcept = default;

    /**
     * @brief Allocate device memory for count elements.
     *
     * @param count Number of elements to allocate
     * @throws std::runtime_error if cudaMalloc fails
     * @throws std::invalid_argument if count is 0
     */
    explicit CudaMemory(size_t count) : count_(count) {
        if (count == 0) {
            throw std::invalid_argument("CudaMemory: count must be > 0");
        }
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    /**
     * @brief Destructor - frees device memory if owned.
     */
    ~CudaMemory() noexcept {
        if (ptr_) {
            cudaFree(ptr_);  // Ignore errors in destructor
        }
    }

    // Non-copyable
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;

    /**
     * @brief Move constructor - transfers ownership.
     */
    CudaMemory(CudaMemory&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    /**
     * @brief Move assignment - transfers ownership, frees existing memory.
     */
    CudaMemory& operator=(CudaMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    /**
     * @brief Get raw device pointer.
     * @return Device pointer (may be nullptr)
     */
    [[nodiscard]] T* get() noexcept { return ptr_; }

    /**
     * @brief Get raw device pointer (const).
     * @return Device pointer (may be nullptr)
     */
    [[nodiscard]] const T* get() const noexcept { return ptr_; }

    /**
     * @brief Get number of elements.
     * @return Element count (0 if empty)
     */
    [[nodiscard]] size_t count() const noexcept { return count_; }

    /**
     * @brief Get size in bytes.
     * @return Size in bytes (0 if empty)
     */
    [[nodiscard]] size_t size_bytes() const noexcept {
        return count_ * sizeof(T);
    }

    /**
     * @brief Check if memory is allocated.
     * @return true if ptr_ is not nullptr
     */
    [[nodiscard]] explicit operator bool() const noexcept {
        return ptr_ != nullptr;
    }

    /**
     * @brief Copy data from host to device.
     *
     * @param host_data Pointer to host data
     * @param num_elements Number of elements to copy (must be <= count_)
     * @throws std::runtime_error on CUDA error
     * @throws std::out_of_range if num_elements > count_
     */
    void copyFromHost(const T* host_data, size_t num_elements) {
        if (num_elements > count_) {
            throw std::out_of_range(
                "CudaMemory::copyFromHost: num_elements (" +
                std::to_string(num_elements) + ") > count (" +
                std::to_string(count_) + ")");
        }
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, num_elements * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    /**
     * @brief Copy data from device to host.
     *
     * @param host_data Pointer to host buffer
     * @param num_elements Number of elements to copy (must be <= count_)
     * @throws std::runtime_error on CUDA error
     * @throws std::out_of_range if num_elements > count_
     */
    void copyToHost(T* host_data, size_t num_elements) const {
        if (num_elements > count_) {
            throw std::out_of_range(
                "CudaMemory::copyToHost: num_elements (" +
                std::to_string(num_elements) + ") > count (" +
                std::to_string(count_) + ")");
        }
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, num_elements * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

    /**
     * @brief Set all bytes to zero.
     * @throws std::runtime_error on CUDA error
     */
    void zero() {
        if (ptr_) {
            CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
        }
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

}  // namespace research

#else  // !CUDA_ENABLED

// Stub for non-CUDA builds
namespace research {

template <typename T>
class CudaMemory {
public:
    CudaMemory() = default;
    explicit CudaMemory(size_t) {
        throw std::runtime_error("CUDA support not enabled");
    }
    [[nodiscard]] T* get() noexcept { return nullptr; }
    [[nodiscard]] const T* get() const noexcept { return nullptr; }
    [[nodiscard]] size_t count() const noexcept { return 0; }
};

}  // namespace research

#endif  // CUDA_ENABLED
