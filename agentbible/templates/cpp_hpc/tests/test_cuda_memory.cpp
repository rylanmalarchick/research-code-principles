// SPDX-License-Identifier: MIT
// Copyright (c) {{YEAR}} {{AUTHOR_NAME}}

/**
 * @file test_cuda_memory.cpp
 * @brief Tests for CudaMemory RAII wrapper
 *
 * These tests verify correct memory management and error handling.
 * GPU tests are skipped if CUDA is not available.
 */

#include "CudaMemory.hpp"

#include <gtest/gtest.h>

namespace research {
namespace {

#ifdef CUDA_ENABLED

class CudaMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA device is available
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            GTEST_SKIP() << "No CUDA device available";
        }
    }
};

TEST_F(CudaMemoryTest, DefaultConstructorCreatesNull) {
    CudaMemory<float> mem;
    EXPECT_EQ(mem.get(), nullptr);
    EXPECT_EQ(mem.count(), 0);
    EXPECT_FALSE(mem);
}

TEST_F(CudaMemoryTest, AllocationSucceeds) {
    CudaMemory<float> mem(1024);
    EXPECT_NE(mem.get(), nullptr);
    EXPECT_EQ(mem.count(), 1024);
    EXPECT_EQ(mem.size_bytes(), 1024 * sizeof(float));
    EXPECT_TRUE(mem);
}

TEST_F(CudaMemoryTest, ZeroCountThrows) {
    EXPECT_THROW(CudaMemory<float>(0), std::invalid_argument);
}

TEST_F(CudaMemoryTest, MoveConstructorTransfersOwnership) {
    CudaMemory<float> original(512);
    float* originalPtr = original.get();

    CudaMemory<float> moved(std::move(original));

    EXPECT_EQ(moved.get(), originalPtr);
    EXPECT_EQ(moved.count(), 512);
    EXPECT_EQ(original.get(), nullptr);
    EXPECT_EQ(original.count(), 0);
}

TEST_F(CudaMemoryTest, MoveAssignmentTransfersOwnership) {
    CudaMemory<float> original(512);
    float* originalPtr = original.get();

    CudaMemory<float> target(256);
    target = std::move(original);

    EXPECT_EQ(target.get(), originalPtr);
    EXPECT_EQ(target.count(), 512);
    EXPECT_EQ(original.get(), nullptr);
}

TEST_F(CudaMemoryTest, CopyFromHostAndBack) {
    constexpr size_t count = 128;
    CudaMemory<int> mem(count);

    // Create host data
    std::vector<int> hostData(count);
    for (size_t i = 0; i < count; ++i) {
        hostData[i] = static_cast<int>(i * 2);
    }

    // Copy to device
    mem.copyFromHost(hostData.data(), count);

    // Copy back
    std::vector<int> result(count);
    mem.copyToHost(result.data(), count);

    // Verify
    EXPECT_EQ(hostData, result);
}

TEST_F(CudaMemoryTest, CopyFromHostExceedingCountThrows) {
    CudaMemory<int> mem(64);
    std::vector<int> hostData(128);

    EXPECT_THROW(mem.copyFromHost(hostData.data(), 128), std::out_of_range);
}

TEST_F(CudaMemoryTest, CopyToHostExceedingCountThrows) {
    CudaMemory<int> mem(64);
    std::vector<int> hostData(128);

    EXPECT_THROW(mem.copyToHost(hostData.data(), 128), std::out_of_range);
}

TEST_F(CudaMemoryTest, ZeroMemory) {
    constexpr size_t count = 256;
    CudaMemory<int> mem(count);

    // Initialize with non-zero
    std::vector<int> hostData(count, 42);
    mem.copyFromHost(hostData.data(), count);

    // Zero the memory
    mem.zero();

    // Copy back and verify
    std::vector<int> result(count, 99);
    mem.copyToHost(result.data(), count);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(result[i], 0) << "Index " << i << " not zeroed";
    }
}

#else  // !CUDA_ENABLED

class CudaMemoryNoCudaTest : public ::testing::Test {};

TEST_F(CudaMemoryNoCudaTest, AllocationThrowsWithoutCuda) {
    EXPECT_THROW(CudaMemory<float>(1024), std::runtime_error);
}

TEST_F(CudaMemoryNoCudaTest, DefaultConstructorWorks) {
    CudaMemory<float> mem;
    EXPECT_EQ(mem.get(), nullptr);
    EXPECT_EQ(mem.count(), 0);
}

#endif  // CUDA_ENABLED

}  // namespace
}  // namespace research
