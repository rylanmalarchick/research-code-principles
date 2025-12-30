// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Your Name

/**
 * @file test_core.cpp
 * @brief Tests for core validation functions
 *
 * These tests demonstrate the specification-before-code principle.
 */

#include "core.hpp"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>

namespace research {
namespace {

// ============================================================================
// Test Fixtures
// ============================================================================

// Pauli X gate: [[0, 1], [1, 0]]
std::vector<std::complex<double>> pauliX() {
    return {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
}

// Pauli Z gate: [[1, 0], [0, -1]]
std::vector<std::complex<double>> pauliZ() {
    return {{1, 0}, {0, 0}, {0, 0}, {-1, 0}};
}

// Hadamard gate: [[1, 1], [1, -1]] / sqrt(2)
std::vector<std::complex<double>> hadamard() {
    const double s = 1.0 / std::sqrt(2.0);
    return {{s, 0}, {s, 0}, {s, 0}, {-s, 0}};
}

// Identity 2x2
std::vector<std::complex<double>> identity2() {
    return {{1, 0}, {0, 0}, {0, 0}, {1, 0}};
}

// |0⟩ state
std::vector<std::complex<double>> zeroState() { return {{1, 0}, {0, 0}}; }

// |+⟩ state
std::vector<std::complex<double>> plusState() {
    const double s = 1.0 / std::sqrt(2.0);
    return {{s, 0}, {s, 0}};
}

// ============================================================================
// checkUnitarity Tests
// ============================================================================

class CheckUnitarityTest : public ::testing::Test {};

TEST_F(CheckUnitarityTest, PauliXIsUnitary) {
    EXPECT_NO_THROW(checkUnitarity(pauliX(), 2));
}

TEST_F(CheckUnitarityTest, PauliZIsUnitary) {
    EXPECT_NO_THROW(checkUnitarity(pauliZ(), 2));
}

TEST_F(CheckUnitarityTest, HadamardIsUnitary) {
    EXPECT_NO_THROW(checkUnitarity(hadamard(), 2));
}

TEST_F(CheckUnitarityTest, IdentityIsUnitary) {
    EXPECT_NO_THROW(checkUnitarity(identity2(), 2));
}

TEST_F(CheckUnitarityTest, NonUnitaryThrows) {
    // [[1, 1], [0, 1]] is not unitary
    std::vector<std::complex<double>> nonUnitary = {
        {1, 0}, {1, 0}, {0, 0}, {1, 0}};
    EXPECT_THROW(checkUnitarity(nonUnitary, 2), std::invalid_argument);
}

TEST_F(CheckUnitarityTest, WrongSizeThrows) {
    auto matrix = pauliX();
    EXPECT_THROW(checkUnitarity(matrix, 3), std::invalid_argument);
}

TEST_F(CheckUnitarityTest, ErrorMessageIncludesName) {
    std::vector<std::complex<double>> nonUnitary = {
        {2, 0}, {0, 0}, {0, 0}, {1, 0}};
    try {
        checkUnitarity(nonUnitary, 2, 1e-10, "MyGate");
        FAIL() << "Expected exception";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("MyGate"), std::string::npos);
    }
}

// ============================================================================
// checkHermitian Tests
// ============================================================================

class CheckHermitianTest : public ::testing::Test {};

TEST_F(CheckHermitianTest, PauliMatricesAreHermitian) {
    EXPECT_NO_THROW(checkHermitian(pauliX(), 2));
    EXPECT_NO_THROW(checkHermitian(pauliZ(), 2));
}

TEST_F(CheckHermitianTest, RealSymmetricIsHermitian) {
    // [[1, 2], [2, 3]]
    std::vector<std::complex<double>> realSym = {
        {1, 0}, {2, 0}, {2, 0}, {3, 0}};
    EXPECT_NO_THROW(checkHermitian(realSym, 2));
}

TEST_F(CheckHermitianTest, NonHermitianThrows) {
    // [[0, 1], [0, 0]] is not Hermitian
    std::vector<std::complex<double>> nonHerm = {
        {0, 0}, {1, 0}, {0, 0}, {0, 0}};
    EXPECT_THROW(checkHermitian(nonHerm, 2), std::invalid_argument);
}

// ============================================================================
// checkNormalized Tests
// ============================================================================

class CheckNormalizedTest : public ::testing::Test {};

TEST_F(CheckNormalizedTest, ZeroStateIsNormalized) {
    EXPECT_NO_THROW(checkNormalized(zeroState()));
}

TEST_F(CheckNormalizedTest, PlusStateIsNormalized) {
    EXPECT_NO_THROW(checkNormalized(plusState()));
}

TEST_F(CheckNormalizedTest, UnnormalizedThrows) {
    std::vector<std::complex<double>> unnorm = {{1, 0}, {1, 0}};  // norm = sqrt(2)
    EXPECT_THROW(checkNormalized(unnorm), std::invalid_argument);
}

TEST_F(CheckNormalizedTest, ZeroVectorThrows) {
    std::vector<std::complex<double>> zero = {{0, 0}, {0, 0}};
    EXPECT_THROW(checkNormalized(zero), std::invalid_argument);
}

TEST_F(CheckNormalizedTest, EmptyVectorThrows) {
    std::vector<std::complex<double>> empty;
    EXPECT_THROW(checkNormalized(empty), std::invalid_argument);
}

}  // namespace
}  // namespace research
