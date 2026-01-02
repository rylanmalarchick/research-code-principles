// SPDX-License-Identifier: MIT
// Copyright (c) {{YEAR}} {{AUTHOR_NAME}}

/**
 * @file core.hpp
 * @brief Core utilities and validation functions
 * @author {{AUTHOR_NAME}}
 * @date {{YEAR}}
 */
#pragma once

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace research {

/**
 * @brief Validate that a complex matrix is unitary (U†U = I).
 *
 * @param matrix Row-major complex matrix (size must be n*n)
 * @param n Matrix dimension
 * @param tolerance Numerical tolerance for comparison
 * @param name Optional name for error messages
 *
 * @throws std::invalid_argument if matrix is not unitary
 *
 * @references
 * Nielsen & Chuang, "Quantum Computation and Quantum Information", Sec 2.1.4
 */
void checkUnitarity(const std::vector<std::complex<double>>& matrix, size_t n,
                    double tolerance = 1e-10, const std::string& name = "");

/**
 * @brief Validate that a complex matrix is Hermitian (H = H†).
 *
 * @param matrix Row-major complex matrix (size must be n*n)
 * @param n Matrix dimension
 * @param tolerance Numerical tolerance for comparison
 * @param name Optional name for error messages
 *
 * @throws std::invalid_argument if matrix is not Hermitian
 */
void checkHermitian(const std::vector<std::complex<double>>& matrix, size_t n,
                    double tolerance = 1e-10, const std::string& name = "");

/**
 * @brief Validate that a state vector is normalized (⟨ψ|ψ⟩ = 1).
 *
 * @param state Complex state vector
 * @param tolerance Numerical tolerance for comparison
 * @param name Optional name for error messages
 *
 * @throws std::invalid_argument if state is not normalized
 */
void checkNormalized(const std::vector<std::complex<double>>& state,
                     double tolerance = 1e-10, const std::string& name = "");

}  // namespace research
