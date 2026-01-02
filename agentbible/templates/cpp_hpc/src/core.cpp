// SPDX-License-Identifier: MIT
// Copyright (c) {{YEAR}} {{AUTHOR_NAME}}

#include "core.hpp"

#include <cmath>
#include <numeric>

namespace research {

void checkUnitarity(const std::vector<std::complex<double>>& matrix, size_t n,
                    double tolerance, const std::string& name) {
    if (matrix.size() != n * n) {
        throw std::invalid_argument(
            "Matrix" + (name.empty() ? "" : " '" + name + "'") +
            " size mismatch: expected " + std::to_string(n * n) + ", got " +
            std::to_string(matrix.size()));
    }

    // Compute U†U and check against identity
    double maxError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            std::complex<double> sum(0.0, 0.0);
            for (size_t k = 0; k < n; ++k) {
                // U†[i,k] * U[k,j] = conj(U[k,i]) * U[k,j]
                sum += std::conj(matrix[k * n + i]) * matrix[k * n + j];
            }
            double expected = (i == j) ? 1.0 : 0.0;
            double error = std::abs(sum - std::complex<double>(expected, 0.0));
            maxError = std::max(maxError, error);
        }
    }

    if (maxError > tolerance) {
        throw std::invalid_argument(
            "Matrix" + (name.empty() ? "" : " '" + name + "'") +
            " is not unitary: max|U†U - I| = " + std::to_string(maxError) +
            " (tolerance: " + std::to_string(tolerance) + ")");
    }
}

void checkHermitian(const std::vector<std::complex<double>>& matrix, size_t n,
                    double tolerance, const std::string& name) {
    if (matrix.size() != n * n) {
        throw std::invalid_argument(
            "Matrix" + (name.empty() ? "" : " '" + name + "'") +
            " size mismatch: expected " + std::to_string(n * n) + ", got " +
            std::to_string(matrix.size()));
    }

    double maxError = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // H[i,j] should equal conj(H[j,i])
            auto diff = matrix[i * n + j] - std::conj(matrix[j * n + i]);
            maxError = std::max(maxError, std::abs(diff));
        }
    }

    if (maxError > tolerance) {
        throw std::invalid_argument(
            "Matrix" + (name.empty() ? "" : " '" + name + "'") +
            " is not Hermitian: max|H - H†| = " + std::to_string(maxError));
    }
}

void checkNormalized(const std::vector<std::complex<double>>& state,
                     double tolerance, const std::string& name) {
    if (state.empty()) {
        throw std::invalid_argument(
            "State" + (name.empty() ? "" : " '" + name + "'") + " is empty");
    }

    double normSquared = 0.0;
    for (const auto& amp : state) {
        normSquared += std::norm(amp);  // |z|^2
    }
    double norm = std::sqrt(normSquared);

    if (std::abs(norm - 1.0) > tolerance) {
        throw std::invalid_argument(
            "State" + (name.empty() ? "" : " '" + name + "'") +
            " is not normalized: |⟨ψ|ψ⟩| = " + std::to_string(norm) +
            " (expected 1.0)");
    }
}

}  // namespace research
