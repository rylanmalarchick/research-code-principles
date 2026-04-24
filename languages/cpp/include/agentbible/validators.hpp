#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "agentbible/provenance.hpp"

namespace agentbible::detail {

#define AGENTBIBLE_STRINGIFY_IMPL(x) #x
#define AGENTBIBLE_STRINGIFY(x) AGENTBIBLE_STRINGIFY_IMPL(x)

inline void handle_failure(
    const std::string& check_name,
    double rtol,
    double atol,
    const std::string& norm_used,
    const std::string& error_message) {
    emit_provenance(std::cerr, {{check_name, false, rtol, atol, norm_used, error_message}});
    std::cerr << '\n';
#if defined(AGENTBIBLE_ON_FAIL)
    constexpr std::string_view on_fail = AGENTBIBLE_STRINGIFY(AGENTBIBLE_ON_FAIL);
    if constexpr (on_fail == "ABORT") {
        std::abort();
    } else {
        throw std::runtime_error(error_message);
    }
#else
    throw std::runtime_error(error_message);
#endif
}

template <typename T>
inline bool is_finite_value(const T value) {
    return std::isfinite(static_cast<double>(value));
}

template <>
inline bool is_finite_value(const std::complex<double> value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

template <typename T>
inline void validate_finite(const T* values, const std::size_t n) {
    for (std::size_t index = 0; index < n; ++index) {
        if (!is_finite_value(values[index])) {
            handle_failure("finite_array", 0.0, 0.0, "n/a", "non-finite value detected");
        }
    }
}

inline void validate_positive(const double* values, const std::size_t n) {
    validate_finite(values, n);
    for (std::size_t index = 0; index < n; ++index) {
        if (!(values[index] > 0.0)) {
            handle_failure("positive_array", 0.0, 0.0, "n/a", "non-positive value detected");
        }
    }
}

inline void validate_non_negative(const double* values, const std::size_t n) {
    validate_finite(values, n);
    for (std::size_t index = 0; index < n; ++index) {
        if (values[index] < 0.0) {
            handle_failure("non_negative_array", 0.0, 0.0, "n/a", "negative value detected");
        }
    }
}

inline void validate_probability(const double* values, const std::size_t n) {
    validate_finite(values, n);
    for (std::size_t index = 0; index < n; ++index) {
        if (values[index] < 0.0 || values[index] > 1.0) {
            handle_failure("probability_array", 0.0, 0.0, "n/a", "probability outside [0, 1]");
        }
    }
}

inline void validate_normalized_l1(const double* values, const std::size_t n, const double atol) {
    validate_finite(values, n);
    double total = 0.0;
    for (std::size_t index = 0; index < n; ++index) {
        total += values[index];
    }
    if (std::abs(total - 1.0) > atol) {
        handle_failure("normalized_l1", 0.0, atol, "l1", "L1 sum is not within tolerance of 1");
    }
}

inline void validate_symmetric(const double* matrix, const std::size_t n, const double atol) {
    validate_finite(matrix, n * n);
    double max_delta = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            const double delta = std::abs(matrix[row * n + col] - matrix[col * n + row]);
            max_delta = std::max(max_delta, delta);
        }
    }
    if (max_delta > atol) {
        handle_failure("symmetric", 0.0, atol, "max_elementwise", "matrix is not symmetric");
    }
}

inline void validate_unitary(
    const std::complex<double>* matrix,
    const std::size_t n,
    const double rtol,
    const double atol) {
    validate_finite(matrix, n * n);
    double residual_squared = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            std::complex<double> product = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                product += std::conj(matrix[k * n + row]) * matrix[k * n + col];
            }
            const auto target = (row == col) ? std::complex<double>(1.0, 0.0) : std::complex<double>(0.0, 0.0);
            residual_squared += std::norm(product - target);
        }
    }
    const double residual = std::sqrt(residual_squared);
    if (residual > atol + (rtol * static_cast<double>(n))) {
        handle_failure("unitary", rtol, atol, "frobenius", "matrix is not unitary");
    }
}

inline void validate_positive_definite(const double* matrix, const std::size_t n) {
    validate_finite(matrix, n * n);
    std::vector<double> lower(n * n, 0.0);
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
            double sum = matrix[row * n + col];
            for (std::size_t k = 0; k < col; ++k) {
                sum -= lower[row * n + k] * lower[col * n + k];
            }
            if (row == col) {
                if (sum <= 0.0) {
                    handle_failure("positive_definite", 0.0, 0.0, "n/a", "Cholesky factorization failed");
                }
                lower[row * n + col] = std::sqrt(sum);
            } else {
                lower[row * n + col] = sum / lower[col * n + col];
            }
        }
    }
}

}  // namespace agentbible::detail

#ifdef AGENTBIBLE_DISABLE
#define AGENTBIBLE_VALIDATE_FINITE(ptr, n) ((void)0)
#define AGENTBIBLE_VALIDATE_POSITIVE(ptr, n) ((void)0)
#define AGENTBIBLE_VALIDATE_NON_NEGATIVE(ptr, n) ((void)0)
#define AGENTBIBLE_VALIDATE_PROBABILITY(ptr, n) ((void)0)
#define AGENTBIBLE_VALIDATE_NORMALIZED_L1(ptr, n, atol) ((void)0)
#define AGENTBIBLE_VALIDATE_SYMMETRIC(mat, n, atol) ((void)0)
#define AGENTBIBLE_VALIDATE_UNITARY(mat, n, rtol, atol) ((void)0)
#define AGENTBIBLE_VALIDATE_POSITIVE_DEFINITE(mat, n) ((void)0)
#else
#define AGENTBIBLE_VALIDATE_FINITE(ptr, n) ::agentbible::detail::validate_finite((ptr), (n))
#define AGENTBIBLE_VALIDATE_POSITIVE(ptr, n) ::agentbible::detail::validate_positive((ptr), (n))
#define AGENTBIBLE_VALIDATE_NON_NEGATIVE(ptr, n) ::agentbible::detail::validate_non_negative((ptr), (n))
#define AGENTBIBLE_VALIDATE_PROBABILITY(ptr, n) ::agentbible::detail::validate_probability((ptr), (n))
#define AGENTBIBLE_VALIDATE_NORMALIZED_L1(ptr, n, atol) ::agentbible::detail::validate_normalized_l1((ptr), (n), (atol))
#define AGENTBIBLE_VALIDATE_SYMMETRIC(mat, n, atol) ::agentbible::detail::validate_symmetric((mat), (n), (atol))
#define AGENTBIBLE_VALIDATE_UNITARY(mat, n, rtol, atol) ::agentbible::detail::validate_unitary((mat), (n), (rtol), (atol))
#define AGENTBIBLE_VALIDATE_POSITIVE_DEFINITE(mat, n) ::agentbible::detail::validate_positive_definite((mat), (n))
#endif
