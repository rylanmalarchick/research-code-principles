#include <array>
#include <complex>
#include <functional>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "agentbible/agentbible.hpp"

namespace {

std::array<std::complex<double>, 4> hadamard() {
    return {
        std::complex<double>(0.7071067811865475, 0.0),
        std::complex<double>(0.7071067811865475, 0.0),
        std::complex<double>(0.7071067811865475, 0.0),
        std::complex<double>(-0.7071067811865475, 0.0),
    };
}

nlohmann::json capture_failure_json(const std::function<void()>& fn) {
    testing::internal::CaptureStderr();
    EXPECT_THROW(fn(), std::runtime_error);
    return nlohmann::json::parse(testing::internal::GetCapturedStderr());
}

}  // namespace

TEST(Conformance, UnitaryPassesAtDefaultTolerance) {
    const auto base = hadamard();
    EXPECT_NO_THROW(AGENTBIBLE_VALIDATE_UNITARY(base.data(), 2, 1e-10, 1e-12));

    auto perturbed = base;
    const auto theta = std::sqrt(2.0) * 1e-9;
    const auto phase = std::exp(std::complex<double>(0.0, theta));
    perturbed[1] *= phase;
    perturbed[3] *= phase;
    EXPECT_NO_THROW(AGENTBIBLE_VALIDATE_UNITARY(perturbed.data(), 2, 1e-10, 1e-12));
}

TEST(Conformance, UnitaryFailsAfterLargePerturbation) {
    auto failing = hadamard();
    failing[0] += std::complex<double>(1e-5, 0.0);
    const auto payload = capture_failure_json(
        [&]() { AGENTBIBLE_VALIDATE_UNITARY(failing.data(), 2, 1e-10, 1e-12); });
    EXPECT_EQ(payload["checks_passed"][0]["check_name"], "unitary");
}

TEST(Conformance, NaNInjectionFailsFinite) {
    const double values[] = {1.0, std::numeric_limits<double>::quiet_NaN()};
    const auto payload =
        capture_failure_json([&]() { AGENTBIBLE_VALIDATE_FINITE(values, 2); });
    EXPECT_EQ(payload["checks_passed"][0]["check_name"], "finite_array");
}

TEST(Conformance, ProvenanceRoundtrip) {
    const auto payload = agentbible::serialize_record(
        {{"unitary", true, 1e-10, 1e-12, "frobenius", ""}});
    const auto parsed = nlohmann::json::parse(payload);
    const auto roundtrip = nlohmann::json::parse(parsed.dump());
    EXPECT_EQ(roundtrip["spec_version"], "1.0");
    EXPECT_EQ(roundtrip["language"], "cpp");
    EXPECT_EQ(roundtrip["checks_passed"][0]["check_name"], "unitary");
}
