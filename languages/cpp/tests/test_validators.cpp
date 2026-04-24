#include <complex>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "agentbible/agentbible.hpp"

namespace {

void expect_record_shape(const nlohmann::json& payload) {
    ASSERT_EQ(payload.at("spec_version"), "1.0");
    ASSERT_EQ(payload.at("language"), "cpp");
    ASSERT_TRUE(payload.contains("timestamp"));
    ASSERT_TRUE(payload.contains("git_sha"));
    ASSERT_TRUE(payload.at("checks_passed").is_array());
}

nlohmann::json capture_failure_json(const std::function<void()>& fn) {
    testing::internal::CaptureStderr();
    EXPECT_THROW(fn(), std::runtime_error);
    const auto stderr_output = testing::internal::GetCapturedStderr();
    return nlohmann::json::parse(stderr_output);
}

}  // namespace

TEST(Validators, FinitePasses) {
    const double values[] = {1.0, 2.0, 3.0};
    EXPECT_NO_THROW(AGENTBIBLE_VALIDATE_FINITE(values, 3));
    const auto payload = agentbible::make_provenance_record({{"finite_array", true, 0.0, 0.0, "n/a", ""}});
    expect_record_shape(payload);
}

TEST(Validators, FiniteFails) {
    const double values[] = {1.0, std::numeric_limits<double>::quiet_NaN()};
    const auto payload = capture_failure_json([&]() { AGENTBIBLE_VALIDATE_FINITE(values, 2); });
    expect_record_shape(payload);
    EXPECT_EQ(payload["checks_passed"][0]["check_name"], "finite_array");
}

TEST(Validators, PositiveChecks) {
    const double good[] = {1.0, 2.0};
    const double bad[] = {1.0, 0.0};
    EXPECT_NO_THROW(AGENTBIBLE_VALIDATE_POSITIVE(good, 2));
    const auto payload = capture_failure_json([&]() { AGENTBIBLE_VALIDATE_POSITIVE(bad, 2); });
    EXPECT_EQ(payload["checks_passed"][0]["check_name"], "positive_array");
}

TEST(Validators, NonNegativeChecks) {
    const double good[] = {0.0, 2.0};
    const double bad[] = {0.0, -1.0};
    EXPECT_NO_THROW(AGENTBIBLE_VALIDATE_NON_NEGATIVE(good, 2));
    const auto payload = capture_failure_json([&]() { AGENTBIBLE_VALIDATE_NON_NEGATIVE(bad, 2); });
    EXPECT_EQ(payload["checks_passed"][0]["check_name"], "non_negative_array");
}

TEST(Validators, ProbabilityChecks) {
    const double good[] = {0.2, 0.3, 0.5};
    const double bad[] = {0.2, 1.3};
    EXPECT_NO_THROW(AGENTBIBLE_VALIDATE_PROBABILITY(good, 3));
    const auto payload = capture_failure_json([&]() { AGENTBIBLE_VALIDATE_PROBABILITY(bad, 2); });
    EXPECT_EQ(payload["checks_passed"][0]["check_name"], "probability_array");
}

TEST(Validators, NormalizedL1Checks) {
    const double good[] = {0.25, 0.25, 0.5};
    const double bad[] = {0.3, 0.3, 0.3};
    EXPECT_NO_THROW(AGENTBIBLE_VALIDATE_NORMALIZED_L1(good, 3, 1e-10));
    const auto payload = capture_failure_json([&]() { AGENTBIBLE_VALIDATE_NORMALIZED_L1(bad, 3, 1e-10); });
    EXPECT_EQ(payload["checks_passed"][0]["check_name"], "normalized_l1");
}

TEST(Validators, SymmetricChecks) {
    const double good[] = {1.0, 2.0, 2.0, 3.0};
    const double bad[] = {1.0, 0.0, 1.0, 3.0};
    EXPECT_NO_THROW(AGENTBIBLE_VALIDATE_SYMMETRIC(good, 2, 1e-12));
    const auto payload = capture_failure_json([&]() { AGENTBIBLE_VALIDATE_SYMMETRIC(bad, 2, 1e-12); });
    EXPECT_EQ(payload["checks_passed"][0]["check_name"], "symmetric");
}

TEST(Validators, UnitaryChecks) {
    const std::complex<double> hadamard[] = {
        {0.7071067811865475, 0.0},
        {0.7071067811865475, 0.0},
        {0.7071067811865475, 0.0},
        {-0.7071067811865475, 0.0},
    };
    const std::complex<double> bad[] = {
        {1.0, 0.0},
        {1.0, 0.0},
        {0.0, 0.0},
        {1.0, 0.0},
    };
    EXPECT_NO_THROW(AGENTBIBLE_VALIDATE_UNITARY(hadamard, 2, 1e-10, 1e-12));
    const auto payload = capture_failure_json([&]() { AGENTBIBLE_VALIDATE_UNITARY(bad, 2, 1e-10, 1e-12); });
    EXPECT_EQ(payload["checks_passed"][0]["check_name"], "unitary");
}

TEST(Validators, PositiveDefiniteChecks) {
    const double good[] = {2.0, 1.0, 1.0, 2.0};
    const double bad[] = {0.0, 1.0, 1.0, 0.0};
    EXPECT_NO_THROW(AGENTBIBLE_VALIDATE_POSITIVE_DEFINITE(good, 2));
    const auto payload = capture_failure_json([&]() { AGENTBIBLE_VALIDATE_POSITIVE_DEFINITE(bad, 2); });
    EXPECT_EQ(payload["checks_passed"][0]["check_name"], "positive_definite");
}
