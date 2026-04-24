#include <fstream>
#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "agentbible/agentbible.hpp"

namespace {

bool is_iso_timestamp(const std::string& value) {
    return value.size() >= 20 && value.back() == 'Z' && value.find('T') != std::string::npos;
}

void expect_schema_contract(const nlohmann::json& payload) {
    ASSERT_EQ(payload.at("spec_version"), "1.0");
    ASSERT_EQ(payload.at("language"), "cpp");
    ASSERT_TRUE(is_iso_timestamp(payload.at("timestamp").get<std::string>()));
    ASSERT_TRUE(payload.at("git_sha").is_string());
    ASSERT_TRUE(payload.at("checks_passed").is_array());
    for (const auto& check : payload.at("checks_passed")) {
        ASSERT_TRUE(check.at("check_name").is_string());
        ASSERT_TRUE(check.at("passed").is_boolean());
        ASSERT_TRUE(check.at("rtol").is_number());
        ASSERT_TRUE(check.at("atol").is_number());
    }
}

}  // namespace

TEST(Provenance, RoundTripMatchesSchema) {
    const auto payload = agentbible::make_provenance_record(
        {
            {"finite_array", true, 0.0, 0.0, "n/a", ""},
            {"unitary", true, 1e-10, 1e-12, "frobenius", ""},
        },
        {
            {"git_branch", "main"},
            {"git_dirty", false},
            {"hostname", "localhost"},
            {"platform", "linux"},
            {"cpu_model", "test-cpu"},
            {"memory_gb", 16.0},
            {"gpu_info", nullptr},
            {"slurm_job_id", nullptr},
            {"slurm_nodelist", nullptr},
            {"mpi_rank", nullptr},
            {"mpi_size", nullptr},
            {"random_seed_numpy", nullptr},
            {"random_seed_python", nullptr},
            {"packages", {{"nlohmann_json", "3.x"}}},
            {"pip_freeze", ""},
            {"quantum_backend", nullptr},
            {"quantum_shots", nullptr},
        });
    expect_schema_contract(payload);

    const auto path = std::string("test_provenance_cpp.json");
    std::ofstream output(path);
    output << payload.dump(2);
    output.close();

    std::ifstream input(path);
    nlohmann::json roundtrip;
    input >> roundtrip;
    expect_schema_contract(roundtrip);
}
