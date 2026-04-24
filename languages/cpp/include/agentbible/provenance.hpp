#pragma once

#include <chrono>
#include <ctime>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef AGENTBIBLE_GIT_SHA
#define AGENTBIBLE_GIT_SHA "unknown"
#endif

namespace agentbible {

inline constexpr const char* SPEC_VERSION = "1.0";

struct CheckResult {
    std::string check_name;
    bool passed{false};
    double rtol{0.0};
    double atol{0.0};
    std::string norm_used{"n/a"};
    std::string error_message{};
};

inline std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto as_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &as_time_t);
#else
    gmtime_r(&as_time_t, &tm);
#endif
    std::ostringstream stream;
    stream << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

inline nlohmann::json to_json(const CheckResult& result) {
    return {
        {"check_name", result.check_name},
        {"passed", result.passed},
        {"rtol", result.rtol},
        {"atol", result.atol},
        {"norm_used", result.norm_used},
        {"error_message", result.error_message.empty() ? nullptr : nlohmann::json(result.error_message)},
    };
}

inline nlohmann::json make_provenance_record(
    const std::vector<CheckResult>& checks,
    const nlohmann::json& metadata = nlohmann::json::object()) {
    nlohmann::json payload = {
        {"spec_version", SPEC_VERSION},
        {"language", "cpp"},
        {"timestamp", utc_timestamp()},
        {"git_sha", AGENTBIBLE_GIT_SHA},
        {"checks_passed", nlohmann::json::array()},
    };
    for (const auto& check : checks) {
        payload["checks_passed"].push_back(to_json(check));
    }
    if (!metadata.empty()) {
        payload["metadata"] = metadata;
    }
    return payload;
}

inline std::string serialize_record(
    const std::vector<CheckResult>& checks,
    const nlohmann::json& metadata = nlohmann::json::object()) {
    return make_provenance_record(checks, metadata).dump();
}

inline void emit_provenance(
    std::ostream& stream,
    const std::vector<CheckResult>& checks,
    const nlohmann::json& metadata = nlohmann::json::object()) {
    stream << serialize_record(checks, metadata);
}

}  // namespace agentbible
