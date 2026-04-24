using Dates
using JSON3
using LibGit2

const SPEC_VERSION = "1.0"

struct CheckResult
    check_name::String
    passed::Bool
    rtol::Float64
    atol::Float64
    norm_used::String
    error_message::Union{Nothing, String}
end

struct ProvenanceRecord
    spec_version::String
    language::String
    timestamp::String
    git_sha::String
    checks_passed::Vector{CheckResult}
    metadata::Dict{String, Any}
end

function _repo_root()
    return normpath(joinpath(@__DIR__, "..", "..", ".."))
end

function _git_sha()
    try
        repo = LibGit2.GitRepo(_repo_root())
        return string(LibGit2.head_oid(repo))
    catch
        return "unknown"
    end
end

function _timestamp()
    return Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ")
end

function _check_dict(check::CheckResult)
    return Dict(
        "check_name" => check.check_name,
        "passed" => check.passed,
        "rtol" => check.rtol,
        "atol" => check.atol,
        "norm_used" => check.norm_used,
        "error_message" => check.error_message,
    )
end

function _record_dict(record::ProvenanceRecord)
    payload = Dict(
        "spec_version" => record.spec_version,
        "language" => record.language,
        "timestamp" => record.timestamp,
        "git_sha" => record.git_sha,
        "checks_passed" => [_check_dict(check) for check in record.checks_passed],
    )
    if !isempty(record.metadata)
        payload["metadata"] = record.metadata
    end
    return payload
end

function ProvenanceRecord(checks_passed::Vector{CheckResult}; metadata::Dict{String, Any}=Dict{String, Any}())
    return ProvenanceRecord(
        SPEC_VERSION,
        "julia",
        _timestamp(),
        _git_sha(),
        checks_passed,
        metadata,
    )
end

function emit_provenance(record::ProvenanceRecord, path::String)
    open(path, "w") do io
        JSON3.write(io, _record_dict(record))
    end
end

function emit_provenance(record::ProvenanceRecord, io::IO)
    JSON3.write(io, _record_dict(record))
end
