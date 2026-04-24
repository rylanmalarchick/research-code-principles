using AgentBible
using JSON3
using Test

@testset "Provenance" begin
    record = ProvenanceRecord([
        CheckResult("finite_array", true, 0.0, 0.0, "n/a", nothing),
        CheckResult("unitary", true, 1e-10, 1e-12, "frobenius", nothing),
    ])

    io = IOBuffer()
    emit_provenance(record, io)
    payload = JSON3.read(String(take!(io)))

    @test payload["spec_version"] == "1.0"
    @test payload["language"] == "julia"
    @test haskey(payload, "timestamp")
    @test haskey(payload, "git_sha")
    @test length(payload["checks_passed"]) == 2
end
