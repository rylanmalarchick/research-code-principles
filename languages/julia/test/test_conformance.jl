using AgentBible
using JSON3
using Test

@testset "Conformance" begin
    hadamard = ComplexF64[1.0 1.0; 1.0 -1.0] / sqrt(2.0)
    @test AgentBible.check_unitary(hadamard) == hadamard

    perturbed = copy(hadamard)
    theta = sqrt(2.0) * 1e-9
    perturbed[:, 2] .*= exp(im * theta)
    @test AgentBible.check_unitary(perturbed) == perturbed

    failing = copy(hadamard)
    failing[1, 1] += 1e-5
    @test_throws AgentBibleError AgentBible.check_unitary(failing)

    @test_throws AgentBibleError AgentBible.check_finite(NaN)

    record = ProvenanceRecord([
        CheckResult("unitary", true, 1e-10, 1e-12, "frobenius", nothing),
    ])
    io = IOBuffer()
    emit_provenance(record, io)
    payload = JSON3.read(String(take!(io)))

    @test payload["spec_version"] == "1.0"
    @test payload["language"] == "julia"
    @test payload["checks_passed"][1]["check_name"] == "unitary"
end
