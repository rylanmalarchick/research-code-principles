using AgentBible
using LinearAlgebra
using Test

@testset "Validators" begin
    @test AgentBible.check_finite([1.0, 2.0]) == [1.0, 2.0]
    @test AgentBible.check_probability(0.5) == 0.5
    @test_throws AgentBibleError AgentBible.check_positive(0.0)
    @test AgentBible.check_normalized_l1([0.25, 0.75]) == [0.25, 0.75]

    hadamard = ComplexF64[1.0 1.0; 1.0 -1.0] / sqrt(2.0)
    @test AgentBible.check_unitary(hadamard) == hadamard

    perturbed = copy(hadamard)
    theta = sqrt(2.0) * 1e-9
    perturbed[:, 2] .*= exp(im * theta)
    @test AgentBible.check_unitary(perturbed) == perturbed

    failing = copy(hadamard)
    failing[1, 1] += 1e-5
    @test_throws AgentBibleError AgentBible.check_unitary(failing)

    positive_definite = [2.0 1.0; 1.0 2.0]
    @test AgentBible.check_positive_definite(positive_definite) == positive_definite
    @test_throws AgentBibleError AgentBible.check_positive_definite([0.0 1.0; 1.0 0.0])
end
