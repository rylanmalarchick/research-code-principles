module AgentBible

include("errors.jl")
include("provenance.jl")
include("validators.jl")

export AgentBibleError
export CheckResult, ProvenanceRecord, SPEC_VERSION, emit_provenance
export DEFAULT_RTOL, DEFAULT_ATOL
export check_finite, check_positive, check_non_negative, check_probability
export check_normalized_l1, check_symmetric, check_unitary, check_positive_definite
export @validate_finite, @validate_positive, @validate_non_negative
export @validate_probability, @validate_normalized_l1, @validate_symmetric
export @validate_unitary, @validate_positive_definite

end
