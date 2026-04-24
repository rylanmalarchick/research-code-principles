using LinearAlgebra

const DEFAULT_RTOL = 1e-10
const DEFAULT_ATOL = 1e-12

function _fail!(check_name::String, rtol::Float64, atol::Float64, norm_used::String, message::String)
    record = ProvenanceRecord([CheckResult(check_name, false, rtol, atol, norm_used, message)])
    emit_provenance(record, stderr)
    println(stderr)
    throw(AgentBibleError(message))
end

function check_finite(x; name::String="value")
    if all(isfinite, x isa Number ? (x,) : x)
        return x
    end
    _fail!("finite", 0.0, 0.0, "n/a", "$name is not finite")
end

function check_positive(x; name::String="value")
    check_finite(x; name=name)
    values = x isa Number ? (x,) : x
    all(>(0.0), values) || _fail!("positive", 0.0, 0.0, "n/a", "$name is not strictly positive")
    return x
end

function check_non_negative(x; name::String="value")
    check_finite(x; name=name)
    values = x isa Number ? (x,) : x
    all(>=(0.0), values) || _fail!("non_negative", 0.0, 0.0, "n/a", "$name contains a negative value")
    return x
end

function check_probability(x; name::String="value")
    check_finite(x; name=name)
    values = x isa Number ? (x,) : x
    all(v -> 0.0 <= v <= 1.0, values) || _fail!("probability", 0.0, 0.0, "n/a", "$name is outside [0, 1]")
    return x
end

function check_normalized_l1(x, atol::Float64=1e-10; name::String="value")
    check_finite(x; name=name)
    abs(sum(x) - 1.0) <= atol || _fail!("normalized_l1", 0.0, atol, "l1", "$name is not L1-normalized")
    return x
end

function check_symmetric(A, atol::Float64=DEFAULT_ATOL; name::String="matrix")
    check_finite(A; name=name)
    maximum(abs.(A .- transpose(A))) <= atol || _fail!("symmetric", 0.0, atol, "max_elementwise", "$name is not symmetric")
    return A
end

function check_unitary(U; rtol::Float64=DEFAULT_RTOL, atol::Float64=DEFAULT_ATOL, name::String="matrix")
    check_finite(U; name=name)
    residual = adjoint(U) * U - I
    norm(residual) <= atol + rtol * size(U, 1) || _fail!("unitary", rtol, atol, "frobenius", "$name is not unitary")
    return U
end

function check_positive_definite(A; name::String="matrix")
    check_finite(A; name=name)
    try
        cholesky(Hermitian(Matrix(A)))
    catch
        _fail!("positive_definite", 0.0, 0.0, "n/a", "$name is not positive definite")
    end
    return A
end

macro validate_finite(x)
    return :(check_finite($(esc(x)); name=$(string(x))))
end

macro validate_positive(x)
    return :(check_positive($(esc(x)); name=$(string(x))))
end

macro validate_non_negative(x)
    return :(check_non_negative($(esc(x)); name=$(string(x))))
end

macro validate_probability(x)
    return :(check_probability($(esc(x)); name=$(string(x))))
end

macro validate_normalized_l1(x, atol)
    return :(check_normalized_l1($(esc(x)), $(esc(atol)); name=$(string(x))))
end

macro validate_symmetric(A, atol)
    return :(check_symmetric($(esc(A)), $(esc(atol)); name=$(string(A))))
end

macro validate_unitary(U, rtol, atol)
    return :(check_unitary($(esc(U)); rtol=$(esc(rtol)), atol=$(esc(atol)), name=$(string(U))))
end

macro validate_positive_definite(A)
    return :(check_positive_definite($(esc(A)); name=$(string(A))))
end
