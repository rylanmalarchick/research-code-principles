#[macro_use]
pub mod macros;

use ndarray::Array2;
use num_complex::Complex64;

use crate::provenance::{emit_failure_and_panic, CheckResult, ProvenanceRecord};
use crate::types::{
    FiniteFloat, PositiveDefiniteMatrix, PositiveFloat, Probability, UnitaryMatrix,
};
use crate::AgentBibleError;

pub fn check_finite(value: f64) -> Result<(), AgentBibleError> {
    FiniteFloat::new(value).map(|_| ())
}

pub fn check_finite_array(values: &[f64]) -> Result<(), AgentBibleError> {
    values.iter().try_for_each(|value| check_finite(*value))
}

pub fn check_positive(value: f64) -> Result<(), AgentBibleError> {
    PositiveFloat::new(value).map(|_| ())
}

pub fn check_probability(value: f64) -> Result<(), AgentBibleError> {
    Probability::new(value).map(|_| ())
}

pub fn check_normalized_l1(values: &[f64], atol: f64) -> Result<(), AgentBibleError> {
    check_finite_array(values)?;
    let total = values.iter().sum::<f64>();
    if (total - 1.0).abs() <= atol {
        Ok(())
    } else {
        Err(AgentBibleError::NotNormalized)
    }
}

pub fn check_unitary(matrix: &Array2<Complex64>) -> Result<(), AgentBibleError> {
    UnitaryMatrix::new(matrix.clone()).map(|_| ())
}

pub fn check_unitary_tol(
    matrix: &Array2<Complex64>,
    rtol: f64,
    atol: f64,
) -> Result<(), AgentBibleError> {
    UnitaryMatrix::new_with_tol(matrix.clone(), rtol, atol).map(|_| ())
}

pub fn check_positive_definite(matrix: &Array2<f64>) -> Result<(), AgentBibleError> {
    PositiveDefiniteMatrix::new(matrix.clone()).map(|_| ())
}

pub fn failure_record(
    check_name: &str,
    rtol: f64,
    atol: f64,
    norm_used: &str,
    message: &str,
) -> ProvenanceRecord {
    ProvenanceRecord::new(vec![CheckResult::failure(
        check_name, rtol, atol, norm_used, message,
    )])
}

pub fn fail_with_provenance(
    check_name: &str,
    rtol: f64,
    atol: f64,
    norm_used: &str,
    message: &str,
) -> ! {
    emit_failure_and_panic(check_name, rtol, atol, norm_used, message)
}
