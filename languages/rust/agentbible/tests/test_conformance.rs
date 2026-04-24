use std::panic::catch_unwind;

use ndarray::array;
use num_complex::Complex64;

use agentbible::provenance::{CheckResult, ProvenanceRecord};
use agentbible::types::UnitaryMatrix;
use agentbible::validate_finite;

fn hadamard() -> ndarray::Array2<Complex64> {
    array![
        [
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        ],
        [
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(-1.0 / 2_f64.sqrt(), 0.0),
        ]
    ]
}

#[test]
fn unitary_conformance_matches_spec_thresholds() {
    let base = hadamard();
    assert!(UnitaryMatrix::new(base.clone()).is_ok());

    let mut perturbed = base.clone();
    let theta = std::f64::consts::SQRT_2 * 1e-9;
    let phase = Complex64::new(0.0, theta).exp();
    perturbed[(0, 1)] *= phase;
    perturbed[(1, 1)] *= phase;
    assert!(UnitaryMatrix::new(perturbed).is_ok());

    let mut failing = base;
    failing[(0, 0)] += Complex64::new(1e-5, 0.0);
    assert!(UnitaryMatrix::new(failing).is_err());
}

#[test]
fn nan_injection_fails_finite() {
    assert!(catch_unwind(|| validate_finite!(f64::NAN)).is_err());
}

#[test]
fn provenance_json_roundtrip() {
    let record = ProvenanceRecord::new(vec![CheckResult::success(
        "unitary",
        1e-10,
        1e-12,
        "frobenius",
    )]);
    let encoded = record.to_json_string();
    let decoded: ProvenanceRecord =
        serde_json::from_str(&encoded).expect("provenance roundtrip should decode");
    assert_eq!(decoded.spec_version, "1.0");
    assert_eq!(decoded.language, "rust");
    assert_eq!(decoded.checks_passed[0].check_name, "unitary");
}
