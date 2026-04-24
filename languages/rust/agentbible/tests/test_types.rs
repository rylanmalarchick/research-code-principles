use ndarray::array;
use num_complex::Complex64;

use agentbible::types::{
    FiniteFloat, PositiveDefiniteMatrix, PositiveFloat, Probability, UnitaryMatrix,
};

#[test]
fn probability_newtype_validates_bounds() {
    let probability = Probability::new(0.5).expect("0.5 should be valid");
    assert_eq!(probability.value(), 0.5);
    assert!(Probability::new(1.5).is_err());
}

#[test]
fn finite_and_positive_newtypes_enforce_invariants() {
    assert!(FiniteFloat::new(f64::NAN).is_err());
    assert!(PositiveFloat::new(0.0).is_err());
    assert_eq!(PositiveFloat::new(2.0).expect("positive").value(), 2.0);
}

#[test]
fn unitary_matrix_uses_spec_tolerances() {
    let hadamard = array![
        [
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0)
        ],
        [
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(-1.0 / 2_f64.sqrt(), 0.0)
        ]
    ];
    assert!(UnitaryMatrix::new(hadamard.clone()).is_ok());

    let mut perturbed = hadamard.clone();
    let theta = std::f64::consts::SQRT_2 * 1e-9;
    let phase = Complex64::new(0.0, theta).exp();
    perturbed[(0, 1)] *= phase;
    perturbed[(1, 1)] *= phase;
    assert!(UnitaryMatrix::new(perturbed).is_ok());

    let mut failing = hadamard;
    failing[(0, 0)] += Complex64::new(1e-5, 0.0);
    assert!(UnitaryMatrix::new(failing).is_err());
}

#[test]
fn positive_definite_uses_cholesky() {
    let good = array![[2.0, 1.0], [1.0, 2.0]];
    let bad = array![[0.0, 1.0], [1.0, 0.0]];
    assert!(PositiveDefiniteMatrix::new(good).is_ok());
    assert!(PositiveDefiniteMatrix::new(bad).is_err());
}
