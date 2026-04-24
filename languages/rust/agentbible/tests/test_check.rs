use std::panic::catch_unwind;

use ndarray::array;
use num_complex::Complex64;

use agentbible::{
    validate_finite, validate_finite_array, validate_normalized_l1, validate_positive,
    validate_positive_definite, validate_probability, validate_unitary, validate_unitary_tol,
};

#[test]
fn runtime_macros_accept_valid_inputs() {
    validate_finite!(1.0_f64);
    validate_finite_array!(&[1.0_f64, 2.0_f64]);
    validate_positive!(2.0_f64);
    validate_probability!(0.5_f64);
    validate_normalized_l1!(&[0.25_f64, 0.75_f64], 1e-10_f64);
    let unitary = array![
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
    ];
    validate_unitary!(&unitary);
    validate_unitary_tol!(&unitary, 1e-10_f64, 1e-12_f64);
    let positive_definite = array![[2.0_f64, 1.0_f64], [1.0_f64, 2.0_f64]];
    validate_positive_definite!(&positive_definite);
}

#[test]
fn runtime_macros_panic_on_invalid_inputs() {
    assert!(catch_unwind(|| validate_finite!(f64::NAN)).is_err());
    assert!(catch_unwind(|| validate_probability!(2.0_f64)).is_err());
    let bad = array![[0.0_f64, 1.0_f64], [1.0_f64, 0.0_f64]];
    assert!(catch_unwind(|| validate_positive_definite!(&bad)).is_err());
}
