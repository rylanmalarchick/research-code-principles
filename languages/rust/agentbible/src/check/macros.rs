#[macro_export]
macro_rules! validate_finite {
    ($x:expr) => {
        if let Err(error) = $crate::check::check_finite($x) {
            $crate::check::fail_with_provenance("finite", 0.0, 0.0, "n/a", &error.to_string());
        }
    };
}

#[macro_export]
macro_rules! validate_finite_array {
    ($arr:expr) => {
        if let Err(error) = $crate::check::check_finite_array($arr) {
            $crate::check::fail_with_provenance(
                "finite_array",
                0.0,
                0.0,
                "n/a",
                &error.to_string(),
            );
        }
    };
}

#[macro_export]
macro_rules! validate_positive {
    ($x:expr) => {
        if let Err(error) = $crate::check::check_positive($x) {
            $crate::check::fail_with_provenance("positive", 0.0, 0.0, "n/a", &error.to_string());
        }
    };
}

#[macro_export]
macro_rules! validate_probability {
    ($x:expr) => {
        if let Err(error) = $crate::check::check_probability($x) {
            $crate::check::fail_with_provenance("probability", 0.0, 0.0, "n/a", &error.to_string());
        }
    };
}

#[macro_export]
macro_rules! validate_normalized_l1 {
    ($arr:expr, $atol:expr) => {
        if let Err(error) = $crate::check::check_normalized_l1($arr, $atol) {
            $crate::check::fail_with_provenance(
                "normalized_l1",
                0.0,
                $atol,
                "l1",
                &error.to_string(),
            );
        }
    };
}

#[macro_export]
macro_rules! validate_unitary {
    ($matrix:expr) => {
        if let Err(error) = $crate::check::check_unitary($matrix) {
            $crate::check::fail_with_provenance(
                "unitary",
                1e-10,
                1e-12,
                "frobenius",
                &error.to_string(),
            );
        }
    };
}

#[macro_export]
macro_rules! validate_unitary_tol {
    ($matrix:expr, $rtol:expr, $atol:expr) => {
        if let Err(error) = $crate::check::check_unitary_tol($matrix, $rtol, $atol) {
            $crate::check::fail_with_provenance(
                "unitary",
                $rtol,
                $atol,
                "frobenius",
                &error.to_string(),
            );
        }
    };
}

#[macro_export]
macro_rules! validate_positive_definite {
    ($matrix:expr) => {
        if let Err(error) = $crate::check::check_positive_definite($matrix) {
            $crate::check::fail_with_provenance(
                "positive_definite",
                0.0,
                0.0,
                "n/a",
                &error.to_string(),
            );
        }
    };
}
