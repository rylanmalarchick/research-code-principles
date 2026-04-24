use ndarray::Array2;
use num_complex::Complex64;

use crate::AgentBibleError;

#[derive(Debug, Clone)]
pub struct UnitaryMatrix(Array2<Complex64>);

impl UnitaryMatrix {
    pub fn new(m: Array2<Complex64>) -> Result<Self, AgentBibleError> {
        Self::new_with_tol(m, 1e-10, 1e-12)
    }

    pub fn new_with_tol(
        m: Array2<Complex64>,
        rtol: f64,
        atol: f64,
    ) -> Result<Self, AgentBibleError> {
        if m.nrows() != m.ncols() {
            return Err(AgentBibleError::NotUnitary);
        }
        if !m
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
        {
            return Err(AgentBibleError::NonFinite);
        }
        let residual =
            m.t().mapv(|value| value.conj()).dot(&m) - Array2::<Complex64>::eye(m.nrows());
        let frobenius = residual
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if frobenius <= atol + (rtol * m.nrows() as f64) {
            Ok(Self(m))
        } else {
            Err(AgentBibleError::NotUnitary)
        }
    }

    pub fn value(&self) -> &Array2<Complex64> {
        &self.0
    }
}
