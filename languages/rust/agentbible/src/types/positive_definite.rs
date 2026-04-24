use ndarray::Array2;

use crate::AgentBibleError;

#[derive(Debug, Clone)]
pub struct PositiveDefiniteMatrix(Array2<f64>);

impl PositiveDefiniteMatrix {
    pub fn new(m: Array2<f64>) -> Result<Self, AgentBibleError> {
        if m.nrows() != m.ncols() {
            return Err(AgentBibleError::NotPositiveDefinite);
        }
        if !m.iter().all(|value| value.is_finite()) {
            return Err(AgentBibleError::NonFinite);
        }
        let n = m.nrows();
        let mut lower = vec![0.0_f64; n * n];
        for row in 0..n {
            for col in 0..=row {
                let mut sum = m[(row, col)];
                for k in 0..col {
                    sum -= lower[row * n + k] * lower[col * n + k];
                }
                if row == col {
                    if sum <= 0.0 {
                        return Err(AgentBibleError::NotPositiveDefinite);
                    }
                    lower[row * n + col] = sum.sqrt();
                } else {
                    lower[row * n + col] = sum / lower[col * n + col];
                }
            }
        }
        Ok(Self(m))
    }

    pub fn value(&self) -> &Array2<f64> {
        &self.0
    }
}
