pub mod check;
pub mod provenance;
pub mod types;

use thiserror::Error;

pub const SPEC_VERSION: &str = "1.0";

#[derive(Debug, Error, Clone)]
pub enum AgentBibleError {
    #[error("value must be finite")]
    NonFinite,
    #[error("value must be strictly positive")]
    NotPositive,
    #[error("value must be in [0, 1]")]
    InvalidProbability,
    #[error("L1 sum must be within tolerance of 1")]
    NotNormalized,
    #[error("matrix is not unitary")]
    NotUnitary,
    #[error("matrix is not positive definite")]
    NotPositiveDefinite,
}
