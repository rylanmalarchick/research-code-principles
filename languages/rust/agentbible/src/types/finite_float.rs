use crate::AgentBibleError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteFloat(f64);

impl FiniteFloat {
    pub fn new(v: f64) -> Result<Self, AgentBibleError> {
        if v.is_finite() {
            Ok(Self(v))
        } else {
            Err(AgentBibleError::NonFinite)
        }
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}
