use crate::AgentBibleError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositiveFloat(f64);

impl PositiveFloat {
    pub fn new(v: f64) -> Result<Self, AgentBibleError> {
        if !v.is_finite() {
            return Err(AgentBibleError::NonFinite);
        }
        if v > 0.0 {
            Ok(Self(v))
        } else {
            Err(AgentBibleError::NotPositive)
        }
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}
