use crate::AgentBibleError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probability(f64);

impl Probability {
    pub fn new(v: f64) -> Result<Self, AgentBibleError> {
        if !v.is_finite() {
            return Err(AgentBibleError::NonFinite);
        }
        if (0.0..=1.0).contains(&v) {
            Ok(Self(v))
        } else {
            Err(AgentBibleError::InvalidProbability)
        }
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}
