use candle_core::{Result, Tensor};
use candle_nn::loss::{binary_cross_entropy_with_logit, cross_entropy, mse, nll};

/// Pick your poison.
#[derive(Debug, Clone)]
pub enum LossKind {
    NLL,           // inp is log-probs, target is class indices
    CrossEntropy,  // inp is raw logits, target is class indices
    MSE,           // inp and target are same shape
    BCEWithLogits, // inp is raw logits, target is same shape
}

/// A callable loss function.
#[derive(Debug, Clone)]
pub struct LossFn {
    kind: LossKind,
}

impl LossFn {
    /// Choose which loss to use.
    pub fn new(kind: LossKind) -> Self {
        Self { kind }
    }

    /// Compute the loss scalar for a batch.
    pub fn compute(&self, inp: &Tensor, target: &Tensor) -> Result<Tensor> {
        match self.kind {
            LossKind::NLL => nll(inp, target),
            LossKind::CrossEntropy => cross_entropy(inp, target),
            LossKind::MSE => mse(inp, target),
            LossKind::BCEWithLogits => binary_cross_entropy_with_logit(inp, target),
        }
    }
}
