use std::rc::Rc;

use crate::{
    error::MathError,
    math::{cell::OpenSet, sheaf::CellularSheaf, tensors::Matrix},
    nn::activations::Activations,
};

pub struct DiffusionLayer<O: OpenSet> {
    sheaf: Rc<CellularSheaf<O>>,
    weights: Matrix,
    activation: Activations,
    k: usize,
}

impl<O: OpenSet> DiffusionLayer<O> {
    pub fn new(
        k: usize,
        activation: Activations,
        sheaf: Rc<CellularSheaf<O>>,
    ) -> Result<Self, MathError> {
        let device = sheaf.device.clone();
        let dtype = sheaf.dtype;
        let dim = sheaf.section_spaces[k][0].0.dimension();
        let weights = Matrix::rand(dim, dim, device, dtype).map_err(MathError::Candle)?;
        Ok(Self {
            sheaf,
            weights,
            activation,
            k,
        })
    }

    pub fn diffuse(&self, k_features: Matrix, down_included: bool) -> Result<Matrix, MathError> {
        let weighted_feats = self
            .weights
            .matmul(&k_features)
            .map_err(MathError::Candle)?;
        let out = self
            .sheaf
            .k_hodge_laplacian(self.k, weighted_feats, down_included)?;
        self.activation.activate(out).map_err(MathError::Candle)
    }
}
