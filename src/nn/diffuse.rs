use std::rc::Rc;

use crate::{
    error::MathError,
    math::{cell::OpenSet, sheaf::CellularSheaf, tensors::Matrix},
    nn::activate::Activations,
};

pub struct DiffusionLayer<O: OpenSet> {
    sheaf: Rc<CellularSheaf<O>>,
    weights: Matrix,
    activation: Activations,
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
        })
    }

    pub fn diffuse(&self, k: usize, k_features: Matrix, down_included: bool) -> Result<Matrix, MathError> {
        let diff = self
            .sheaf
            .k_hodge_laplacian(k, k_features, down_included)?;
        let weighted = self
            .weights
            .matmul(&diff)
            .map_err(MathError::Candle)?;
        self.activation.activate(weighted).map_err(MathError::Candle)
    }
}
