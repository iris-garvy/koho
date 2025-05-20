use candle_core::{Tensor, Var};
use error::MathError;
use math::{cell::OpenSet, tensors::Matrix};
use nn::{
    diffuse::DiffusionLayer,
    loss::{LossFn, LossKind},
    metrics::{EpochMetrics, TrainingMetrics},
    optimize::{OptimKind, Optimize, OptimizerParams},
};

pub mod error;
pub mod math;
pub mod nn;

pub trait Parameterized {
    /// return a Vec of every Var this module owns
    fn parameters(&self) -> Vec<Var>;
}

/// A sheaf neural network sitting on k-cells, with toggleable down-laplacian inclusion in the diffusion operator.
pub struct SheafNN<O: OpenSet> {
    layers: Vec<DiffusionLayer<O>>,
    loss_fn: LossFn,
    optimizer: Optimize,
    k: usize,
}

impl<O: OpenSet> SheafNN<O> {
    /// Spawn a new sheaf neural network of dimension `k` with the provided set of `DiffusionLayer<O>`, and loss type.
    pub fn sequential(k: usize, layers: Vec<DiffusionLayer<O>>, loss: LossKind) -> Self {
        let loss_fn = LossFn::new(loss);
        Self {
            layers,
            loss_fn,
            optimizer: Optimize::default(),
            k,
        }
    }
    /// Sets the optimizer for the sheaf neural network
    pub fn set_optimizer(
        &mut self,
        kind: OptimKind,
        lr: f64,
        extra_params: OptimizerParams,
    ) -> Result<(), MathError> {
        let vars = self.parameters();
        let mut opt = Optimize::new(kind, extra_params, lr);
        opt.new_weights(vars).map_err(MathError::Candle)?;
        self.optimizer = opt;
        Ok(())
    }

    pub fn refresh_optimizer(&mut self) -> Result<(), MathError> {
        let weights = self.parameters();
        self.optimizer
            .new_weights(weights)
            .map_err(MathError::Candle)
    }

    /// Runs a foward pass through all diffusion layers on a given `input: Matrix`, with toggle for down-laplacian inclusion
    pub fn forward(&mut self, input: Matrix, down_included: bool) -> Result<Matrix, MathError> {
        let mut output = input;
        for i in &self.layers {
            output = i.diffuse(self.k, output, down_included)?;
        }
        Ok(output)
    }
    /// Runs a backward pass using the provided optimizer for supervised training. Optimizer must be set!
    pub fn backward(&mut self, output: &Matrix, target: &Matrix) -> Result<(), MathError> {
        let output_tensor: &Tensor = output.inner();
        let target_tensor: &Tensor = target.inner();

        let loss_tensor = self
            .loss_fn
            .compute(output_tensor, target_tensor)
            .map_err(MathError::Candle)?;

        self.optimizer
            .backward_step(&loss_tensor)
            .map_err(MathError::Candle)?;
        Ok(())
    }

    /// Runs a full training loop over `epochs: usize` for the given training pairs
    pub fn train(
        &mut self,
        data: &[(Matrix, Matrix)],
        epochs: usize,
        down_included: bool,
    ) -> Result<TrainingMetrics, MathError> {
        let mut output = TrainingMetrics::new(epochs);
        for epoch in 1..=epochs {
            let mut total_loss = 0.0_f32;
            for (input, target) in data {
                let output = self.forward(input.clone(), down_included)?;
                let loss_tensor = self
                    .loss_fn
                    .compute(output.inner(), target.inner())
                    .map_err(MathError::Candle)?;

                let loss_val = loss_tensor.to_scalar::<f32>().unwrap_or(f32::NAN);
                total_loss += loss_val;
                self.backward(&output, target)?;

                let new_weights = self.optimizer.into_inner()?;

                for (i, var) in new_weights.into_iter().enumerate() {
                    self.layers[i].update_weights(var);
                }
                self.refresh_optimizer()?;
            }

            let avg_loss = total_loss / (data.len() as f32);
            output.push(EpochMetrics::new(epoch, avg_loss));
        }
        Ok(output)
    }
}

impl<O: OpenSet> Parameterized for SheafNN<O> {
    fn parameters(&self) -> Vec<Var> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
