use candle_core::{Error, Tensor, Var};
use candle_optimisers::{Decay, Momentum};
use error::MathError;
use math::{cell::OpenSet, tensors::Matrix};
use nn::{diffuse::DiffusionLayer, loss::{LossFn, LossKind}, optimize::{OptimKind, Optimize}};

pub mod error;
pub mod math;
pub mod nn;

pub enum OptimizerParams {
    SGD {
        dampening: Option<f64>,
        weight_decay: Option<Decay>,
        momentum: Option<Momentum>
    },
    AdamW {
        beta_1: f64,
        beta_2: f64,
        eps: f64, 
        weight_decay: Option<Decay>,
        amsgrad: bool, 
    },
    Else

}

pub struct SheafNN<O: OpenSet> {
    layers: Vec<DiffusionLayer<O>>,
    loss_fn: LossFn,
    optimizer: Option<Optimize>,
    k: usize,
}

impl<O: OpenSet> SheafNN<O> {
    pub fn sequential(k: usize, layers: Vec<DiffusionLayer<O>>, loss: LossKind) -> Self {
        let loss_fn = LossFn::new(loss);
        Self {
            layers,
            loss_fn,
            optimizer: None,
            k
        }
    }

    pub fn set_optimizer(&mut self, kind: OptimKind, vars: Vec<Var>, lr: f64, extra_params: OptimizerParams) -> Result<(), MathError> {
        self.optimizer = Some(Optimize::new(kind, vars, lr, extra_params).map_err(MathError::Candle)?);
        Ok(())
    }

    pub fn forward(&mut self, input: Matrix, down_included: bool) -> Result<Matrix, MathError> {
        let mut output = input;
        for i in &self.layers {
            output = i.diffuse(self.k, output, down_included)?;
        }
        Ok(output)
    }

    pub fn backward(&mut self, output: &Matrix, target: &Matrix) -> Result<(), MathError> {
        if self.optimizer.is_none() {
            let msg = "No optimizer set – call set_optimizer before training";
            return Err(MathError::Candle(Error::Msg(msg.to_owned())));
        }

        let output_tensor: &Tensor = output.inner();  
        let target_tensor: &Tensor = target.inner();

        let loss_tensor = self.loss_fn
            .compute(output_tensor, target_tensor)
            .map_err(MathError::Candle)?;

        self.optimizer
            .as_mut()
            .unwrap()
            .backward_step(&loss_tensor)
            .map_err(MathError::Candle)?;
        Ok(())
    }

    pub fn train(&mut self, data: &[(Matrix, Matrix)], epochs: usize, down_included: bool) -> Result<(), MathError> {
        for epoch in 1..=epochs {
            let mut total_loss = 0.0_f32;
            for (input, target) in data {
                let output = self.forward(input.clone(), down_included)?; 
                let loss_tensor = self.loss_fn
                    .compute(output.inner(), target.inner())
                    .map_err(MathError::Candle)?;

                let loss_val = loss_tensor.to_scalar::<f32>().unwrap_or(f32::NAN);
                total_loss += loss_val;
                self.backward(&output, target)?; 
            }

            let avg_loss = total_loss / (data.len() as f32);
            println!("Epoch {}/{} - Average Loss: {:.6}", epoch, epochs, avg_loss);
        }
        Ok(())
    }
}