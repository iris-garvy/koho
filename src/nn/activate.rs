use candle_core::{Error, Tensor};

use crate::math::tensors::Matrix;

pub enum Activations {
    Step,
    Linear,
    Sigmoid,
    Tanh,
    ReLU,
    Softmax,
    Swish,
    GeLU,
    Sinc,
    SeLU,
}

impl Activations {
    pub fn activate(&self, tensor: Tensor) -> Result<Matrix, Error> {
        let device = tensor.device();
        let dtype = tensor.dtype();

        let activated = match self {
            Activations::Step => {
                // Heaviside step function: 1.0 where x >= 0.0, else 0.0
                let zeros = Tensor::zeros_like(&tensor)?;
                let ones = Tensor::ones_like(&tensor)?;
                tensor.ge(&zeros)?.where_cond(&ones, &zeros)?
            }
            Activations::Linear => tensor.clone(),
            Activations::Tanh => tensor.tanh()?,
            Activations::ReLU => tensor.relu()?,
            Activations::Sinc => {
                // sinc(x) = sin(x) / x, define 1 at x=0
                // Using a small epsilon to handle division by zero.
                // If x is near zero, output 1, else sin(x)/x
                let eps_val = 1e-7f64;
                let eps = Tensor::full(eps_val, tensor.dims(), device)?.to_dtype(dtype)?;
                let near_zero = tensor.abs()?.le(&eps)?;

                let numerator = tensor.sin()?;
                let denominator = tensor.clone(); // Clone to avoid consuming tensor
                let value = numerator.div(&denominator)?;

                near_zero.where_cond(&Tensor::ones_like(&tensor)?, &value)?
            }
            Activations::Sigmoid => {
                // Sigmoid(x) = 1 / (1 + exp(-x))
                let neg_x = tensor.neg()?;
                let exp_neg_x = neg_x.exp()?;
                let one = Tensor::ones_like(&exp_neg_x)?;
                let one_plus_exp_neg_x = one.add(&exp_neg_x)?;
                one_plus_exp_neg_x.recip()? // 1 / (1 + exp(-x))
            }
            Activations::Softmax => {
                // Softmax(x_i) = exp(x_i) / sum(exp(x_j)) along the last dimension
                // For a Matrix (rank 2), apply along dim 1 (columns) for each row.
                let exp_x = tensor.exp()?;
                // Sum along the last dimension, keeping the dimension for broadcasting
                let sum_exp_x = exp_x.sum_keepdim(1)?;
                exp_x.broadcast_div(&sum_exp_x)?
            }
            Activations::Swish => {
                // Swish(x) = x * Sigmoid(x)
                let neg_x = tensor.neg()?;
                let exp_neg_x = neg_x.exp()?;
                let one = Tensor::ones_like(&exp_neg_x)?;
                let one_plus_exp_neg_x = one.add(&exp_neg_x)?;
                let sigmoid_x = one_plus_exp_neg_x.recip()?;
                tensor.mul(&sigmoid_x)?
            }
            Activations::GeLU => {
                // GeLU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
                let sqrt_two_val = 2.0f64.sqrt();
                let sqrt_two =
                    Tensor::full(sqrt_two_val, tensor.dims(), device)?.to_dtype(dtype)?;

                let x_div_sqrt_two = tensor.div(&sqrt_two)?;
                let erf_val = x_div_sqrt_two.erf()?;
                let one = Tensor::ones_like(&erf_val)?;
                let one_plus_erf = one.add(&erf_val)?;

                let half_val = 0.5f64;
                let half = Tensor::full(half_val, tensor.dims(), device)?.to_dtype(dtype)?;

                tensor.mul(&half)?.mul(&one_plus_erf)?
            }
            Activations::SeLU => {
                // SeLU(x) = lambda * (x if x > 0 else alpha * (exp(x) - 1))
                // Standard constants for SeLU
                let alpha_val = 1.673_263_242_354_377_2_f64;
                let lambda_val = 1.050_700_987_355_480_5_f64;

                let alpha = Tensor::full(alpha_val, tensor.dims(), device)?.to_dtype(dtype)?;
                let lambda = Tensor::full(lambda_val, tensor.dims(), device)?.to_dtype(dtype)?;
                let zero = Tensor::zeros_like(&tensor)?;

                // Condition: x > 0
                let cond_gt_zero = tensor.gt(&zero)?;

                // Case for x > 0: just x
                let case_gt_zero = tensor.clone();

                // Case for x <= 0: alpha * (exp(x) - 1)
                let exp_x = tensor.exp()?;
                let one_for_sub = Tensor::ones_like(&exp_x)?;
                let exp_x_minus_one = exp_x.sub(&one_for_sub)?;
                let case_le_zero = alpha.mul(&exp_x_minus_one)?;

                let result = cond_gt_zero.where_cond(&case_gt_zero, &case_le_zero)?;
                lambda.mul(&result)?
            }
        };

        Matrix::new(activated, device.clone(), dtype)
    }
}
