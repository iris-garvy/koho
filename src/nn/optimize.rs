use candle_core::{Error, Result, Tensor, Var};
use candle_core::backprop::GradStore;
use candle_optimisers::{
    adadelta::{Adadelta, ParamsAdaDelta},
    adagrad::{Adagrad, ParamsAdaGrad},
    adam::{Adam, ParamsAdam},
    adamax::{Adamax, ParamsAdaMax},
    esgd::{SGD, ParamsSGD},
    nadam::{NAdam, ParamsNAdam},
    radam::{RAdam, ParamsRAdam},
    rmsprop::{RMSprop, ParamsRMSprop},
};
use candle_nn::Optimizer;

use crate::OptimizerParams;

/// Enum to represent which optimizer type to use
#[derive(Debug, Clone)]
pub enum OptimKind {
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    AdamW,    // will use Adam with decoupled weight decay
    Adamax,
    NAdam,
    RAdam,
    RMSprop,
}

/// Enum to hold the concrete optimizer instance
pub enum Optim {
    SGD(SGD),
    Adadelta(Adadelta),
    Adagrad(Adagrad),
    Adam(Adam),
    Adamax(Adamax),
    NAdam(NAdam),
    RAdam(RAdam),
    RMSprop(RMSprop),
}

/// Wrapper struct providing a single interface
pub struct Optimize {
    optim: Optim,
}

impl Optimize {
    /// Factory method to create the appropriate optimizer.
    pub fn new(kind: OptimKind, vars: Vec<Var>, lr: f64, extra_params: OptimizerParams) -> Result<Optimize> {
        let optim = match kind {
            OptimKind::SGD => {
                if let OptimizerParams::SGD { dampening, weight_decay, momentum } = extra_params {
                    let dampening = if dampening.is_some() { dampening.unwrap() } else { 0.0 };
                    let cfg = ParamsSGD { lr, weight_decay, momentum, dampening };
                    Optim::SGD(SGD::new(vars, cfg)?)
                } else {
                    return Err(Error::Msg("Parameters provided were incorrect given the OptimKind".to_owned()));
                }
            }
            OptimKind::Adadelta => {
                let cfg = ParamsAdaDelta { lr, ..Default::default() };
                Optim::Adadelta(Adadelta::new(vars, cfg)?)
            }
            OptimKind::Adagrad => {
                let cfg = ParamsAdaGrad { lr, ..Default::default() };
                Optim::Adagrad(Adagrad::new(vars, cfg)?)
            }
            OptimKind::Adam => {
                let cfg = ParamsAdam { lr, ..Default::default() };
                Optim::Adam(Adam::new(vars, cfg)?)
            }
            OptimKind::AdamW => {
                if let OptimizerParams::AdamW { beta_1, beta_2, eps, weight_decay, amsgrad } = extra_params {
                    let cfg = ParamsAdam { 
                        lr,
                        beta_1,
                        beta_2,
                        eps, 
                        weight_decay,
                        amsgrad, 
                    };
                    Optim::Adam(Adam::new(vars, cfg)?)  // still stored as Optim::Adam
                } else {
                    return Err(Error::Msg("Parameters provided were incorrect given the OptimKind".to_owned()));
                }
            }
            OptimKind::Adamax => {
                let cfg = ParamsAdaMax { lr, ..Default::default() };
                Optim::Adamax(Adamax::new(vars, cfg)?)
            }
            OptimKind::NAdam => {
                let cfg = ParamsNAdam { lr, ..Default::default() };
                Optim::NAdam(NAdam::new(vars, cfg)?)
            }
            OptimKind::RAdam => {
                let cfg = ParamsRAdam { lr, ..Default::default() };
                Optim::RAdam(RAdam::new(vars, cfg)?)
            }
            OptimKind::RMSprop => {
                let cfg = ParamsRMSprop { lr, ..Default::default() };
                Optim::RMSprop(RMSprop::new(vars, cfg)?)
            }
        };
        Ok(Optimize { optim })
    }

    /// Perform one optimization step using stored optimizer.
    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        match &mut self.optim {
            Optim::SGD(opt)      => opt.step(grads),
            Optim::Adadelta(opt) => opt.step(grads),
            Optim::Adagrad(opt)  => opt.step(grads),
            Optim::Adam(opt)     => opt.step(grads),
            Optim::Adamax(opt)   => opt.step(grads),
            Optim::NAdam(opt)    => opt.step(grads),
            Optim::RAdam(opt)    => opt.step(grads),
            Optim::RMSprop(opt)  => opt.step(grads),
        }
    }

    /// Convenience: backward pass on loss and step update
    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        match &mut self.optim {
            Optim::SGD(opt)      => opt.backward_step(loss),
            Optim::Adadelta(opt) => opt.backward_step(loss),
            Optim::Adagrad(opt)  => opt.backward_step(loss),
            Optim::Adam(opt)     => opt.backward_step(loss),
            Optim::Adamax(opt)   => opt.backward_step(loss),
            Optim::NAdam(opt)    => opt.backward_step(loss),
            Optim::RAdam(opt)    => opt.backward_step(loss),
            Optim::RMSprop(opt)  => opt.backward_step(loss),
        }
    }
}
