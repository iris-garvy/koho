use candle_core::backprop::GradStore;
use candle_core::{Error, Result as CandleResult, Tensor, Var};
use candle_nn::Optimizer;
use candle_optimisers::{
    adadelta::{Adadelta, ParamsAdaDelta},
    adagrad::{Adagrad, ParamsAdaGrad},
    adam::{Adam, ParamsAdam},
    adamax::{Adamax, ParamsAdaMax},
    esgd::{ParamsSGD, SGD},
    nadam::{NAdam, ParamsNAdam},
    radam::{ParamsRAdam, RAdam},
    rmsprop::{ParamsRMSprop, RMSprop},
};
use candle_optimisers::{Decay, Momentum};

use crate::error::KohoError;

/// Extra parameters passed as input during to the `Optimize` creation for the network.
pub enum OptimizerParams {
    SGD {
        dampening: Option<f64>,
        weight_decay: Option<Decay>,
        momentum: Option<Momentum>,
    },
    AdamW {
        beta_1: f64,
        beta_2: f64,
        eps: f64,
        weight_decay: Option<Decay>,
        amsgrad: bool,
    },
    Else,
}

/// Enum to represent which optimizer type to use
#[derive(Debug, Clone)]
pub enum OptimKind {
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    AdamW, // will use Adam with decoupled weight decay
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
    kind: OptimKind,
    params: OptimizerParams,
    lr: f64,
    optim: Option<Optim>,
}

impl Optimize {
    /// Factory method to create the appropriate optimizer.
    pub fn new(kind: OptimKind, params: OptimizerParams, lr: f64) -> Self {
        Self {
            kind,
            params,
            lr,
            optim: None,
        }
    }
    /// Update the weights passed to the optimizer
    pub fn new_weights(&mut self, vars: Vec<Var>) -> CandleResult<()> {
        let optim = match self.kind {
            OptimKind::SGD => {
                if let OptimizerParams::SGD {
                    dampening,
                    weight_decay,
                    momentum,
                } = self.params
                {
                    if let Some(dampening) = dampening {
                        let cfg = ParamsSGD {
                            lr: self.lr,
                            weight_decay,
                            momentum,
                            dampening,
                        };
                        Optim::SGD(SGD::new(vars, cfg)?)
                    } else {
                        let cfg = ParamsSGD {
                            lr: self.lr,
                            weight_decay,
                            momentum,
                            dampening: 0.0,
                        };
                        Optim::SGD(SGD::new(vars, cfg)?)
                    }
                } else {
                    return Err(Error::Msg(
                        "Parameters provided were incorrect given the OptimKind".to_owned(),
                    ));
                }
            }
            OptimKind::Adadelta => {
                let cfg = ParamsAdaDelta {
                    lr: self.lr,
                    ..Default::default()
                };
                Optim::Adadelta(Adadelta::new(vars, cfg)?)
            }
            OptimKind::Adagrad => {
                let cfg = ParamsAdaGrad {
                    lr: self.lr,
                    ..Default::default()
                };
                Optim::Adagrad(Adagrad::new(vars, cfg)?)
            }
            OptimKind::Adam => {
                let cfg = ParamsAdam {
                    lr: self.lr,
                    ..Default::default()
                };
                Optim::Adam(Adam::new(vars, cfg)?)
            }
            OptimKind::AdamW => {
                if let OptimizerParams::AdamW {
                    beta_1,
                    beta_2,
                    eps,
                    weight_decay,
                    amsgrad,
                } = self.params
                {
                    let cfg = ParamsAdam {
                        lr: self.lr,
                        beta_1,
                        beta_2,
                        eps,
                        weight_decay,
                        amsgrad,
                    };
                    Optim::Adam(Adam::new(vars, cfg)?) // still stored as Optim::Adam
                } else {
                    return Err(Error::Msg(
                        "Parameters provided were incorrect given the OptimKind".to_owned(),
                    ));
                }
            }
            OptimKind::Adamax => {
                let cfg = ParamsAdaMax {
                    lr: self.lr,
                    ..Default::default()
                };
                Optim::Adamax(Adamax::new(vars, cfg)?)
            }
            OptimKind::NAdam => {
                let cfg = ParamsNAdam {
                    lr: self.lr,
                    ..Default::default()
                };
                Optim::NAdam(NAdam::new(vars, cfg)?)
            }
            OptimKind::RAdam => {
                let cfg = ParamsRAdam {
                    lr: self.lr,
                    ..Default::default()
                };
                Optim::RAdam(RAdam::new(vars, cfg)?)
            }
            OptimKind::RMSprop => {
                let cfg = ParamsRMSprop {
                    lr: self.lr,
                    ..Default::default()
                };
                Optim::RMSprop(RMSprop::new(vars, cfg)?)
            }
        };
        self.optim = Some(optim);
        Ok(())
    }

    /// Perform one optimization step using stored optimizer.
    pub fn step(&mut self, grads: &GradStore) -> CandleResult<()> {
        match &mut self.optim.as_mut().unwrap() {
            Optim::SGD(opt) => opt.step(grads),
            Optim::Adadelta(opt) => opt.step(grads),
            Optim::Adagrad(opt) => opt.step(grads),
            Optim::Adam(opt) => opt.step(grads),
            Optim::Adamax(opt) => opt.step(grads),
            Optim::NAdam(opt) => opt.step(grads),
            Optim::RAdam(opt) => opt.step(grads),
            Optim::RMSprop(opt) => opt.step(grads),
        }
    }

    /// Convenience: backward pass on loss and step update
    pub fn backward_step(&mut self, loss: &Tensor) -> CandleResult<()> {
        match &mut self.optim.as_mut().unwrap() {
            Optim::SGD(opt) => opt.backward_step(loss),
            Optim::Adadelta(opt) => opt.backward_step(loss),
            Optim::Adagrad(opt) => opt.backward_step(loss),
            Optim::Adam(opt) => opt.backward_step(loss),
            Optim::Adamax(opt) => opt.backward_step(loss),
            Optim::NAdam(opt) => opt.backward_step(loss),
            Optim::RAdam(opt) => opt.backward_step(loss),
            Optim::RMSprop(opt) => opt.backward_step(loss),
        }
    }

    /// claims the mutated weights from the Optim enum structs.
    pub fn into_inner(&mut self) -> Result<Vec<Var>, KohoError> {
        if self.optim.is_none() {
            return Err(KohoError::Msg(
                "No optimizer step set currently".to_string(),
            ));
        }
        let inner = std::mem::take(&mut self.optim).unwrap();
        Ok(match inner {
            Optim::SGD(sgd) => sgd.into_inner(),
            Optim::Adadelta(adadelta) => adadelta.into_inner(),
            Optim::Adagrad(adagrad) => adagrad.into_inner(),
            Optim::Adam(adam) => adam.into_inner(),
            Optim::Adamax(adamax) => adamax.into_inner(),
            Optim::NAdam(nadam) => nadam.into_inner(),
            Optim::RAdam(radam) => radam.into_inner(),
            Optim::RMSprop(rmsprop) => rmsprop.into_inner(),
        })
    }
}

impl Default for Optimize {
    fn default() -> Self {
        Self {
            kind: OptimKind::Adam,
            params: OptimizerParams::Else,
            lr: 0.001,
            optim: None,
        }
    }
}
