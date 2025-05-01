use candle_core::{DType, Device, Result as CandleResult, Tensor};
use std::collections::{HashMap, HashSet};

use crate::{
    cw::{KCell, Skeleton},
    error::MathError,
};

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Point<T: Eq + std::hash::Hash + Clone + Sized>(T);

pub trait OpenSet: IntoIterator<Item = Self::Point> + Clone {
    type Point: Eq + std::hash::Hash;
    fn union(&self, other: Self) -> Self;
    fn intersection(&self, other: Self) -> Self;
    fn from(iter: Box<dyn Iterator<Item = Self::Point>>) -> Self;
}

pub trait Topology {
    type Point;
    type OpenSet: OpenSet<Point = Self::Point>;
    fn points(&self) -> HashSet<Self::Point>;
    fn neighborhoods(&self, point: Self::Point) -> HashSet<Self::OpenSet>;
    fn is_open(&self, set: Self::OpenSet) -> bool;
}

pub struct Sections {
    pub data: Tensor,
    pub bases: Option<Tensor>,
}

impl Sections {
    pub fn new(dim: usize, device: &Device, dtype: DType) -> CandleResult<Self> {
        Ok(Self {
            data: Tensor::zeros(&[0, dim], dtype, device)?,
            bases: None,
        })
    }
    pub fn add_bases(&mut self, bases: Tensor) {
        self.bases = Some(bases)
    }
    pub fn add_section_data(&mut self, section: Tensor) -> CandleResult<()> {
        self.data = Tensor::cat(&[self.data.clone(), section], 0)?;
        Ok(())
    }
}
pub struct CellularSheaf<T: Eq + std::hash::Hash + Clone, O: OpenSet> {
    pub cw: Skeleton<T, O>,
    pub section_spaces: Vec<Sections>,
    pub restrictions: HashMap<(usize, usize), Tensor>,
    pub global_sections: Vec<Sections>,
    device: Device,
    dtype: DType,
}

impl<T: Eq + std::hash::Hash + Clone, O: OpenSet> CellularSheaf<T, O> {
    pub fn init(dtype: DType, device: Device) -> Self {
        Self {
            cw: Skeleton::init(),
            section_spaces: Vec::new(),
            restrictions: HashMap::new(),
            global_sections: Vec::new(),
            device,
            dtype,
        }
    }
    /// Attaches a cell to the base cell complex, and spawns a section space in the cellular sheaf
    pub fn attach(
        &mut self,
        cell: Box<dyn KCell<T, O>>,
        data: Option<Sections>,
        section_dimension: usize,
    ) -> Result<(), MathError> {
        self.section_spaces.push(if let Some(item) = data {
            item
        } else {
            Sections::new(section_dimension, &self.device, self.dtype).map_err(MathError::Candle)?
        });
        self.cw.attach(cell)?;
        Ok(())
    }
    /// Update section data
    pub fn update(&mut self, cell_idx: usize, val: Tensor) -> Result<(), MathError> {
        if cell_idx >= self.section_spaces.len() {
            return Err(MathError::InvalidCellIdx);
        }
        self.section_spaces[cell_idx].data = val;
        Ok(())
    }

    pub fn new_data(&mut self, cell_idx: usize, val: Tensor) -> Result<(), MathError> {
        if cell_idx >= self.section_spaces.len() {
            return Err(MathError::InvalidCellIdx);
        }
        self.section_spaces[cell_idx]
            .add_section_data(val)
            .map_err(MathError::Candle)?;
        Ok(())
    }

    pub fn set_restriction(
        &mut self,
        start_cell: usize,
        final_cell: usize,
        map: Tensor,
    ) -> Result<(), MathError> {
        if start_cell >= self.cw.cells.len() || final_cell >= self.cw.cells.len() {
            return Err(MathError::InvalidCellIdx);
        }
        if self.cw.cells[start_cell].cell.dimension() >= self.cw.cells[final_cell].cell.dimension()
        {
            return Err(MathError::NoRestrictionDefined);
        }
        self.restrictions.insert((start_cell, final_cell), map);
        Ok(())
    }

    pub fn k_coboundary(&self, cell_idx: usize) -> Result<Vec<(Tensor, usize)>, MathError> {
        if cell_idx >= self.cw.cells.len() {
            return Err(MathError::InvalidCellIdx);
        }
        let mut results = Vec::new();
        for i in self.cw.filter_incident_by_dim(cell_idx)?.1 {
            let restriction = self.restrictions.get(&(cell_idx, i));
            if restriction.is_none() {
                return Err(MathError::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            results.push((
                restrict.mul(&self.section_spaces[cell_idx].data).unwrap(),
                i,
            ));
        }
        Ok(results)
    }

    pub fn k_coboundary_adjoint(
        &self,
        cell_idx: usize,
        cochain: Vec<(Tensor, usize)>,
    ) -> Result<Tensor, MathError> {
        if cell_idx >= self.cw.cells.len() {
            return Err(MathError::InvalidCellIdx);
        }
        let idxs = cochain.iter().map(|(_, x)| *x).collect::<Vec<_>>();
        let mut results = None;
        for i in idxs {
            if !self.cw.filter_incident_by_dim(cell_idx)?.1.contains(&i) {
                return Err(MathError::BadCochain);
            };
            let restriction = self.restrictions.get(&(cell_idx, i));
            if restriction.is_none() {
                return Err(MathError::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            let new = restrict
                .transpose(0, 1)
                .map_err(MathError::Candle)?
                .mul(&cochain[i].0)
                .map_err(MathError::Candle)?;
            if results.is_none() {
                results = Some(new.clone());
            }
            results = Some(results.unwrap().add(&new).map_err(MathError::Candle)?)
        }
        Ok(results.unwrap())
    }

    pub fn k_minus_1_coboundary_adjoint(
        &self,
        cell_idx: usize,
    ) -> Result<Vec<(Tensor, usize)>, MathError> {
        if cell_idx >= self.cw.cells.len() {
            return Err(MathError::InvalidCellIdx);
        }
        let data = &self.section_spaces[cell_idx].data;
        let mut results = Vec::new();
        for i in self.cw.filter_incident_by_dim(cell_idx)?.0 {
            let restriction = self.restrictions.get(&(i, cell_idx));
            if restriction.is_none() {
                return Err(MathError::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();

            results.push((
                restrict
                    .transpose(0, 1)
                    .map_err(MathError::Candle)?
                    .mul(data)
                    .map_err(MathError::Candle)?,
                i,
            ));
        }
        Ok(results)
    }

    pub fn k_minus_1_coboundary(
        &self,
        cell_idx: usize,
        cochain: Vec<(Tensor, usize)>,
    ) -> Result<Tensor, MathError> {
        if cell_idx >= self.cw.cells.len() {
            return Err(MathError::InvalidCellIdx);
        }
        let idxs = cochain.iter().map(|(_, x)| *x).collect::<Vec<_>>();
        let mut results = None;
        for i in idxs {
            if !self.cw.filter_incident_by_dim(cell_idx)?.0.contains(&i) {
                return Err(MathError::BadCochain);
            };
            let restriction = self.restrictions.get(&(i, cell_idx));
            if restriction.is_none() {
                return Err(MathError::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            let new = restrict.mul(&cochain[i].0).map_err(MathError::Candle)?;
            if results.is_none() {
                results = Some(new.clone());
            }
            results = Some(results.unwrap().add(&new).map_err(MathError::Candle)?);
        }
        Ok(results.unwrap())
    }

    pub fn local_up_laplacian(&self, cell_idx: usize) -> Result<Tensor, MathError> {
        let (_, up) = self.cw.filter_incident_by_dim(cell_idx)?;
        let mut matrix = None;
        for i in up {
            let restriction = self.restrictions.get(&(cell_idx, i));
            if restriction.is_none() {
                return Err(MathError::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            let new = restrict
                .transpose(0, 1)
                .map_err(MathError::Candle)?
                .mul(restrict)
                .map_err(MathError::Candle)?;
            if matrix.is_none() {
                matrix = Some(new.clone());
            }
            matrix = Some(matrix.unwrap().add(&new).map_err(MathError::Candle)?);
        }
        Ok(matrix.unwrap())
    }

    pub fn local_down_laplacian(&self, cell_idx: usize) -> Result<Tensor, MathError> {
        let (down, _) = self.cw.filter_incident_by_dim(cell_idx)?;
        let mut matrix = None;
        for i in down {
            let restriction = self.restrictions.get(&(i, cell_idx));
            if restriction.is_none() {
                return Err(MathError::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            let new = restrict
                .mul(&restrict.transpose(0, 1).map_err(MathError::Candle)?)
                .map_err(MathError::Candle)?;
            if matrix.is_none() {
                matrix = Some(new.clone());
            }
            matrix = Some(matrix.unwrap().add(&new).map_err(MathError::Candle)?);
        }
        Ok(matrix.unwrap())
    }

    pub fn local_laplacian(&self, cell_idx: usize) -> Result<Tensor, MathError> {
        self.local_up_laplacian(cell_idx)?
            .add(&self.local_down_laplacian(cell_idx)?)
            .map_err(MathError::Candle)
    }

    pub fn k_laplacian(&self, k: usize) -> Result<Tensor, MathError> {
        let mut valid = Vec::new();
        self.cw.cells.iter().enumerate().for_each(|(i, x)| {
            if x.cell.dimension() == k {
                valid.push(i)
            }
        });
        let dim = self.section_spaces[valid[0]].data.dims()[0];
        if valid.is_empty() {
            return Err(MathError::NoCellsofDimensionK);
        }
        let mut global = None;
        for i in valid {
            if self.section_spaces[i].data.dims()[0] != dim {
                return Err(MathError::DimensionMismatch);
            }
            if global.is_none() {
                global = Some(self.local_laplacian(i)?);
            }
            global = Some(
                global
                    .unwrap()
                    .add(&self.local_laplacian(i)?)
                    .map_err(MathError::Candle)?,
            );
        }
        Ok(global.unwrap())
    }

    pub fn check_glue(&mut self, start_cell: usize, final_cell: usize) -> Result<bool, MathError> {
        if start_cell >= self.cw.cells.len() || final_cell >= self.cw.cells.len() {
            return Err(MathError::InvalidCellIdx);
        }
        if self.cw.cells[start_cell].cell.dimension() <= self.cw.cells[final_cell].cell.dimension()
        {
            return Err(MathError::DimensionMismatch);
        }
        let restriction = self.restrictions.get(&(start_cell, final_cell));
        if restriction.is_none() {
            return Err(MathError::NoRestrictionDefined);
        }
        let restrict = restriction.unwrap();
        let eq = restrict
            .mul(&self.section_spaces[start_cell].data)
            .unwrap()
            .eq(&self.section_spaces[final_cell].data)
            .map_err(MathError::Candle)?;
        let sum = eq.sum_all().map_err(MathError::Candle)?;
        let count = sum.elem_count();
        if sum.to_scalar::<u32>().map_err(MathError::Candle)? == count as u32 {
            return Ok(false);
        }
        Ok(true)
    }
}
