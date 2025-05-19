use candle_core::{DType, Device, Result as CandleResult, Tensor, WithDType};
use std::collections::HashMap;

use crate::{
    error::MathError, math::{cell::{Cell, OpenSet, Skeleton}, tensors::Vector},
};

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Point<T: Eq + std::hash::Hash + Clone + Sized>(T);

pub struct Section(Vector);

impl Section {
    pub fn new<T: WithDType>(tensor: &[T], dimension: usize, device: Device, dtype: DType) -> CandleResult<Self> {
        Ok(Self(Vector::from_slice(tensor, dimension, device, dtype)?))
    }
}
pub struct CellularSheaf<O: OpenSet> {
    pub cw: Skeleton<O>,
    pub section_spaces: Vec<Vec<Section>>,
    pub restrictions: HashMap<(usize, usize, usize), Tensor>,
    pub global_sections: Vec<Section>,
    device: Device,
    dtype: DType,
}

impl<O: OpenSet> CellularSheaf<O> {
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
        cell: Cell<O>,
        data: Section,
    ) -> Result<(usize, usize), MathError> {
        let k = cell.dimension;
        let idx = self.cw.attach(cell)?;
        let current_dim = self.section_spaces.len();
        if current_dim < k {
            return Err(MathError::DimensionMismatch)
        } else if current_dim == k {
            self.section_spaces.push(vec![data]);
        }
        Ok((k, idx))
    }
    /// Update section data
    pub fn update_stalk(&mut self, k: usize, cell_idx: usize, val: Vector) -> Result<(), MathError> {
        if cell_idx >= self.section_spaces.len() {
            return Err(MathError::InvalidCellIdx);
        }
        self.section_spaces[k][cell_idx].0 = val;
        Ok(())
    }

    pub fn set_restriction(
        &mut self,
        k: usize,
        cell_id: usize,
        final_cell: usize,
        map: Tensor,
    ) -> Result<(), MathError> {
        if cell_id >= self.cw.cells[k].len() || final_cell >= self.cw.cells[k].len() {
            return Err(MathError::InvalidCellIdx);
        }
        if self.cw.cells[k][cell_id].dimension >= self.cw.cells[k][final_cell].dimension
        {
            return Err(MathError::NoRestrictionDefined);
        }
        self.restrictions.insert((k, cell_id, final_cell), map);
        Ok(())
    }

    pub fn k_coboundary(
        &self,
        k_cochain: Tensor 
    ) {

    }
}
