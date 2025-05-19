use candle_core::{DType, Device, Error, Result as CandleResult, Tensor, WithDType};
use std::collections::HashMap;

use crate::{
    error::MathError, math::{cell::{Cell, OpenSet, Skeleton}, tensors::{Matrix, Vector}},
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
    pub cells: Skeleton<O>,
    pub section_spaces: Vec<Vec<Section>>,
    pub restrictions: HashMap<(usize, usize, usize, usize), Matrix>,
    pub interlinked: HashMap<(usize, usize, usize, usize), i8>,
    pub global_sections: Vec<Section>,
    device: Device,
    dtype: DType,
}

impl<O: OpenSet> CellularSheaf<O> {
    pub fn init(dtype: DType, device: Device) -> Self {
        Self {
            cells: Skeleton::init(),
            section_spaces: Vec::new(),
            restrictions: HashMap::new(),
            interlinked: HashMap::new(),
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
        let idx = self.cells.attach(cell)?;
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
        final_k: usize,
        final_cell: usize,
        map: Matrix,
        interlink: i8
    ) -> Result<(), MathError> {
        if cell_id >= self.cells.cells[k].len() || final_cell >= self.cells.cells[k].len() {
            return Err(MathError::InvalidCellIdx);
        }
        if self.cells.cells[k][cell_id].dimension >= self.cells.cells[k][final_cell].dimension
        {
            return Err(MathError::NoRestrictionDefined);
        }
        self.restrictions.insert((k, cell_id, final_k, final_cell), map);
        self.interlinked.insert((k, cell_id, final_k, final_cell), interlink);
        Ok(())
    }

    pub fn k_coboundary(
        &self,
        k: usize,
        stalk_dim: usize
    ) -> Result<Matrix, MathError> {
        let mut output = Vec::new();
        for (i, _) in self.section_spaces[k].iter().enumerate() {
            let (_, upper) = &self.cells.filter_incident_by_dim(k, i)?;
            let mut rows = Vector::from_slice(&vec![0f32; stalk_dim], stalk_dim, self.device.clone(), self.dtype).map_err(MathError::Candle)?;
            for (j, k_plus) in upper {
                let section = &self.section_spaces[*j][*k_plus];
                let stalk_dim = self.section_spaces[*j][*k_plus].0.dimension();
                let mut acc = Vector::from_slice(&vec![0f32; stalk_dim], stalk_dim, self.device.clone(), self.dtype).map_err(MathError::Candle)?;
                
                if let Some(r) = self.restrictions.get(&(*j, *k_plus, k, i)) {
                    // R: Matrix mapping section_spaces[k][i] → section_spaces[k+1][j]
                    let mut piece = r.matvec(&section.0).map_err(MathError::Candle)?;
                    if *self.interlinked.get(&(*j, *k_plus, k, i)).unwrap_or(&1) < 0 {
                        piece = piece.scale(-1.0).map_err(MathError::Candle)?;
                    }
                    acc = acc.add(&piece).map_err(MathError::Candle)?;
                }
                rows = rows.add(&acc).map_err(MathError::Candle)?;
            }
            output.push(rows);
        }
        Matrix::from_vecs(output).map_err(MathError::Candle)
    }

}
