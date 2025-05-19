use candle_core::{DType, Device, Result as CandleResult, WithDType};
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
        k_cochain: Vec<Vector>,
        stalk_dim_output: usize
    ) -> Result<Vec<Vector>, MathError> {
        let num_k_plus_1_cells = self.cells.cells.get(k + 1).map_or(0, |cells| cells.len());
        if num_k_plus_1_cells == 0 && k + 1 < self.cells.cells.len() {
            return Ok(Vec::new());
        } else if k + 1 >= self.cells.cells.len() {
            return Err(MathError::DimensionMismatch);
        }
        let mut output_k_plus_1_cochain = vec![
            Vector::from_slice(&vec![0f32; stalk_dim_output], stalk_dim_output, self.device.clone(), self.dtype).map_err(MathError::Candle)?;
            num_k_plus_1_cells
        ];
        for tau_idx in 0..num_k_plus_1_cells {
            let tau_cell_dim = k + 1;
            let (lower_incident_cells, _) = &self.cells.filter_incident_by_dim(tau_cell_dim, tau_idx)?;
            for (sigma_dim, sigma_idx) in lower_incident_cells {
                if *sigma_dim != k { continue; }
                let x_sigma = &k_cochain[*sigma_idx];
                if let Some(r) = self.restrictions.get(&(*sigma_dim, *sigma_idx, tau_cell_dim, tau_idx)) {
                    let mut term = r.matvec(x_sigma).map_err(MathError::Candle)?;

                    if let Some(incidence_sign) = self.interlinked.get(&(*sigma_dim, *sigma_idx, tau_cell_dim, tau_idx)) {
                        if *incidence_sign < 0 {
                            term = term.scale(-1.0).map_err(MathError::Candle)?;
                        }
                    }
                    output_k_plus_1_cochain[tau_idx] = output_k_plus_1_cochain[tau_idx].add(&term).map_err(MathError::Candle)?;
                }
            }
        }
        Ok(output_k_plus_1_cochain)
    }

    /// Computes the adjoint of the k-th coboundary operator ((delta^k)*).
    pub fn k_adjoint_coboundary(
        &self,
        k: usize,
        k_coboundary_output: Vec<Vector>,
        stalk_dim_output: usize
    ) -> Result<Vec<Vector>, MathError> {
        let num_k_cells = self.cells.cells.get(k).map_or(0, |cells| cells.len());
        if num_k_cells == 0 && k < self.cells.cells.len() {
            return Ok(Vec::new());
        } else if k >= self.cells.cells.len() {
            return Err(MathError::DimensionMismatch);
        }

        let mut output_k_cochain = vec![
            Vector::from_slice(&vec![0f32; stalk_dim_output], stalk_dim_output, self.device.clone(), self.dtype).map_err(MathError::Candle)?;
            num_k_cells
        ];

        for tau_idx in 0..k_coboundary_output.len() {
            let y_tau = &k_coboundary_output[tau_idx];
            let tau_cell_dim = k + 1;

            let (lower_incident_cells, _) = &self.cells.filter_incident_by_dim(tau_cell_dim, tau_idx)?;

            for (sigma_dim, sigma_idx) in lower_incident_cells {
                if *sigma_dim != k { continue; }

                if let Some(r) = self.restrictions.get(&(*sigma_dim, *sigma_idx, tau_cell_dim, tau_idx)) {
                    let mut term = r.transpose().map_err(MathError::Candle)?.matvec(y_tau).map_err(MathError::Candle)?;

                    if let Some(incidence_sign) = self.interlinked.get(&(*sigma_dim, *sigma_idx, tau_cell_dim, tau_idx)) {
                        if *incidence_sign < 0 {
                            term = term.scale(-1.0).map_err(MathError::Candle)?;
                        }
                    }

                    output_k_cochain[*sigma_idx] = output_k_cochain[*sigma_idx].add(&term).map_err(MathError::Candle)?;
                }
            }
        }
        Ok(output_k_cochain)
    }

    /// Retrieves the cochain (Vec<Vector>) for a given dimension k.
    pub fn get_k_cochain(&self, k: usize) -> Result<Vec<Vector>, MathError> {
        if k >= self.section_spaces.len() {
            return Err(MathError::DimensionMismatch);
        }
        let k_sections = &self.section_spaces[k];
        let k_cochain: Vec<Vector> = k_sections
            .iter()
            .map(|section| section.0.clone())
            .collect();

        Ok(k_cochain)
    }

    pub fn k_hodge_laplacian(&self, k: usize, k_cochain: Matrix, k_stalk_dim: usize, k_plus_stalk_dim: usize, k_minus_stalk_dim: usize) -> Result<Matrix, MathError> {
        let vecs = k_cochain.to_vectors().map_err(MathError::Candle)?;
        let up_a = self.k_coboundary(k, vecs.clone(), k_plus_stalk_dim)?;
        let up_b = self.k_adjoint_coboundary(k, up_a, k_stalk_dim)?;

        let down_a = self.k_adjoint_coboundary(k, vecs, k_minus_stalk_dim)?;
        let down_b = self.k_coboundary(k, down_a, k_stalk_dim)?;
        let out = Matrix::from_vecs(up_b).map_err(MathError::Candle)?.add(&Matrix::from_vecs(down_b).map_err(MathError::Candle)?).map_err(MathError::Candle)?;
        Ok(out)
    }
}
