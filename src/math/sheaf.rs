//! Cellular sheaf mathematics implementation.
//!
//! This module provides data structures and algorithms for working with cellular sheaves,
//! which are mathematical constructs that assign vector spaces to cells in a cell complex
//! and linear maps between these spaces. Cellular sheaves can be used for data analysis,
//! signal processing on topological domains, and other applications where the topology
//! of data is important.

use candle_core::{DType, Device, Result as CandleResult, WithDType};
use std::collections::HashMap;

use crate::{
    error::KohoError,
    math::{
        cell::{Cell, OpenSet, Skeleton},
        tensors::{Matrix, Vector},
    },
};

/// Represents a section of a sheaf, which is a vector of data associated with a cell.
pub struct Section(pub Vector);

impl Section {
    /// Creates a new section from a slice of data.
    ///
    /// # Arguments
    /// * `tensor` - The data to initialize the section with
    /// * `dimension` - The dimension of the vector space this section lives in
    /// * `device` - The device to store the tensor on (CPU/GPU)
    /// * `dtype` - The data type of the tensor elements
    pub fn new<T: WithDType>(
        tensor: &[T],
        dimension: usize,
        device: Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        Ok(Self(Vector::from_slice(tensor, dimension, device, dtype)?))
    }
}

/// A cellular sheaf defined over a cell complex.
///
/// Cellular sheaves assign data (vector spaces) to cells in a cell complex,
/// along with restriction maps that specify how data on higher-dimensional cells
/// relates to data on lower-dimensional cells.
pub struct CellularSheaf<O: OpenSet> {
    /// The underlying cell complex structure
    pub cells: Skeleton<O>,
    /// Vector spaces (stalks) assigned to each cell, organized by dimension
    pub section_spaces: Vec<Vec<Section>>,
    /// Maps between vector spaces on different cells (restriction maps)
    pub restrictions: HashMap<(usize, usize, usize, usize), Matrix>,
    /// Signs indicating the orientation relationship between cells
    pub interlinked: HashMap<(usize, usize, usize, usize), i8>,
    /// Global sections of the sheaf (data consistent across all cells)
    pub global_sections: Vec<Section>,
    /// Device for tensor operations (CPU/GPU)
    pub device: Device,
    /// Data type for tensor elements
    pub dtype: DType,
}

impl<O: OpenSet> CellularSheaf<O> {
    /// Initializes a new, empty cellular sheaf.
    ///
    /// # Arguments
    /// * `dtype` - The data type to use for tensor elements
    /// * `device` - The device to use for tensor operations
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

    /// Attaches a cell to the base cell complex and creates a corresponding section space.
    ///
    /// # Arguments
    /// * `cell` - The cell to attach
    /// * `data` - The section data to associate with this cell
    ///
    /// # Returns
    /// A tuple of the dimension and index of the newly attached cell
    pub fn attach(&mut self, cell: Cell<O>, data: Section) -> Result<(usize, usize), KohoError> {
        let k = cell.dimension;
        let idx = self.cells.attach(cell)?;
        let current_dim = self.section_spaces.len();
        if current_dim < k {
            return Err(KohoError::DimensionMismatch);
        } else if current_dim == k {
            self.section_spaces.push(vec![data]);
        }
        Ok((k, idx))
    }

    /// Sets a restriction map between two cells.
    ///
    /// Restriction maps define how data transforms when moving from one cell to another.
    ///
    /// # Arguments
    /// * `k` - The dimension of the source cell
    /// * `cell_id` - The index of the source cell
    /// * `final_k` - The dimension of the target cell
    /// * `final_cell` - The index of the target cell
    /// * `map` - The linear transformation matrix
    /// * `interlink` - Sign indicating orientation relationship (-1 or 1)
    pub fn set_restriction(
        &mut self,
        k: usize,
        cell_id: usize,
        final_k: usize,
        final_cell: usize,
        map: Matrix,
        interlink: i8,
    ) -> Result<(), KohoError> {
        if cell_id >= self.cells.cells[k].len() || final_cell >= self.cells.cells[final_k].len() {
            return Err(KohoError::InvalidCellIdx);
        }
        if self.cells.cells[k][cell_id].dimension >= self.cells.cells[final_k][final_cell].dimension {
            return Err(KohoError::NoRestrictionDefined);
        }
        self.restrictions
            .insert((k, cell_id, final_k, final_cell), map);
        self.interlinked
            .insert((k, cell_id, final_k, final_cell), interlink);
        Ok(())
    }

    /// Computes the k-coboundary operator, which maps k-cochains to (k+1)-cochains.
    ///
    /// This is a key operator in sheaf cohomology that captures how data propagates
    /// from lower to higher-dimensional cells.
    ///
    /// # Arguments
    /// * `k` - The dimension of the input cochain
    /// * `k_cochain` - The input k-cochain as a vector of vectors
    /// * `stalk_dim_output` - The dimension of the output stalk
    ///
    /// # Returns
    /// The resulting (k+1)-cochain
    fn k_coboundary(
        &self,
        k: usize,
        k_cochain: Vec<Vector>,
        stalk_dim_output: usize,
    ) -> Result<Vec<Vector>, KohoError> {
        let num_k_plus_1_cells = self.cells.cells.get(k + 1).map_or(0, |cells| cells.len());
        if num_k_plus_1_cells == 0 && k + 1 < self.cells.cells.len() {
            return Ok(Vec::new());
        } else if k + 1 >= self.cells.cells.len() {
            return Err(KohoError::DimensionMismatch);
        }
        let mut output_k_plus_1_cochain = vec![
            Vector::from_slice(
                &vec![0f32; stalk_dim_output],
                stalk_dim_output,
                self.device.clone(),
                self.dtype
            )
            .map_err(KohoError::Candle)?;
            num_k_plus_1_cells
        ];
        for (tau_idx, tau) in output_k_plus_1_cochain
            .iter_mut()
            .enumerate()
            .take(num_k_plus_1_cells)
        {
            let tau_cell_dim = k + 1;
            let (lower_incident_cells, _) =
                &self.cells.filter_incident_by_dim(tau_cell_dim, tau_idx)?;
            for (sigma_dim, sigma_idx) in lower_incident_cells {
                if *sigma_dim != k {
                    continue;
                }
                let x_sigma = &k_cochain[*sigma_idx];
                if let Some(r) =
                    self.restrictions
                        .get(&(*sigma_dim, *sigma_idx, tau_cell_dim, tau_idx))
                {
                    let mut term = r.matvec(x_sigma).map_err(KohoError::Candle)?;

                    if let Some(incidence_sign) =
                        self.interlinked
                            .get(&(*sigma_dim, *sigma_idx, tau_cell_dim, tau_idx))
                    {
                        if *incidence_sign < 0 {
                            term = term.scale(-1.0).map_err(KohoError::Candle)?;
                        }
                    }
                    *tau = tau.add(&term).map_err(KohoError::Candle)?;
                }
            }
        }
        Ok(output_k_plus_1_cochain)
    }

    /// Computes the adjoint of the k-th coboundary operator ((delta^k)*).
    ///
    /// This operator goes in the reverse direction of the coboundary, from (k+1)-cochains
    /// to k-cochains, and is essential for defining the Hodge Laplacian.
    ///
    /// # Arguments
    /// * `k` - The dimension to compute the adjoint for
    /// * `k_coboundary_output` - The (k+1)-cochain input
    /// * `stalk_dim_output` - The dimension of the output stalk
    ///
    /// # Returns
    /// The resulting k-cochain
    fn k_adjoint_coboundary(
        &self,
        k: usize,
        k_coboundary_output: Vec<Vector>,
        stalk_dim_output: usize,
    ) -> Result<Vec<Vector>, KohoError> {
        let num_k_cells = self.cells.cells.get(k).map_or(0, |cells| cells.len());
        if num_k_cells == 0 && k < self.cells.cells.len() {
            return Ok(Vec::new());
        } else if k >= self.cells.cells.len() {
            return Err(KohoError::DimensionMismatch);
        }

        let mut output_k_cochain = vec![
            Vector::from_slice(
                &vec![0f32; stalk_dim_output],
                stalk_dim_output,
                self.device.clone(),
                self.dtype
            )
            .map_err(KohoError::Candle)?;
            num_k_cells
        ];

        for (tau_idx, y_tau) in k_coboundary_output.iter().enumerate() {
            let tau_cell_dim = k + 1;

            let (lower_incident_cells, _) =
                &self.cells.filter_incident_by_dim(tau_cell_dim, tau_idx)?;

            for (sigma_dim, sigma_idx) in lower_incident_cells {
                if *sigma_dim != k {
                    continue;
                }

                if let Some(r) =
                    self.restrictions
                        .get(&(*sigma_dim, *sigma_idx, tau_cell_dim, tau_idx))
                {
                    let mut term = r
                        .transpose()
                        .map_err(KohoError::Candle)?
                        .matvec(y_tau)
                        .map_err(KohoError::Candle)?;

                    if let Some(incidence_sign) =
                        self.interlinked
                            .get(&(*sigma_dim, *sigma_idx, tau_cell_dim, tau_idx))
                    {
                        if *incidence_sign < 0 {
                            term = term.scale(-1.0).map_err(KohoError::Candle)?;
                        }
                    }

                    output_k_cochain[*sigma_idx] = output_k_cochain[*sigma_idx]
                        .add(&term)
                        .map_err(KohoError::Candle)?;
                }
            }
        }
        Ok(output_k_cochain)
    }

    /// Retrieves the cochain (collection of vector data) for a given dimension k.
    ///
    /// # Arguments
    /// * `k` - The dimension to retrieve the cochain for
    ///
    /// # Returns
    /// A matrix containing all vectors in the k-cochain
    pub fn get_k_cochain(&self, k: usize) -> Result<Matrix, KohoError> {
        if k >= self.section_spaces.len() {
            return Err(KohoError::DimensionMismatch);
        }
        let k_sections = &self.section_spaces[k];
        let k_cochain: Vec<Vector> = k_sections.iter().map(|section| section.0.clone()).collect();
        Matrix::from_vecs(k_cochain).map_err(KohoError::Candle)
    }

    /// Computes the k-th Hodge Laplacian operator.
    ///
    /// The Hodge Laplacian combines the coboundary and its adjoint to create an operator
    /// that measures how "harmonic" data is across the sheaf. It's used for spectral
    /// analysis, smoothing, and finding patterns in data.
    ///
    /// # Arguments
    /// * `k` - The dimension to compute the Laplacian for
    /// * `k_cochain` - The input k-cochain as a matrix
    /// * `down_included` - Whether to include the "down" component (from (k-1)-cells)
    ///
    /// # Returns
    /// The Hodge Laplacian applied to the input cochain
    pub fn k_hodge_laplacian(
        &self,
        k: usize,
        k_cochain: Matrix,
        down_included: bool,
    ) -> Result<Matrix, KohoError> {
        let vecs = k_cochain.to_vectors().map_err(KohoError::Candle)?;

        let k_plus_stalk_dim = self.section_spaces[k + 1][0].0.dimension();
        let k_stalk_dim = self.section_spaces[k][0].0.dimension();
        
        let up_a = self.k_coboundary(k, vecs.clone(), k_plus_stalk_dim)?;
        let up_b = self.k_adjoint_coboundary(k, up_a, k_stalk_dim)?;
        if down_included {
            let k_minus_stalk_dim = self.section_spaces[k - 1][0].0.dimension();
            let down_a = self.k_adjoint_coboundary(k, vecs, k_minus_stalk_dim)?;
            let down_b = self.k_coboundary(k, down_a, k_stalk_dim)?;
            let out = Matrix::from_vecs(up_b)
                .map_err(KohoError::Candle)?
                .add(&Matrix::from_vecs(down_b).map_err(KohoError::Candle)?)
                .map_err(KohoError::Candle)?;
            return Ok(out);
        }
        Matrix::from_vecs(up_b).map_err(KohoError::Candle)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use crate::math::cell::{Attach, Cell, OpenSet, Skeleton};
    use super::*;
    use crate::math::tensors::Matrix;

    /// A trivial point type.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct TestPoint(&'static str);

    /// A super‐minimal OpenSet impl wrapping a HashSet<TestPoint>.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestOpenSet {
        points: HashSet<TestPoint>,
    }

    impl TestOpenSet {
        fn new<I: IntoIterator<Item = TestPoint>>(pts: I) -> Self {
            Self {
                points: pts.into_iter().collect(),
            }
        }
    }

    impl IntoIterator for TestOpenSet {
        type IntoIter = std::collections::hash_set::IntoIter<TestPoint>;
        type Item = TestPoint;

        fn into_iter(self) -> Self::IntoIter {
            self.points.into_iter()
        }
    }

    impl OpenSet for TestOpenSet {
        fn from(iter: Box<dyn Iterator<Item = TestPoint>>) -> Self {
            let pts = iter.collect();
            Self { points: pts }
        }
        type Point = TestPoint;

        fn union(&self, other: &Self) -> Self {
            let mut pts = self.points.clone();
            pts.extend(other.points.iter().cloned());
            Self { points: pts }
        }

        fn intersect(&self, other: &Self) -> Self {
            let pts = self.points.intersection(&other.points).cloned().collect();
            Self { points: pts }
        }

        fn contains(&self, pt: &Self::Point) -> bool {
            self.points.contains(pt)
        }

        fn difference(&self, other: &Self) -> Self {
            let pts = self.points.difference(&other.points).cloned().collect();
            Self { points: pts }
        }

        fn is_empty(&self) -> bool {
            self.points.is_empty()
        }
    }

    // A no-op attach: just returns the same set.
    impl Attach<TestOpenSet> for TestOpenSet {
        fn attach_boundary(&self, _skeleton: &Skeleton<TestOpenSet>) -> TestOpenSet {
            self.clone()
        }
    }


    #[test]
    fn simple_example_sheaf() -> Result<(), KohoError>{
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mut example_sheaf: CellularSheaf<TestOpenSet> = CellularSheaf::init(dtype, device.clone());

        let point_a = Cell::new(
            TestOpenSet::new(vec![TestPoint("A")]), 
            TestOpenSet::new(vec![]), 
            0
        );
        let line_aa = Cell::new(
            TestOpenSet::new(vec![TestPoint("B")]), 
            TestOpenSet::new(vec![TestPoint("A")]), 
            1
        );

        let section_a = Section::new(&[1.0f32, 0.0, 0.0, 1.0], 2, device.clone(), dtype).unwrap();
        let section_aa = Section::new(&[1.0f32, 0.0, 0.0, 1.0], 2, device.clone(), dtype).unwrap();

        match example_sheaf.attach(point_a, section_a) {
            Ok((k, idx)) => {println!("Attached cell of dimension {} at index {}", k, idx);}
            Err(e) => {eprintln!("Failed to attach cell: {:?}", e);}
        }
        match example_sheaf.attach(line_aa, section_aa) {
            Ok((k, idx)) => {println!("Attached cell of dimension {} at index {}", k, idx);}
            Err(e) => {eprintln!("Failed to attach cell: {:?}", e);}
        }

        let identity_restrict = Matrix::from_slice(&[1.0f32, 0.0, 0.0, 1.0], 2, 2,device.clone(),dtype).unwrap();

        match example_sheaf.set_restriction(0,0,1,0,identity_restrict.clone(),0) {
            Ok(()) => {println!("restriction attached");}
            Err(e) => {eprintln!("Failed to attach cell: {:?}", e);}
        } 

        let k_cochain = example_sheaf.get_k_cochain(0).unwrap();
        println!("k_cochain: {:?}", k_cochain);
        let laplacian = example_sheaf.k_hodge_laplacian(0, k_cochain, false)?;
        println!("here's the laplacian {:?}", laplacian.shape());


        Ok(())
    }

}
