use std::collections::{HashMap, HashSet};

use crate::{
    algebra::{add_vectors, Field, Matrix},
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

pub struct Sections<F: Field> {
    pub data: Vec<Vec<F>>,
    pub bases: Option<Vec<Vec<F>>>,
    pub dimension: usize,
}

impl<F: Field> Sections<F> {
    pub fn new(dimension: usize) -> Self {
        Self {
            data: Vec::new(),
            bases: None,
            dimension,
        }
    }
    pub fn add_bases(&mut self, bases: Vec<Vec<F>>) {
        self.bases = Some(bases)
    }
    pub fn add_section(&mut self, section: Vec<F>) {
        self.data.push(section);
    }
}
pub struct CellularSheaf<F: Field, T: Eq + std::hash::Hash + Clone, O: OpenSet> {
    pub cw: Skeleton<T, O>,
    pub section_spaces: Vec<Sections<F>>,
    pub restrictions: HashMap<(usize, usize), Matrix<F>>,
    pub global_sections: Vec<Sections<F>>,
}

impl<F: Field, T: Eq + std::hash::Hash + Clone, O: OpenSet> CellularSheaf<F, T, O> {
    pub fn init() -> Self {
        Self {
            cw: Skeleton::init(),
            section_spaces: Vec::new(),
            restrictions: HashMap::new(),
            global_sections: Vec::new(),
        }
    }

    pub fn attach(
        &mut self,
        cell: Box<dyn KCell<T, O>>,
        data: Option<Sections<F>>,
        section_dimension: usize,
    ) -> Result<(), MathError> {
        self.section_spaces.push(if data.is_some() {
            data.unwrap()
        } else {
            Sections::new(section_dimension)
        });
        self.cw.attach(cell)?;
        Ok(())
    }

    pub fn update(
        &mut self,
        cell_idx: usize,
        data_idx: usize,
        val: Vec<F>,
    ) -> Result<(), MathError> {
        if cell_idx >= self.section_spaces.len() {
            return Err(MathError::InvalidCellIdx);
        }
        if data_idx >= self.section_spaces[cell_idx].data.len() {
            return Err(MathError::InvalidDataIdx);
        }
        self.section_spaces[cell_idx].data[data_idx] = val;
        Ok(())
    }

    pub fn new_data(&mut self, cell_idx: usize, val: Vec<F>) -> Result<(), MathError> {
        if cell_idx >= self.section_spaces.len() {
            return Err(MathError::InvalidCellIdx);
        }
        self.section_spaces[cell_idx].data.push(val);
        Ok(())
    }

    pub fn set_restriction(
        &mut self,
        start_cell: usize,
        final_cell: usize,
        map: Matrix<F>,
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

    pub fn k_coboundary(
        &self,
        cell_idx: usize,
        data_idx: usize,
    ) -> Result<Vec<(Vec<F>, usize)>, MathError> {
        if cell_idx >= self.cw.cells.len() {
            return Err(MathError::InvalidCellIdx);
        }
        if data_idx >= self.section_spaces[cell_idx].data.len() {
            return Err(MathError::InvalidDataIdx);
        }
        let mut results = Vec::new();
        for i in self.cw.filter_incident_by_dim(cell_idx)?.1 {
            let restriction = self.restrictions.get(&(cell_idx, i));
            if restriction.is_none() {
                return Err(MathError::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            results.push((
                restrict
                    .transform(&self.section_spaces[cell_idx].data[data_idx])
                    .unwrap(),
                i,
            ));
        }
        Ok(results)
    }

    pub fn k_coboundary_adjoint(
        &self,
        cell_idx: usize,
        cochain: Vec<(Vec<F>, usize)>,
    ) -> Result<Vec<F>, MathError> {
        if cell_idx >= self.cw.cells.len() {
            return Err(MathError::InvalidCellIdx);
        }
        let idxs = cochain.iter().map(|(_, x)| *x).collect::<Vec<_>>();
        let domain_bases = &self.section_spaces[cell_idx].bases;
        let mut results = Vec::new();
        for i in idxs {
            if !self.cw.filter_incident_by_dim(cell_idx)?.1.contains(&i) {
                return Err(MathError::BadCochain);
            };
            let restriction = self.restrictions.get(&(cell_idx, i));
            if restriction.is_none() {
                return Err(MathError::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            if domain_bases.is_some() && self.section_spaces[i].bases.is_some() {
                results = restrict
                    .adjoint(
                        &domain_bases.as_ref().unwrap(),
                        &self.section_spaces[i].bases.as_ref().unwrap(),
                    )?
                    .transform(&cochain[i].0)?;
                continue;
            }
            results = restrict.transpose().transform(&cochain[i].0)?;
        }
        Ok(results)
    }

    pub fn k_minus_1_coboundary_adjoint(
        &self,
        cell_idx: usize,
        data_idx: usize,
    ) -> Result<Vec<(Vec<F>, usize)>, MathError> {
        if cell_idx >= self.cw.cells.len() {
            return Err(MathError::InvalidCellIdx);
        }
        let domain_bases = &self.section_spaces[cell_idx].bases;
        let data = &self.section_spaces[cell_idx].data[data_idx];
        let mut results = Vec::new();
        for i in self.cw.filter_incident_by_dim(cell_idx)?.0 {
            let restriction = self.restrictions.get(&(i, cell_idx));
            if restriction.is_none() {
                return Err(MathError::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            if domain_bases.is_some() && self.section_spaces[i].bases.is_some() {
                results.push((
                    restrict
                        .adjoint(
                            &domain_bases.as_ref().unwrap(),
                            &self.section_spaces[i].bases.as_ref().unwrap(),
                        )?
                        .transform(data)?,
                    i,
                ));
                continue;
            }
            results.push((restrict.transpose().transform(data)?, i));
        }
        Ok(results)
    }

    pub fn k_minus_1_coboundary(
        &self,
        cell_idx: usize,
        cochain: Vec<(Vec<F>, usize)>,
    ) -> Result<Vec<F>, MathError> {
        if cell_idx >= self.cw.cells.len() {
            return Err(MathError::InvalidCellIdx);
        }
        let idxs = cochain.iter().map(|(_, x)| *x).collect::<Vec<_>>();
        let mut results = vec![F::zero()];
        for i in idxs {
            if !self.cw.filter_incident_by_dim(cell_idx)?.0.contains(&i) {
                return Err(MathError::BadCochain);
            };
            let restriction = self.restrictions.get(&(i, cell_idx));
            if restriction.is_none() {
                return Err(MathError::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            results = add_vectors(&results, &restrict.transform(&cochain[i].0)?)?;
        }
        Ok(results)
    }

    pub fn local_up_laplacian(&self, cell_idx: usize) -> Result<Matrix<F>, MathError> {
        let (_, up) = self.cw.filter_incident_by_dim(cell_idx)?;
        let mut matrix = Matrix::new(
            self.section_spaces[cell_idx].dimension,
            self.section_spaces[cell_idx].dimension,
        );
        for i in up {
            let restriction = self.restrictions.get(&(cell_idx, i));
            if restriction.is_none() {
                return Err(MathError::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            if self.section_spaces[cell_idx].bases.is_some()
                && self.section_spaces[i].bases.is_some()
            {
                matrix = matrix.add(
                    restrict
                        .adjoint(
                            &self.section_spaces[cell_idx].bases.as_ref().unwrap(),
                            &self.section_spaces[i].bases.as_ref().unwrap(),
                        )?
                        .multiply(restrict)?,
                )?;
                continue;
            }
            matrix = matrix.add(restrict.transpose().multiply(restrict)?)?;
        }
        Ok(matrix)
    }

    pub fn local_down_laplacian(&self, cell_idx: usize) -> Result<Matrix<F>, MathError> {
        let (down, _) = self.cw.filter_incident_by_dim(cell_idx)?;
        let mut matrix = Matrix::new(
            self.section_spaces[cell_idx].dimension,
            self.section_spaces[cell_idx].dimension,
        );
        for i in down {
            let restriction = self.restrictions.get(&(i, cell_idx));
            if restriction.is_none() {
                return Err(MathError::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            if self.section_spaces[cell_idx].bases.is_some()
                && self.section_spaces[i].bases.is_some()
            {
                matrix = matrix.add(restrict.multiply(&restrict.adjoint(
                    &self.section_spaces[cell_idx].bases.as_ref().unwrap(),
                    &self.section_spaces[i].bases.as_ref().unwrap(),
                )?)?)?;
                continue;
            }
            matrix = matrix.add(restrict.multiply(&restrict.transpose())?)?;
        }
        Ok(matrix)
    }

    pub fn local_laplacian(&self, cell_idx: usize) -> Result<Matrix<F>, MathError> {
        Ok(self
            .local_up_laplacian(cell_idx)?
            .add(self.local_down_laplacian(cell_idx)?)?)
    }

    pub fn k_laplacian(&self, k: usize) -> Result<Matrix<F>, MathError> {
        let mut valid = Vec::new();
        self.cw.cells.iter().enumerate().for_each(|(i, x)| {
            if x.cell.dimension() == k {
                valid.push(i)
            }
        });
        if valid.is_empty() {
            return Err(MathError::NoCellsofDimensionK);
        }
        let mut global = Matrix::new(
            self.section_spaces[valid[0]].dimension,
            self.section_spaces[valid[0]].dimension,
        );
        for i in valid {
            if self.section_spaces[i].dimension != global.dimensions().0 {
                return Err(MathError::DimensionMismatch);
            }
            global = global.add(self.local_laplacian(i)?)?;
        }
        Ok(global)
    }

    pub fn check_glue(
        &mut self,
        start_cell: usize,
        final_cell: usize,
        data_idx: usize,
    ) -> Result<bool, MathError> {
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
        if restrict
            .transform(&self.section_spaces[start_cell].data[data_idx])
            .unwrap()
            != self.section_spaces[final_cell].data[data_idx]
        {
            return Ok(false);
        }
        Ok(true)
    }
}
