use std::collections::{HashMap, HashSet};

use crate::{
    algebra::{Field, Matrix},
    cw::{KCell, Skeleton},
    error::Error,
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
}

impl<F: Field> Sections<F> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            bases: None,
        }
    }
    pub fn add_bases(&mut self, bases: Vec<Vec<F>>) {
        self.bases = Some(bases)
    }
}
pub struct CellularSheaf<F: Field, T: Eq + std::hash::Hash + Clone, O: OpenSet> {
    pub cw: Skeleton<T, O>,
    pub data: Vec<Sections<F>>,
    pub restrictions: HashMap<(usize, usize), Matrix<F>>,
    pub global_sections: Vec<(Sections<F>)>,
}

impl<F: Field, T: Eq + std::hash::Hash + Clone, O: OpenSet> CellularSheaf<F, T, O> {
    pub fn init() -> Self {
        Self {
            cw: Skeleton::init(),
            data: Vec::new(),
            restrictions: HashMap::new(),
            global_sections: Vec::new(),
        }
    }

    pub fn attach(
        &mut self,
        cell: Box<dyn KCell<T, O>>,
        data: Option<Sections<F>>,
    ) -> Result<(), Error> {
        self.data.push(if data.is_some() {
            data.unwrap()
        } else {
            Sections::new()
        });
        self.cw.attach(cell)?;
        Ok(())
    }

    pub fn update(&mut self, cell_idx: usize, data_idx: usize, val: Vec<F>) -> Result<(), Error> {
        if cell_idx >= self.data.len() {
            return Err(Error::InvalidCellIdx);
        }
        if data_idx >= self.data[cell_idx].data.len() {
            return Err(Error::InvalidDataIdx);
        }
        self.data[cell_idx].data[data_idx] = val;
        Ok(())
    }

    pub fn new_data(&mut self, cell_idx: usize, val: Vec<F>) -> Result<(), Error> {
        if cell_idx >= self.data.len() {
            return Err(Error::InvalidCellIdx);
        }
        self.data[cell_idx].data.push(val);
        Ok(())
    }

    pub fn set_restriction(
        &mut self,
        start_cell: usize,
        final_cell: usize,
        map: Matrix<F>,
    ) -> Result<(), Error> {
        if start_cell >= self.cw.cells.len() || final_cell >= self.cw.cells.len() {
            return Err(Error::InvalidCellIdx);
        }
        self.restrictions.insert((start_cell, final_cell), map);
        Ok(())
    }

    pub fn k_coboundary(
        &mut self,
        cell_idx: usize,
        data_idx: usize,
    ) -> Result<Vec<(Vec<F>, usize)>, Error> {
        if cell_idx >= self.cw.cells.len() {
            return Err(Error::InvalidCellIdx);
        }
        if data_idx >= self.data[cell_idx].data.len() {
            return Err(Error::InvalidDataIdx);
        }
        let mut results = Vec::new();
        for i in self.cw.filter_incident_by_dim(cell_idx)?.1 {
            let restriction = self.restrictions.get(&(cell_idx, i));
            if restriction.is_none() {
                return Err(Error::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            results.push((
                restrict
                    .transform(&self.data[cell_idx].data[data_idx])
                    .unwrap(),
                i,
            ));
        }
        Ok(results)
    }

    pub fn k_coboundary_adjoint(
        &mut self,
        cell_idx: usize,
        cochain: Vec<(Vec<F>, usize)>,
    ) -> Result<Vec<F>, Error> {
        if cell_idx >= self.cw.cells.len() {
            return Err(Error::InvalidCellIdx);
        }
        let idxs = cochain.iter().map(|(_, x)| *x).collect::<Vec<_>>();
        let domain_bases = &self.data[cell_idx].bases;
        let mut results = Vec::new();
        for i in idxs {
            if !self.cw.filter_incident_by_dim(cell_idx)?.1.contains(&i) {
                return Err(Error::BadCochain);
            };
            let restriction = self.restrictions.get(&(cell_idx, i));
            if restriction.is_none() {
                return Err(Error::NoRestrictionDefined);
            }
            let restrict = restriction.unwrap();
            if domain_bases.is_some() && self.data[i].bases.is_some() {
                results = restrict
                    .adjoint(
                        &domain_bases.as_ref().unwrap(),
                        &self.data[i].bases.as_ref().unwrap(),
                    )?
                    .transform(&cochain[i].0)?;
                continue;
            }
            results = restrict.transpose().transform(&cochain[i].0)?;
        }
        Ok(results)
    }

    pub fn check_glue(
        &mut self,
        start_cell: usize,
        final_cell: usize,
        data_idx: usize,
    ) -> Result<bool, Error> {
        if start_cell >= self.cw.cells.len() || final_cell >= self.cw.cells.len() {
            return Err(Error::InvalidCellIdx);
        }
        if self.cw.cells[start_cell].cell.dimension() <= self.cw.cells[final_cell].cell.dimension()
        {
            return Err(Error::DimensionMismatch);
        }
        let restriction = self.restrictions.get(&(start_cell, final_cell));
        if restriction.is_none() {
            return Err(Error::NoRestrictionDefined);
        }
        let restrict = restriction.unwrap();
        if restrict
            .transform(&self.data[start_cell].data[data_idx])
            .unwrap()
            != self.data[final_cell].data[data_idx]
        {
            return Ok(false);
        }
        Ok(true)
    }
}
