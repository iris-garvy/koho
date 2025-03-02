use std::{cell, collections::HashSet, hash::Hash};

use crate::{error::Error, sheaf::OpenSet};

pub trait KCell<T: Eq + Hash + Clone, O: OpenSet> {
    fn points(&self) -> HashSet<&O::Point>;
    fn dimension(&self) -> usize;
    fn boundary(&mut self) -> HashSet<O::Point>;
    fn attach(&self, point: &O::Point, skeleton: &mut Skeleton<T, O>) -> O::Point;
    fn remove(&mut self, set: HashSet<O::Point>) -> bool;
}

pub struct Cell<T: Eq + Hash + Clone, O: OpenSet> {
    pub cell: Box<dyn KCell<T, O>>,
    pub incidents: Vec<usize>,
}

impl<T: Eq + Hash + Clone, O: OpenSet> Cell<T, O> {
    pub fn new(cell: Box<dyn KCell<T, O>>) -> Self {
        Self {
            cell,
            incidents: Vec::new(),
        }
    }
}

pub struct Skeleton<T: Eq + Hash + Clone, O: OpenSet> {
    pub dimension: usize,
    pub cells: Vec<Cell<T, O>>,
}

impl<T: Eq + Hash + Clone, O: OpenSet> Skeleton<T, O> {
    pub fn init() -> Self {
        Self {
            dimension: 0,
            cells: Vec::new(),
        }
    }
    pub fn attach(&mut self, cell: Box<dyn KCell<T, O>>) -> Result<(), Error> {
        let incoming_dim = cell.dimension() as i64;
        if incoming_dim - self.dimension as i64 > 1 {
            return Err(Error::DimensionMismatch);
        }
        if self.dimension == 0 && incoming_dim == 1 && self.cells.len() == 0 {
            return Err(Error::CWUninitialized);
        }
        let mut cell = Cell::new(cell);
        let mut boundary = cell.cell.boundary();
        for p in cell.cell.points() {
            let point = cell.cell.attach(p, self);
            let mut truth = false;
            self.cells.iter().enumerate().for_each(|(i, x)| {
                if x.cell.points().contains(&point) {
                    truth = true;
                    cell.incidents.push(i);
                }
            });
            if truth {
                boundary.insert(point);
            }
        }
        cell.cell.remove(boundary);
        self.cells.push(cell);
        if incoming_dim - self.dimension as i64 == 1 {
            self.dimension += 1
        }
        Ok(())
    }

    pub fn fetch_cell_by_point(&self, point: O::Point) -> Result<(&Cell<T, O>, usize), Error> {
        for i in 0..self.cells.len() {
            if self.cells[i].cell.points().contains(&point) {
                return Ok((&self.cells[i], i));
            };
        }
        Err(Error::NoPointFound)
    }

    pub fn incident_cells(&self, cell_idx: usize) -> Result<&Vec<usize>, Error> {
        if cell_idx >= self.cells.len() {
            return Err(Error::InvalidCellIdx);
        }
        Ok(&self.cells[cell_idx].incidents)
    }

    pub fn filter_incident_by_dim(
        &self,
        cell_idx: usize,
    ) -> Result<(Vec<usize>, Vec<usize>), Error> {
        if cell_idx >= self.cells.len() {
            return Err(Error::InvalidCellIdx);
        }
        let incidents = &self.cells[cell_idx].incidents;
        let dim = self.cells[cell_idx].cell.dimension();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        for i in incidents {
            if dim - self.cells[*i].cell.dimension() == 1 {
                lower.push(*i);
            } else if self.cells[*i].cell.dimension() - dim == 1 {
                upper.push(*i);
            }
        }
        Ok((lower, upper))
    }
}
