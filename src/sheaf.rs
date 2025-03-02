use std::{collections::{HashMap, HashSet}, ops::{Add, Mul, Neg}};

use crate::{cw::{Cell, KCell, Skeleton}, error::Error};

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Point<T: Eq + std::hash::Hash + Clone + Sized>(T);

pub trait Field:
    Add<Output = Self> +
    Mul<Output = Self> +
    Neg<Output = Self> + 
    Copy + 
    PartialEq + 
    Sized
{
    fn zero() -> Self;
    fn one() -> Self;
    fn inv(&self) -> Option<Self>;
}

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

pub struct Sections<F: Field>(Vec<Vec<F>>);

impl<F: Field> Sections<F> {
    pub fn new() -> Self {
        Self(Vec::new())
    }
}
pub struct CellularSheaf<F: Field, T: Eq + std::hash::Hash + Clone, O: OpenSet> {
    pub cw: Skeleton<T, O>,
    pub data: Vec<Sections<F>>,
    pub restrictions: HashMap<(usize, usize), Box<dyn Fn(&Vec<F>) -> Vec<F>>>,
    pub global_sections: Vec<(Sections<F>)>
}

impl<F: Field, T: Eq + std::hash::Hash + Clone, O: OpenSet>  CellularSheaf<F, T, O>{
    pub fn init() -> Self {
        Self {
            cw: Skeleton::init(),
            data: Vec::new(),
            restrictions: HashMap::new(),
            global_sections: Vec::new()
        }
    }

    pub fn attach(&mut self, cell: Box<dyn KCell<T,O>>, data: Option<Sections<F>>) -> Result<(), Error> {
        self.data.push(if data.is_some() { data.unwrap() } else { Sections::new() });
        self.cw.attach(cell)?;
        Ok(())
    }

    pub fn update(&mut self, cell_idx: usize, data_idx: usize, val: Vec<F>) -> Result<(), Error> {
        if cell_idx >= self.data.len() {
            return Err(Error::InvalidCellIdx)
        }
        if data_idx >= self.data[cell_idx].0.len() {
            return Err(Error::InvalidDataIdx)
        }
        self.data[cell_idx].0[data_idx] = val;
        Ok(())
    }

    pub fn new_data(&mut self, cell_idx: usize, val: Vec<F>) -> Result<(), Error> {
        if cell_idx >= self.data.len() {
            return Err(Error::InvalidCellIdx)
        }
        self.data[cell_idx].0.push(val);
        Ok(())
    }
     
    pub fn set_restriction(&mut self, start_cell: usize, final_cell: usize, map: Box<dyn Fn(&Vec<F>) -> Vec<F>>) -> Result<(), Error> {
        if start_cell >= self.cw.cells.len() || final_cell >= self.cw.cells.len() {
            return Err(Error::InvalidCellIdx)
        }
        if self.cw.cells[start_cell].cell.dimension() <= self.cw.cells[final_cell].cell.dimension() {
            return Err(Error::DimensionMismatch)
        }
        self.restrictions.insert((start_cell, final_cell), map);
        Ok(())
    }

    pub fn k_coboundary(&mut self, cell_idx: usize, data_idx: usize) -> Result<(Vec<(Vec<F>, usize)>, usize), Error> {
        if cell_idx >= self.cw.cells.len() {
            return Err(Error::InvalidCellIdx)
        }
        if data_idx >= self.data[cell_idx].0.len() {
            return Err(Error::InvalidDataIdx);
        }
        let mut results = Vec::new();
        for i in self.cw.filter_incident_by_dim(cell_idx)? {
            let restriction = self.restrictions.get(&(cell_idx, i));
            if restriction.is_none() {
                return Err(Error::NoRestrictionDefined)
            }
            let restrict = restriction.unwrap();
            results.push((restrict(&self.data[cell_idx].0[data_idx]), i));
        }
        Ok((results, data_idx))
    }

    pub fn check_glue(&mut self, start_cell: usize, final_cell: usize, data_idx: usize) -> Result<bool, Error> {
        if start_cell >= self.cw.cells.len() || final_cell >= self.cw.cells.len() {
            return Err(Error::InvalidCellIdx)
        }
        if self.cw.cells[start_cell].cell.dimension() <= self.cw.cells[final_cell].cell.dimension() {
            return Err(Error::DimensionMismatch)
        }
        let restriction = self.restrictions.get(&(start_cell, final_cell));
        if restriction.is_none() {
            return Err(Error::NoRestrictionDefined)
        }
        let restrict = restriction.unwrap();
        if restrict(&self.data[start_cell].0[data_idx]) != self.data[final_cell].0[data_idx] {
            return Ok(false)
        }
        Ok(true)
    }
}