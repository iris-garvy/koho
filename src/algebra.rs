use std::ops::{Add, Mul, Neg};

use crate::error::MathError;

pub trait Field:
    Add<Output = Self> + Mul<Output = Self> + Neg<Output = Self> + Copy + PartialEq + Sized
{
    fn zero() -> Self;
    fn one() -> Self;
    fn inv(&self) -> Self;
    fn conj(&self) -> Self;
}

pub struct Matrix<F: Field> {
    data: Vec<Vec<F>>,
    rows: usize,
    cols: usize,
}

impl<F: Field> Matrix<F> {
    pub fn isafield() -> Result<(), MathError> {
        if F::one().inv() + F::one() != F::zero()
            || F::one() + F::zero() != F::one()
            || F::zero() + F::zero() != F::zero()
            || F::one() * F::one() != F::one()
            || F::one() * F::zero() != F::zero()
        {
            return Err(MathError::BadField);
        }
        Ok(())
    }

    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![F::zero(); cols]; rows],
            rows,
            cols,
        }
    }

    pub fn from_vec(mut data: Vec<Vec<F>>) -> Result<Self, MathError> {
        if data.is_empty() {
            return Err(MathError::NullMatrix);
        }

        let rows = data.len();
        let mut cols = 0;
        data.iter().for_each(|x| {
            if x.len() > cols {
                cols = x.len()
            }
        });
        data.iter_mut().for_each(|x| {
            while cols as i64 - x.len() as i64 > 0 {
                x.push(F::zero());
            }
        });

        for row in &data {
            if row.len() != cols {
                return Err(MathError::DimensionMismatch);
            }
        }

        Ok(Self { data, rows, cols })
    }

    pub fn identity(size: usize) -> Self {
        let mut matrix = Self::new(size, size);
        for i in 0..size {
            matrix.data[i][i] = F::one();
        }
        matrix
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j].conj();
            }
        }
        result
    }

    pub fn adjoint(
        &self,
        domain_bases: &[Vec<F>],
        codomain_bases: &[Vec<F>],
    ) -> Result<Self, MathError> {
        if domain_bases.len() != self.cols || codomain_bases.len() != self.rows {
            return Err(MathError::ImproperBases);
        }
        let mut domain = Matrix::new(self.cols, self.cols);
        let mut codomain = Matrix::new(self.rows, self.rows);
        for i in 0..self.rows {
            for j in 0..self.rows {
                codomain.data[i][j] =
                    inner_product(&codomain_bases[i], &codomain_bases[j]).unwrap();
            }
        }
        for i in 0..self.cols {
            for j in 0..self.cols {
                domain.data[i][j] = inner_product(&domain_bases[i], &domain_bases[j]).unwrap();
            }
        }
        let adjoint = domain
            .multiply(&self.transpose().multiply(&codomain).unwrap())
            .unwrap();
        Ok(adjoint)
    }

    pub fn multiply(&self, other: &Matrix<F>) -> Result<Self, MathError> {
        if other.rows != self.cols {
            return Err(MathError::DimensionMismatch);
        }
        let mut result = Self::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = F::zero();
                for k in 0..self.cols {
                    sum = sum + (self.data[i][k] * other.data[k][j]);
                }
                result.data[i][j] = sum;
            }
        }
        Ok(result)
    }

    pub fn add(&self, other: Matrix<F>) -> Result<Self, MathError> {
        if other.rows != self.rows || other.cols != self.cols {
            return Err(MathError::DimensionMismatch);
        }
        let mut new = Matrix::new(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                new.data[i][j] = self.data[i][j] + other.data[i][j]
            }
        }
        Ok(new)
    }

    pub fn transform(&self, vec: &[F]) -> Result<Vec<F>, MathError> {
        if vec.len() != self.cols {
            return Err(MathError::DimensionMismatch);
        }
        let mut new = vec![F::zero(); self.rows];
        for i in 0..self.rows {
            let mut sum = F::zero();
            for j in 0..self.cols {
                sum = self.data[i][j] * vec[j];
            }
            new[i] = sum;
        }
        Ok(new)
    }
}

pub fn add_vectors<F: Field>(a: &[F], b: &[F]) -> Result<Vec<F>, MathError>
where
    F: std::ops::Add<Output = F> + Copy,
{
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch);
    }

    Ok(a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect())
}

pub fn inner_product<F: Field>(a: &[F], b: &[F]) -> Result<F, MathError> {
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch);
    }
    let mut output = F::zero();
    let together = a.iter().zip(b);
    for (a, b) in together {
        output = output + *a * *b;
    }
    Ok(output)
}
