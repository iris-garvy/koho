use std::ops::{Add, Mul, Neg, Sub};

use crate::error::MathError;

pub trait Field:
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Neg<Output = Self> + Copy + PartialEq
{
    fn zero() -> Self;
    fn one() -> Self;
    fn inv(&self) -> Self;
    fn conj(&self) -> Self;
}

#[derive(Clone, PartialEq, Debug)]
pub struct Matrix<F: Field> {
    pub data: Vec<Vec<F>>,
    pub rows: usize,
    pub cols: usize,
}

impl<F: Field> Matrix<F> {
    pub fn isafield() -> Result<(), MathError> {
        if F::one().inv() * F::one() != F::one()
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
        let adjoint = domain.inverse()?.transpose()
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
                sum = sum + self.data[i][j] * vec[j];
            }
            new[i] = sum;
        }
        Ok(new)
    }

    pub fn inverse(&self) -> Result<Self, MathError> {
        if self.rows != self.cols {
            return Err(MathError::DimensionMismatch);
        }
    
        let n = self.rows;
        let mut augmented = Self::new(n, 2 * n);
        
        for i in 0..n {
            for j in 0..n {
                augmented.data[i][j] = self.data[i][j];
            }
        }
        for i in 0..n {
            augmented.data[i][i + n] = F::one();
        }
        for i in 0..n {
            let mut pivot_row = i;
            if augmented.data[i][i] == F::zero() {
                let mut found_pivot = false;
                for k in (i + 1)..n {
                    if augmented.data[k][i] != F::zero() {
                        pivot_row = k;
                        found_pivot = true;
                        break;
                    }
                }
                if !found_pivot {
                    return Err(MathError::NotInvertible);
                }
                if pivot_row != i {
                    for j in 0..(2 * n) {
                        let temp = augmented.data[i][j];
                        augmented.data[i][j] = augmented.data[pivot_row][j];
                        augmented.data[pivot_row][j] = temp;
                    }
                }
            }

            let pivot = augmented.data[i][i];
            let pivot_inv = pivot.inv();
            
            for j in 0..(2 * n) {
                augmented.data[i][j] = augmented.data[i][j] * pivot_inv;
            }
            
            for k in 0..n {
                if k != i {
                    let factor = augmented.data[k][i];
                    for j in 0..(2 * n) {
                        augmented.data[k][j] = augmented.data[k][j] - (factor * augmented.data[i][j]);
                    }
                }
            }
        }
        
        let mut result = Self::new(n, n);
        for i in 0..n {
            for j in 0..n {
                result.data[i][j] = augmented.data[i][j + n];
            }
        }
        
        Ok(result)
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

pub mod numerical {
    use super::Field;
    use num_complex::Complex;

    impl Field for f64 {
        fn zero() -> Self {
            0.0
        }
    
        fn one() -> Self {
            1.0
        }
    
        fn inv(&self) -> Self {
            1.0 / self
        }
    
        fn conj(&self) -> Self {
            *self
        }
    }

    impl Field for f32 {
        fn zero() -> Self {
            0.0
        }
    
        fn one() -> Self {
            1.0
        }
    
        fn inv(&self) -> Self {
            1.0 / self
        }
    
        fn conj(&self) -> Self {
            *self
        }
    }

    impl Field for Complex<f64> {
        fn zero() -> Self {
            Complex::new(0.0, 0.0)
        }
    
        fn one() -> Self {
            Complex::new(1.0, 0.0)
        }
    
        fn inv(&self) -> Self {
            self.inv()
        }
    
        fn conj(&self) -> Self {
            self.conj()
        }
    }

    impl Field for Complex<f32> {
        fn zero() -> Self {
            Complex::new(0.0, 0.0)
        }
    
        fn one() -> Self {
            Complex::new(1.0, 0.0)
        }
    
        fn inv(&self) -> Self {
            self.inv()
        }
    
        fn conj(&self) -> Self {
            self.conj()
        }
    }
}