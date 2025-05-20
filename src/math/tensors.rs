use candle_core::{DType, Device, Error, Result, Tensor, WithDType};

#[derive(Debug, Clone)]
pub struct Vector {
    tensor: Tensor,
    device: Device,
    dtype: DType,
}

impl Vector {
    pub fn new(tensor: Tensor, device: Device, dtype: DType) -> Result<Self> {
        if tensor.rank() != 1 {
            return Err(Error::Msg("Vector must be rank 1".into()));
        }
        Ok(Self {
            tensor,
            device,
            dtype,
        })
    }

    pub fn from_slice<T: WithDType>(
        data: &[T],
        dimension: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let t = Tensor::from_slice(data, &[dimension], &device)?;
        Self::new(t, device, dtype)
    }

    pub fn dimension(&self) -> usize {
        self.tensor.dims()[0]
    }

    pub fn inner(&self) -> &Tensor {
        &self.tensor
    }

    pub fn dot(&self, other: &Vector) -> Result<Tensor> {
        self.tensor.transpose(0, 1)?.matmul(&other.tensor)
    }

    pub fn norm(&self) -> Result<Tensor> {
        self.tensor.sqr()?.sum_all()?.sqrt()
    }

    pub fn normalize(&self) -> Result<Self> {
        let norm = self.norm()?;
        Ok(Self {
            tensor: self.tensor.broadcast_div(&norm)?,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    pub fn add(&self, other: &Vector) -> Result<Self> {
        Ok(Self {
            tensor: self.tensor.add(&other.tensor)?,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    pub fn scale<T: WithDType>(&self, scalar: T) -> Result<Self> {
        if scalar.to_scalar().dtype() == self.dtype {
            let tensor = self.tensor.clone().to_dtype(DType::F64)?;
            return Ok(Self {
                tensor: (tensor * scalar.to_scalar().to_f64())?,
                device: self.device.clone(),
                dtype: self.dtype,
            });
        }
        Err(Error::DTypeMismatchBinaryOp {
            lhs: scalar.to_scalar().dtype(),
            rhs: self.dtype,
            op: "scalar multiply",
        })
    }
}

#[derive(Debug, Clone)]
pub struct Matrix {
    pub tensor: Tensor,
    pub device: Device,
    pub dtype: DType,
}

impl Matrix {
    /// Creates a new `Matrix`.
    pub fn new(tensor: Tensor, device: Device, dtype: DType) -> Result<Self> {
        if tensor.rank() != 2 {
            return Err(Error::Msg("Matrix must be rank 2".into()));
        }
        Ok(Self {
            tensor,
            device,
            dtype,
        })
    }

    /// Creates a new `Matrix` from a slice of data.
    pub fn from_slice<T: WithDType>(
        data: &[T],
        rows: usize,
        cols: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let t = Tensor::from_slice(data, (rows, cols), &device)?;
        Self::new(t, device, dtype)
    }

    pub fn from_vecs(vecs: Vec<Vector>) -> Result<Self> {
        if vecs.is_empty() {
            return Err(Error::Msg(
                "Cannot create a matrix from an empty list of vectors.".into(),
            ));
        }

        let first_vec = &vecs[0];
        let dimension = first_vec.dimension();
        let device = first_vec.device.clone();
        let dtype = first_vec.dtype;

        let mut column_tensors = Vec::with_capacity(vecs.len());

        for (i, vec) in vecs.iter().enumerate() {
            if vec.dimension() != dimension {
                return Err(Error::Msg(format!(
                    "Vector at index {} has dimension {} but expected {}.",
                    i,
                    vec.dimension(),
                    dimension
                )));
            }
            if vec.dtype != dtype {
                return Err(Error::Msg(format!(
                    "Vector at index {} has a different dtype. Expected {:?}, got {:?}.",
                    i, dtype, vec.dtype
                )));
            }

            column_tensors.push(vec.inner().reshape((dimension, 1))?);
        }

        // Concatenate all column tensors along dimension 1 (columns)
        let matrix_tensor = Tensor::cat(&column_tensors, 1)?;

        Self::new(matrix_tensor, device, dtype)
    }

    /// Returns the shape of the matrix as `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        let dims = self.tensor.dims();
        (dims[0], dims[1])
    }

    /// Returns the number of rows in the matrix.
    pub fn rows(&self) -> usize {
        self.tensor.dims()[0]
    }

    /// Returns the number of columns in the matrix.
    pub fn cols(&self) -> usize {
        self.tensor.dims()[1]
    }

    /// Returns a reference to the inner `Tensor`.
    pub fn inner(&self) -> &Tensor {
        &self.tensor
    }

    /// Performs matrix multiplication: `self * other`.
    pub fn matmul(&self, other: &Matrix) -> Result<Self> {
        if self.cols() != other.rows() {
            return Err(Error::Msg(format!(
                "Matrix multiplication dimension mismatch: self_cols ({}) != other_rows ({})",
                self.cols(),
                other.rows()
            )));
        }
        let result_tensor = self.tensor.matmul(other.inner())?;
        Ok(Self {
            tensor: result_tensor,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    pub fn matvec(&self, other: &Vector) -> Result<Vector> {
        if self.cols() != other.dimension() {
            return Err(Error::Msg(format!(
                "Matrix multiplication dimension mismatch: self_cols ({}) != other_rows ({})",
                self.cols(),
                other.dimension()
            )));
        }
        let result_tensor = self.tensor.matmul(&other.inner().reshape((other.dimension(),1)).unwrap())?.reshape(other.dimension())?;
        Ok(Vector {
            tensor: result_tensor,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    /// Transposes the matrix.
    pub fn transpose(&self) -> Result<Self> {
        let result_tensor = self.tensor.transpose(0, 1)?;
        Ok(Self {
            tensor: result_tensor,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    /// Adds another matrix to this matrix element-wise.
    pub fn add(&self, other: &Matrix) -> Result<Self> {
        if self.shape() != other.shape() {
            return Err(Error::Msg(format!(
                "Matrix addition shape mismatch: self {:?} != other {:?}",
                self.shape(),
                other.shape()
            )));
        }
        let result_tensor = self.tensor.add(other.inner())?;
        Ok(Self {
            tensor: result_tensor,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    /// Scales the matrix by a scalar.
    pub fn scale<T: WithDType>(&self, scalar: T) -> Result<Self> {
        let tensor = self.tensor.clone().to_dtype(DType::F64)?;
        Ok(Self {
            tensor: (tensor * scalar.to_scalar().to_f64())?,
            device: self.device.clone(),
            dtype: self.dtype,
        })
    }

    /// Computes the Frobenius norm of the matrix.
    /// The Frobenius norm is sqrt(sum of squares of its elements).
    pub fn frobenius_norm(&self) -> Result<Tensor> {
        self.tensor.sqr()?.sum_all()?.sqrt()
    }

    pub fn to_vectors(&self) -> Result<Vec<Vector>> {
        let (_, cols) = self.shape();
        let mut cols_vectors = Vec::with_capacity(cols);

        // Candle's `chunk` method can split a tensor along a dimension.
        // Splitting along dim 0 (rows) into `rows` chunks will give each row.
        let cols_tensors = self.tensor.chunk(cols, 1)?;

        for col_tensor in cols_tensors {
            // Each chunk will be a tensor of shape (1, cols). Reshape to (cols,) for Vector.
            let reshaped_col = col_tensor.squeeze(1)?; // Squeeze out the 1 dimension
            cols_vectors.push(Vector::new(reshaped_col, self.device.clone(), self.dtype)?);
        }

        Ok(cols_vectors)
    }

    /// Generates a new random matrix with elements sampled from a standard normal distribution (mean 0, std dev 1).
    pub fn rand(rows: usize, cols: usize, device: Device, dtype: DType) -> Result<Self> {
        let tensor = Tensor::randn(0.0f32, 1.0f32, (rows, cols), &device)?.to_dtype(dtype)?;
        Self::new(tensor, device, dtype)
    }
}

// Example Usage (requires a candle_core setup)
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_matrix_new_and_shape() -> Result<()> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let t = Tensor::randn(0f32, 1f32, (2, 3), &device)?.to_dtype(dtype)?;
        let m = Matrix::new(t, device.clone(), dtype)?;

        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(m.inner().dims(), &[2, 3]);
        assert_eq!(m.dtype, dtype);
        Ok(())
    }

    #[test]
    fn test_matrix_from_slice() -> Result<()> {
        let device = Device::Cpu;
        let data_f32: [f32; 6] = [1., 2., 3., 4., 5., 6.];

        let m = Matrix::from_slice(&data_f32, 2, 3, device.clone(), DType::F32)?;
        assert_eq!(m.shape(), (2, 3));
        assert_eq!(
            m.inner().to_vec2::<f32>()?,
            vec![vec![1., 2., 3.], vec![4., 5., 6.]]
        );
        assert_eq!(m.dtype, DType::F32); // This is the struct's dtype field
        assert_eq!(m.inner().dtype(), DType::F32); // Tensor's actual dtype
        Ok(())
    }

    #[test]
    fn test_matrix_add() -> Result<()> {
        let device = Device::Cpu;
        let m1 = Matrix::from_slice(&[1f32, 2., 3., 4.], 2, 2, device.clone(), DType::F32)?;
        let m2 = Matrix::from_slice(&[5f32, 6., 7., 8.], 2, 2, device.clone(), DType::F32)?;
        let m3 = m1.add(&m2)?;
        assert_eq!(
            m3.inner().to_vec2::<f32>()?,
            vec![vec![6., 8.], vec![10., 12.]]
        );
        assert_eq!(m3.dtype, DType::F32);
        Ok(())
    }

    #[test]
    fn test_matrix_matmul() -> Result<()> {
        let device = Device::Cpu;
        let m1 = Matrix::from_slice(&[1f32, 2., 3., 4.], 2, 2, device.clone(), DType::F32)?; // 2x2
        let m2 = Matrix::from_slice(
            &[5f32, 6., 7., 8., 9., 10.],
            2,
            3,
            device.clone(),
            DType::F32,
        )?; // 2x3
        let m3 = m1.matmul(&m2)?; // Expected 2x3

        assert_eq!(m3.shape(), (2, 3));
        assert_eq!(
            m3.inner().to_vec2::<f32>()?,
            vec![vec![21., 24., 27.], vec![47., 54., 61.]]
        );
        assert_eq!(m3.dtype, DType::F32);
        Ok(())
    }

    #[test]
    fn test_matrix_transpose() -> Result<()> {
        let device = Device::Cpu;
        let m1 = Matrix::from_slice(
            &[1f32, 2., 3., 4., 5., 6.],
            2,
            3,
            device.clone(),
            DType::F32,
        )?;
        let m_t = m1.transpose()?;
        assert_eq!(m_t.shape(), (3, 2));
        assert_eq!(
            m_t.inner().to_vec2::<f32>()?,
            vec![vec![1., 4.], vec![2., 5.], vec![3., 6.]]
        );
        Ok(())
    }

    impl Matrix {
        // A more conventional scale method for numeric scalars
        pub fn scale_numeric(&self, scalar_val: f64) -> Result<Self> {
            let result_tensor = (self.tensor.clone() * scalar_val)?;
            Ok(Self {
                tensor: result_tensor,
                device: self.device.clone(),
                dtype: self.dtype,
            })
        }
    }

    #[test]
    fn test_matrix_scale_numeric() -> Result<()> {
        let device = Device::Cpu;
        let m1 = Matrix::from_slice(&[1f32, 2., 3., 4.], 2, 2, device.clone(), DType::F32)?;
        let m_scaled = m1.scale_numeric(2.0)?;
        assert_eq!(
            m_scaled.inner().to_vec2::<f32>()?,
            vec![vec![2., 4.], vec![6., 8.]]
        );
        Ok(())
    }

    // To test the original `scale` method, you'd need a type `T` that satisfies:
    #[test]
    fn test_frobenius_norm() -> Result<()> {
        let device = Device::Cpu;
        let m = Matrix::from_slice(&[3f32, -4., 12.], 1, 3, device.clone(), DType::F32)?; // A row vector as a 1x3 matrix
                                                                                          // Norm = sqrt(3^2 + (-4)^2 + 12^2) = sqrt(9 + 16 + 144) = sqrt(169) = 13
        let norm_tensor = m.frobenius_norm()?;
        let norm_val = norm_tensor.to_scalar::<f32>()?;
        assert!((norm_val - 13.0).abs() < 1e-6);
        Ok(())
    }
}
