pub mod algebra;
pub mod cw;
pub mod error;
pub mod sheaf;

mod prelude {
    use crate::algebra::{add_vectors, inner_product, numerical, Matrix};
    use crate::cw::{KCell, Skeleton};
    use crate::error::MathError;
    use crate::sheaf::CellularSheaf;
}

#[cfg(test)]
mod algebra_tests {
    use std::process::id;

    use num_complex::Complex;
    use rand::Rng;

    use crate::algebra::{add_vectors, inner_product, Matrix};
    #[test]
    fn test_isafield() {
        Matrix::<f64>::isafield().unwrap();
        Matrix::<Complex<f64>>::isafield().unwrap();
        Matrix::<f32>::isafield().unwrap();
        Matrix::<Complex<f32>>::isafield().unwrap();
    }

    #[test]
    fn test_add_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        assert_eq!(
            add_vectors(&a, &b).unwrap(),
            vec![3.0, 6.0, 9.0, 12.0, 15.0]
        )
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        assert_eq!(inner_product(&a, &b).unwrap(), 110.0)
    }

    #[test]
    fn test_matrix_transform() {
        let identity = Matrix::<f64>::identity(8);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0];
        assert_eq!(identity.transform(&a).unwrap(), a)
    }

    #[test]
    fn test_matrix_identity_and_null() {
        let identity = Matrix::<f64>::identity(10);
        let null = Matrix::<f64>::new(10, 10);
        let mut rng = rand::rng();
        let matrix = Matrix::from_vec(vec![vec![rng.random::<f64>(); 10]; 10]).unwrap();

        assert_eq!(identity.multiply(&null).unwrap(), null);
        assert_eq!(identity.multiply(&matrix).unwrap(), matrix.clone());
        assert_eq!(identity.add(null.clone()).unwrap(), identity);
        assert_eq!(&null.add(matrix.clone()).unwrap(), &matrix);
        assert_eq!(null.multiply(&matrix).unwrap(), null);
    }

    #[test]
    fn test_matrix_transpose() {
        let mut rng = rand::rng();
        let matrix = Matrix::from_vec(vec![vec![rng.random::<f64>(); 8]; 10]).unwrap();
        let mt = matrix.transpose();
        let identity = Matrix::<f64>::identity(10);

        assert_eq!(identity.transpose(), identity);
        for i in 0..10 {
            for j in 0..8 {
                assert_eq!(mt.data[j][i], matrix.data[i][j]);
            }
        }
        let test = Matrix::<f64>::new(10, 8);
        assert_eq!(test.transpose().dimensions(), (8, 10));
        assert_eq!(matrix.dimensions(), (10, 8));
        assert_eq!(mt.dimensions(), (8, 10))
    }
}
