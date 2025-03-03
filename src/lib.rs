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
    use crate::error::MathError;

    #[test]
    fn test_isafield() {
        // Test for various number types
        Matrix::<f64>::isafield().unwrap();
        Matrix::<Complex<f64>>::isafield().unwrap();
        Matrix::<f32>::isafield().unwrap();
        Matrix::<Complex<f32>>::isafield().unwrap();

        let a = 2.5f64;
        let b = 3.7f64;
        assert_eq!(a + b, b + a);
        assert_eq!(a * b, b * a);

        let one = 1.0f64;
        let zero = 0.0f64;
        assert_eq!(a * one, a);
        assert_eq!(a + zero, a);
        assert_eq!(a * (one / a), one);
    }

    #[test]
    fn test_add_vectors() {
        // Simple case
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_eq!(
            add_vectors(&a, &b).unwrap(),
            vec![3.0, 6.0, 9.0, 12.0, 15.0]
        );

        // Empty vectors
        let empty: Vec<f64> = vec![];
        assert_eq!(add_vectors(&empty, &empty).unwrap(), empty);

        // Negative numbers
        let c = vec![-1.0, -2.0, -3.0];
        let d = vec![1.0, 2.0, 3.0];
        assert_eq!(add_vectors(&c, &d).unwrap(), vec![0.0, 0.0, 0.0]);

        // Error case: different sizes
        let e = vec![1.0, 2.0];
        let f = vec![3.0, 4.0, 5.0];
        let result = add_vectors(&e, &f);
        assert!(matches!(result, Err(MathError::DimensionMismatch)));

        // Complex numbers
        let g = vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)];
        let h = vec![Complex::new(3.0, 3.0), Complex::new(4.0, 4.0)];
        let expected = vec![Complex::new(4.0, 4.0), Complex::new(6.0, 6.0)];
        assert_eq!(add_vectors(&g, &h).unwrap(), expected);
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert_eq!(inner_product(&a, &b).unwrap(), 110.0);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert_eq!(inner_product(&c, &d).unwrap(), 0.0);

        let e = vec![1.0, 0.0, 0.0];
        assert_eq!(inner_product(&e, &e).unwrap(), 1.0);

        let f = vec![1.0, 2.0];
        let g = vec![3.0, 4.0, 5.0];
        let result = inner_product(&f, &g);
        assert!(matches!(result, Err(MathError::DimensionMismatch)));

        let h = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
        let i = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        assert_eq!(inner_product(&h, &i).unwrap(), Complex::new(1.0, 1.0));
    }

    #[test]
    fn test_matrix_transform() {
        // Identity transform
        let identity = Matrix::<f64>::identity(8);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0];
        assert_eq!(identity.transform(&a).unwrap(), a);

        let mut transform = Matrix::<f64>::new(2, 3);
        transform.data[0][0] = 1.0;
        transform.data[0][1] = 2.0;
        transform.data[0][2] = 3.0;
        transform.data[1][0] = 4.0;
        transform.data[1][1] = 5.0;
        transform.data[1][2] = 6.0;

        let vector = vec![1.0, 2.0, 3.0];
        let result = transform.transform(&vector).unwrap();
        assert_eq!(result, vec![14.0, 32.0]);

        // Error case: dimension mismatch
        let short_vector = vec![1.0, 2.0];
        let transform_result = transform.transform(&short_vector);
        assert!(matches!(
            transform_result,
            Err(MathError::DimensionMismatch)
        ));

        let mut projection = Matrix::<f64>::new(2, 3);
        projection.data[0][0] = 1.0;
        projection.data[0][1] = 0.0;
        projection.data[0][2] = 0.0;
        projection.data[1][0] = 0.0;
        projection.data[1][1] = 1.0;
        projection.data[1][2] = 0.0;

        let vector3d = vec![5.0, 6.0, 7.0];
        let result2d = projection.transform(&vector3d).unwrap();
        assert_eq!(result2d, vec![5.0, 6.0]);
    }

    #[test]
    fn test_matrix_identity_and_null() {
        let identity = Matrix::<f64>::identity(10);
        let null = Matrix::<f64>::new(10, 10);

        let mut fixed_matrix = Matrix::<f64>::new(10, 10);
        for i in 0..10 {
            for j in 0..10 {
                fixed_matrix.data[i][j] = (i * 10 + j) as f64;
            }
        }

        assert_eq!(identity.multiply(&null).unwrap(), null);
        assert_eq!(
            identity.multiply(&fixed_matrix).unwrap(),
            fixed_matrix.clone()
        );
        assert_eq!(identity.add(null.clone()).unwrap(), identity);
        assert_eq!(null.add(fixed_matrix.clone()).unwrap(), fixed_matrix);
        assert_eq!(null.multiply(&fixed_matrix).unwrap(), null);
        assert_eq!(fixed_matrix.multiply(&null).unwrap(), null);
    }

    #[test]
    fn test_matrix_multiply() {
        let mut a = Matrix::<f64>::new(2, 3);
        a.data[0][0] = 1.0;
        a.data[0][1] = 2.0;
        a.data[0][2] = 3.0;
        a.data[1][0] = 4.0;
        a.data[1][1] = 5.0;
        a.data[1][2] = 6.0;

        let mut b = Matrix::<f64>::new(3, 2);
        b.data[0][0] = 7.0;
        b.data[0][1] = 8.0;
        b.data[1][0] = 9.0;
        b.data[1][1] = 10.0;
        b.data[2][0] = 11.0;
        b.data[2][1] = 12.0;

        let result = a.multiply(&b).unwrap();

        assert_eq!(result.dimensions(), (2, 2));
        assert_eq!(result.data[0][0], 58.0);
        assert_eq!(result.data[0][1], 64.0);
        assert_eq!(result.data[1][0], 139.0);
        assert_eq!(result.data[1][1], 154.0);

        let c = Matrix::<f64>::new(4, 3);
        let multiply_result = a.multiply(&c);
        assert!(matches!(multiply_result, Err(MathError::DimensionMismatch)));

        let identity = Matrix::<f64>::identity(3);
        let d = Matrix::<f64>::from_vec(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ])
        .unwrap();

        assert_eq!(identity.multiply(&d).unwrap(), d);
        assert_eq!(d.multiply(&identity).unwrap(), d);
    }

    #[test]
    fn test_matrix_transpose() {
        // Create a fixed test matrix
        let matrix = Matrix::<f64>::from_vec(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ])
        .unwrap();

        let mt = matrix.transpose();

        assert_eq!(matrix.dimensions(), (3, 4));
        assert_eq!(mt.dimensions(), (4, 3));

        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(mt.data[j][i], matrix.data[i][j]);
            }
        }

        let identity = Matrix::<f64>::identity(5);
        assert_eq!(identity.transpose(), identity);
        assert_eq!(mt.transpose().dimensions(), matrix.dimensions());

        let mut complex_matrix = Matrix::<Complex<f64>>::new(2, 2);
        complex_matrix.data[0][0] = Complex::new(1.0, 1.0);
        complex_matrix.data[0][1] = Complex::new(2.0, 2.0);
        complex_matrix.data[1][0] = Complex::new(3.0, 3.0);
        complex_matrix.data[1][1] = Complex::new(4.0, 4.0);

        let ct = complex_matrix.transpose();

        assert_eq!(ct.data[0][0], Complex::new(1.0, -1.0));
        assert_eq!(ct.data[0][1], Complex::new(3.0, -3.0));
        assert_eq!(ct.data[1][0], Complex::new(2.0, -2.0));
        assert_eq!(ct.data[1][1], Complex::new(4.0, -4.0));
    }

    #[test]
    fn test_matrix_inverse() {
        use crate::algebra::Matrix;

        // Test 1: Identity matrix
        let identity = Matrix::<f64>::identity(4);
        let inverse = identity.inverse().unwrap();
        let product = identity.multiply(&inverse).unwrap();

        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert!((product.data[i][j] - 1.0).abs() < 1e-10);
                } else {
                    assert!(product.data[i][j].abs() < 1e-10);
                }
            }
        }

        // Test 2: Simple diagonal matrix
        let mut diagonal = Matrix::<f64>::new(3, 3);
        diagonal.data[0][0] = 2.0;
        diagonal.data[1][1] = 4.0;
        diagonal.data[2][2] = 5.0;

        let diagonal_inverse = diagonal.inverse().unwrap();

        assert!((diagonal_inverse.data[0][0] - 0.5).abs() < 1e-10);
        assert!((diagonal_inverse.data[1][1] - 0.25).abs() < 1e-10);
        assert!((diagonal_inverse.data[2][2] - 0.2).abs() < 1e-10);

        let product = diagonal.multiply(&diagonal_inverse).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((product.data[i][j] - 1.0).abs() < 1e-10);
                } else {
                    assert!(product.data[i][j].abs() < 1e-10);
                }
            }
        }

        // Test 3: General invertible matrix
        let mut general = Matrix::<f64>::new(3, 3);
        general.data[0][0] = 1.0;
        general.data[0][1] = 2.0;
        general.data[0][2] = 3.0;
        general.data[1][0] = 0.0;
        general.data[1][1] = 1.0;
        general.data[1][2] = 4.0;
        general.data[2][0] = 5.0;
        general.data[2][1] = 6.0;
        general.data[2][2] = 0.0;

        let general_inverse = general.inverse().unwrap();

        let product = general.multiply(&general_inverse).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((product.data[i][j] - 1.0).abs() < 1e-10);
                } else {
                    assert!(product.data[i][j].abs() < 1e-10);
                }
            }
        }

        let product2 = general_inverse.multiply(&general).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((product2.data[i][j] - 1.0).abs() < 1e-10);
                } else {
                    assert!(product2.data[i][j].abs() < 1e-10);
                }
            }
        }

        // Test 4: Test for non-invertible matrix
        let mut singular = Matrix::<f64>::new(3, 3);
        singular.data[0][0] = 1.0;
        singular.data[0][1] = 2.0;
        singular.data[0][2] = 3.0;
        singular.data[1][0] = 4.0;
        singular.data[1][1] = 5.0;
        singular.data[1][2] = 6.0;
        singular.data[2][0] = 7.0;
        singular.data[2][1] = 8.0;
        singular.data[2][2] = 9.0;

        let result = singular.inverse();
        assert!(result.is_err());

        // Test 5: Complex number matrix
        use num_complex::Complex;
        let mut complex_matrix = Matrix::<Complex<f64>>::new(2, 2);
        complex_matrix.data[0][0] = Complex::new(1.0, 1.0);
        complex_matrix.data[0][1] = Complex::new(2.0, 0.0);
        complex_matrix.data[1][0] = Complex::new(0.0, 1.0);
        complex_matrix.data[1][1] = Complex::new(1.0, 0.0);

        let complex_inverse = complex_matrix.inverse().unwrap();

        let product = complex_matrix.multiply(&complex_inverse).unwrap();

        let identity = Complex::new(1.0, 0.0);
        let zero = Complex::new(0.0, 0.0);

        assert!((product.data[0][0] - identity).norm() < 1e-10);
        assert!((product.data[1][1] - identity).norm() < 1e-10);

        assert!((product.data[0][1] - zero).norm() < 1e-10);
        assert!((product.data[1][0] - zero).norm() < 1e-10);
    }
}

#[cfg(test)]
mod cw_tests {
    use std::collections::HashSet;
    use std::hash::Hash;

    use crate::cw::{Cell, KCell, Skeleton};
    use crate::error::MathError;
    use crate::sheaf::OpenSet;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct TestPoint(String);

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestOpenSet {
        points: HashSet<TestPoint>,
    }

    impl OpenSet for TestOpenSet {
        type Point = TestPoint;

        fn union(&self, other: Self) -> Self {
            let mut points = self.points.clone();
            for point in other.points.iter() {
                points.insert(point.clone());
            }
            Self { points }
        }
        fn intersection(&self, other: Self) -> Self {
            let mut points = HashSet::new();
            for point in self.points.iter() {
                if other.points.contains(point) {
                    points.insert(point.clone());
                }
            }
            Self { points }
        }
        fn from(iter: Box<dyn Iterator<Item = Self::Point>>) -> Self {
            let points = iter.collect();
            Self { points }
        }
    }

    impl IntoIterator for TestOpenSet {
        type Item = TestPoint;
        type IntoIter = std::collections::hash_set::IntoIter<TestPoint>;

        fn into_iter(self) -> Self::IntoIter {
            self.points.into_iter()
        }
    }

    struct TestVertex {
        point: TestPoint,
    }

    impl KCell<String, TestOpenSet> for TestVertex {
        fn points(&self) -> HashSet<&TestPoint> {
            let mut points = HashSet::new();
            points.insert(&self.point);
            points
        }
        fn dimension(&self) -> usize {
            0
        }
        fn boundary(&mut self) -> HashSet<TestPoint> {
            HashSet::new()
        }
        fn attach(
            &self,
            _point: &TestPoint,
            _skeleton: &mut Skeleton<String, TestOpenSet>,
        ) -> TestPoint {
            self.point.clone()
        }
        fn remove(&mut self, _set: HashSet<TestPoint>) -> bool {
            false
        }
    }

    struct TestEdge {
        start: TestPoint,
        end: TestPoint,
        id: String,
        removed: HashSet<TestPoint>,
    }

    impl KCell<String, TestOpenSet> for TestEdge {
        fn points(&self) -> HashSet<&TestPoint> {
            let mut points = HashSet::new();
            points.insert(&self.start);
            points.insert(&self.end);
            points
        }
        fn dimension(&self) -> usize {
            1
        }
        fn boundary(&mut self) -> HashSet<TestPoint> {
            let mut boundary = HashSet::new();
            boundary.insert(self.start.clone());
            boundary.insert(self.end.clone());
            boundary
        }
        fn attach(
            &self,
            point: &TestPoint,
            _skeleton: &mut Skeleton<String, TestOpenSet>,
        ) -> TestPoint {
            if point == &self.start {
                self.start.clone()
            } else {
                self.end.clone()
            }
        }
        fn remove(&mut self, set: HashSet<TestPoint>) -> bool {
            self.removed = set;
            !self.removed.is_empty()
        }
    }

    struct TestFace {
        vertices: Vec<TestPoint>,
        edges: Vec<String>,
        id: String,
        removed: HashSet<TestPoint>,
    }

    impl KCell<String, TestOpenSet> for TestFace {
        fn points(&self) -> HashSet<&TestPoint> {
            let mut points = HashSet::new();
            for v in &self.vertices {
                points.insert(v);
            }
            points
        }

        fn dimension(&self) -> usize {
            2
        }

        fn boundary(&mut self) -> HashSet<TestPoint> {
            let mut boundary = HashSet::new();
            for v in &self.vertices {
                boundary.insert(v.clone());
            }
            boundary
        }

        fn attach(
            &self,
            point: &TestPoint,
            _skeleton: &mut Skeleton<String, TestOpenSet>,
        ) -> TestPoint {
            for v in &self.vertices {
                if v == point {
                    return v.clone();
                }
            }
            self.vertices[0].clone() // Default to first vertex if not found
        }

        fn remove(&mut self, set: HashSet<TestPoint>) -> bool {
            self.removed = set;
            !self.removed.is_empty()
        }
    }

    #[test]
    fn test_skeleton_init() {
        let skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();
        assert_eq!(skeleton.dimension, 0);
        assert_eq!(skeleton.cells.len(), 0);
    }

    #[test]
    fn test_skeleton_attach_vertex() {
        let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();
        let vertex = TestVertex {
            point: TestPoint("A".to_string()),
        };

        let result = skeleton.attach(Box::new(vertex));
        assert!(result.is_ok());
        assert_eq!(skeleton.dimension, 0);
        assert_eq!(skeleton.cells.len(), 1);
    }

    #[test]
    fn test_skeleton_attach_edge() {
        let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();
        let vertex1 = TestVertex {
            point: TestPoint("A".to_string()),
        };
        let vertex2 = TestVertex {
            point: TestPoint("B".to_string()),
        };
        skeleton.attach(Box::new(vertex1)).unwrap();
        skeleton.attach(Box::new(vertex2)).unwrap();

        let edge = TestEdge {
            start: TestPoint("A".to_string()),
            end: TestPoint("B".to_string()),
            id: "Edge_AB".to_string(),
            removed: HashSet::new(),
        };

        let result = skeleton.attach(Box::new(edge));
        assert!(result.is_ok());
        assert_eq!(skeleton.dimension, 1);
        assert_eq!(skeleton.cells.len(), 3);
    }

    #[test]
    fn test_skeleton_dimension_error() {
        let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();

        let face = TestFace {
            vertices: vec![
                TestPoint("A".to_string()),
                TestPoint("B".to_string()),
                TestPoint("C".to_string()),
            ],
            edges: vec![
                "Edge_AB".to_string(),
                "Edge_BC".to_string(),
                "Edge_CA".to_string(),
            ],
            id: "Face_ABC".to_string(),
            removed: HashSet::new(),
        };

        let result = skeleton.attach(Box::new(face));
        assert!(matches!(result, Err(MathError::DimensionMismatch)));
    }

    #[test]
    fn test_skeleton_uninitialized_error() {
        let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();

        let edge = TestEdge {
            start: TestPoint("A".to_string()),
            end: TestPoint("B".to_string()),
            id: "Edge_AB".to_string(),
            removed: HashSet::new(),
        };

        let result = skeleton.attach(Box::new(edge));
        assert!(matches!(result, Err(MathError::CWUninitialized)));
    }

    #[test]
    fn test_skeleton_fetch_cell_by_point() {
        let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();
        let vertex = TestVertex {
            point: TestPoint("A".to_string()),
        };
        skeleton.attach(Box::new(vertex)).unwrap();

        let result = skeleton.fetch_cell_by_point(TestPoint("A".to_string()));
        assert!(result.is_ok());
        let result = skeleton.fetch_cell_by_point(TestPoint("Z".to_string()));
        assert!(matches!(result, Err(MathError::NoPointFound)));
    }

    #[test]
    fn test_skeleton_incident_cells() {
        let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();
        let vertex1 = TestVertex {
            point: TestPoint("A".to_string()),
        };
        let vertex2 = TestVertex {
            point: TestPoint("B".to_string()),
        };

        skeleton.attach(Box::new(vertex1)).unwrap();
        skeleton.attach(Box::new(vertex2)).unwrap();

        let edge = TestEdge {
            start: TestPoint("A".to_string()),
            end: TestPoint("B".to_string()),
            id: "Edge_AB".to_string(),
            removed: HashSet::new(),
        };
        skeleton.attach(Box::new(edge)).unwrap();

        let result = skeleton.incident_cells(0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
        let result = skeleton.incident_cells(1);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
        let result = skeleton.incident_cells(2);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
        let result = skeleton.incident_cells(99);
        assert!(matches!(result, Err(MathError::InvalidCellIdx)));
    }

    #[test]
    fn test_skeleton_filter_incident_by_dim() {
        let mut skeleton: Skeleton<String, TestOpenSet> = Skeleton::init();
        let vertex1 = TestVertex {
            point: TestPoint("A".to_string()),
        };
        let vertex2 = TestVertex {
            point: TestPoint("B".to_string()),
        };
        let vertex3 = TestVertex {
            point: TestPoint("C".to_string()),
        };

        skeleton.attach(Box::new(vertex1)).unwrap();
        skeleton.attach(Box::new(vertex2)).unwrap();
        skeleton.attach(Box::new(vertex3)).unwrap();

        let edge1 = TestEdge {
            start: TestPoint("A".to_string()),
            end: TestPoint("B".to_string()),
            id: "Edge_AB".to_string(),
            removed: HashSet::new(),
        };
        let edge2 = TestEdge {
            start: TestPoint("B".to_string()),
            end: TestPoint("C".to_string()),
            id: "Edge_BC".to_string(),
            removed: HashSet::new(),
        };
        let edge3 = TestEdge {
            start: TestPoint("C".to_string()),
            end: TestPoint("A".to_string()),
            id: "Edge_CA".to_string(),
            removed: HashSet::new(),
        };
        skeleton.attach(Box::new(edge1)).unwrap();
        skeleton.attach(Box::new(edge2)).unwrap();
        skeleton.attach(Box::new(edge3)).unwrap();

        let face = TestFace {
            vertices: vec![
                TestPoint("A".to_string()),
                TestPoint("B".to_string()),
                TestPoint("C".to_string()),
            ],
            edges: vec![
                "Edge_AB".to_string(),
                "Edge_BC".to_string(),
                "Edge_CA".to_string(),
            ],
            id: "Face_ABC".to_string(),
            removed: HashSet::new(),
        };
        skeleton.attach(Box::new(face)).unwrap();

        let result = skeleton.filter_incident_by_dim(0);
        assert!(result.is_ok());
        let (lower, upper) = result.unwrap();
        assert_eq!(lower.len(), 0);
        assert_eq!(upper.len(), 2);
        let result = skeleton.filter_incident_by_dim(3);
        assert!(result.is_ok());
        let (lower, upper) = result.unwrap();
        assert_eq!(lower.len(), 2);
        assert_eq!(upper.len(), 1);
        let result = skeleton.filter_incident_by_dim(6);
        assert!(result.is_ok());
        let (lower, upper) = result.unwrap();
        assert_eq!(lower.len(), 3);
        assert_eq!(upper.len(), 0);
    }
}

#[cfg(test)]
mod sheaf_tests {
    use std::collections::HashSet;
    use std::hash::Hash;

    use num_complex::Complex;

    use crate::algebra::{Field, Matrix};
    use crate::cw::{KCell, Skeleton};
    use crate::error::MathError;
    use crate::sheaf::{CellularSheaf, OpenSet, Sections};

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct TestPoint(String);

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestOpenSet {
        points: HashSet<TestPoint>,
    }

    impl OpenSet for TestOpenSet {
        type Point = TestPoint;

        fn union(&self, other: Self) -> Self {
            let mut points = self.points.clone();
            for point in other.points.iter() {
                points.insert(point.clone());
            }
            Self { points }
        }
        fn intersection(&self, other: Self) -> Self {
            let mut points = HashSet::new();
            for point in self.points.iter() {
                if other.points.contains(point) {
                    points.insert(point.clone());
                }
            }
            Self { points }
        }
        fn from(iter: Box<dyn Iterator<Item = Self::Point>>) -> Self {
            let points = iter.collect();
            Self { points }
        }
    }

    impl IntoIterator for TestOpenSet {
        type Item = TestPoint;
        type IntoIter = std::collections::hash_set::IntoIter<TestPoint>;

        fn into_iter(self) -> Self::IntoIter {
            self.points.into_iter()
        }
    }

    struct TestVertex {
        point: TestPoint,
    }

    impl KCell<String, TestOpenSet> for TestVertex {
        fn points(&self) -> HashSet<&TestPoint> {
            let mut points = HashSet::new();
            points.insert(&self.point);
            points
        }
        fn dimension(&self) -> usize {
            0
        }
        fn boundary(&mut self) -> HashSet<TestPoint> {
            HashSet::new()
        }
        fn attach(
            &self,
            _point: &TestPoint,
            _skeleton: &mut Skeleton<String, TestOpenSet>,
        ) -> TestPoint {
            self.point.clone()
        }
        fn remove(&mut self, _set: HashSet<TestPoint>) -> bool {
            false
        }
    }

    struct TestEdge {
        start: TestPoint,
        end: TestPoint,
        id: String,
        removed: HashSet<TestPoint>,
    }

    impl KCell<String, TestOpenSet> for TestEdge {
        fn points(&self) -> HashSet<&TestPoint> {
            let mut points = HashSet::new();
            points.insert(&self.start);
            points.insert(&self.end);
            points
        }
        fn dimension(&self) -> usize {
            1
        }
        fn boundary(&mut self) -> HashSet<TestPoint> {
            let mut boundary = HashSet::new();
            boundary.insert(self.start.clone());
            boundary.insert(self.end.clone());
            boundary
        }
        fn attach(
            &self,
            point: &TestPoint,
            _skeleton: &mut Skeleton<String, TestOpenSet>,
        ) -> TestPoint {
            if point == &self.start {
                self.start.clone()
            } else {
                self.end.clone()
            }
        }
        fn remove(&mut self, set: HashSet<TestPoint>) -> bool {
            self.removed = set;
            !self.removed.is_empty()
        }
    }

    #[test]
    fn test_sheaf_init() {
        let sheaf: CellularSheaf<f64, String, TestOpenSet> = CellularSheaf::init();
        assert_eq!(sheaf.cw.dimension, 0);
        assert_eq!(sheaf.cw.cells.len(), 0);
        assert_eq!(sheaf.section_spaces.len(), 0);
        assert_eq!(sheaf.restrictions.len(), 0);
        assert_eq!(sheaf.global_sections.len(), 0);
    }

    #[test]
    fn test_sections_new() {
        let sections = Sections::<f64>::new(3);
        assert_eq!(sections.dimension, 3);
        assert!(sections.data.is_empty());
        assert!(sections.bases.is_none());
    }

    #[test]
    fn test_sheaf_attach() {
        let mut sheaf: CellularSheaf<f64, String, TestOpenSet> = CellularSheaf::init();
        let vertex = TestVertex {
            point: TestPoint("A".to_string()),
        };

        let result = sheaf.attach(Box::new(vertex), None, 2);
        assert!(result.is_ok());
        assert_eq!(sheaf.cw.cells.len(), 1);
        assert_eq!(sheaf.section_spaces.len(), 1);
        assert_eq!(sheaf.section_spaces[0].dimension, 2);

        let vertex2 = TestVertex {
            point: TestPoint("B".to_string()),
        };

        let mut custom_sections = Sections::<f64>::new(3);
        custom_sections.add_section(vec![1.0, 2.0, 3.0]);
        custom_sections.add_section(vec![4.0, 5.0, 6.0]);

        let result = sheaf.attach(Box::new(vertex2), Some(custom_sections), 0);
        assert!(result.is_ok());
        assert_eq!(sheaf.cw.cells.len(), 2);
        assert_eq!(sheaf.section_spaces.len(), 2);
        assert_eq!(sheaf.section_spaces[1].dimension, 3);
        assert_eq!(sheaf.section_spaces[1].data.len(), 2);
        assert_eq!(sheaf.section_spaces[1].data[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(sheaf.section_spaces[1].data[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sheaf_update_data() {
        let mut sheaf: CellularSheaf<f64, String, TestOpenSet> = CellularSheaf::init();
        let vertex = TestVertex {
            point: TestPoint("A".to_string()),
        };

        let mut custom_sections = Sections::<f64>::new(2);
        custom_sections.add_section(vec![1.0, 2.0]);

        sheaf
            .attach(Box::new(vertex), Some(custom_sections), 0)
            .unwrap();

        let result = sheaf.update(0, 0, vec![3.0, 4.0]);
        assert!(result.is_ok());
        assert_eq!(sheaf.section_spaces[0].data[0], vec![3.0, 4.0]);

        let result = sheaf.update(99, 0, vec![5.0, 6.0]);
        assert!(matches!(result, Err(MathError::InvalidCellIdx)));
        let result = sheaf.update(0, 99, vec![5.0, 6.0]);
        assert!(matches!(result, Err(MathError::InvalidDataIdx)));
    }

    #[test]
    fn test_sheaf_new_data() {
        let mut sheaf: CellularSheaf<f64, String, TestOpenSet> = CellularSheaf::init();
        let vertex = TestVertex {
            point: TestPoint("A".to_string()),
        };

        let mut custom_sections = Sections::<f64>::new(2);
        custom_sections.add_section(vec![1.0, 2.0]);

        sheaf
            .attach(Box::new(vertex), Some(custom_sections), 0)
            .unwrap();

        let result = sheaf.new_data(0, vec![3.0, 4.0]);
        assert!(result.is_ok());
        assert_eq!(sheaf.section_spaces[0].data.len(), 2);
        assert_eq!(sheaf.section_spaces[0].data[0], vec![1.0, 2.0]);
        assert_eq!(sheaf.section_spaces[0].data[1], vec![3.0, 4.0]);
        let result = sheaf.new_data(99, vec![5.0, 6.0]);
        assert!(matches!(result, Err(MathError::InvalidCellIdx)));
    }

    fn create_test_graph() -> CellularSheaf<f64, String, TestOpenSet> {
        let mut sheaf: CellularSheaf<f64, String, TestOpenSet> = CellularSheaf::init();

        let vertex_a = TestVertex {
            point: TestPoint("A".to_string()),
        };
        let vertex_b = TestVertex {
            point: TestPoint("B".to_string()),
        };
        let vertex_c = TestVertex {
            point: TestPoint("C".to_string()),
        };

        let mut sections_a = Sections::<f64>::new(1);
        sections_a.add_section(vec![1.0]);
        let mut sections_b = Sections::<f64>::new(1);
        sections_b.add_section(vec![2.0]);
        let mut sections_c = Sections::<f64>::new(1);
        sections_c.add_section(vec![3.0]);

        sheaf
            .attach(Box::new(vertex_a), Some(sections_a), 1)
            .unwrap();
        sheaf
            .attach(Box::new(vertex_b), Some(sections_b), 1)
            .unwrap();
        sheaf
            .attach(Box::new(vertex_c), Some(sections_c), 1)
            .unwrap();

        let edge_ab = TestEdge {
            start: TestPoint("A".to_string()),
            end: TestPoint("B".to_string()),
            id: "Edge_AB".to_string(),
            removed: HashSet::new(),
        };
        let edge_bc = TestEdge {
            start: TestPoint("B".to_string()),
            end: TestPoint("C".to_string()),
            id: "Edge_BC".to_string(),
            removed: HashSet::new(),
        };
        let mut sections_bc = Sections::<f64>::new(1);
        sections_bc.add_section(vec![3.0, 2.0]);
        sheaf
            .attach(Box::new(edge_bc), Some(sections_bc), 2)
            .unwrap();
        let mut sections_ab = Sections::<f64>::new(1);
        sections_ab.add_section(vec![3.0, 1.0]);
        sheaf
            .attach(Box::new(edge_ab), Some(sections_ab), 2)
            .unwrap();

        let mut b_restriction_ab = Matrix::new(2, 1);
        b_restriction_ab.data[0][0] = 3.0;
        b_restriction_ab.data[1][0] = 6.0;
        let mut b_restriction_bc = Matrix::new(2, 1);
        b_restriction_bc.data[0][0] = 1.5;
        b_restriction_bc.data[1][0] = 2.5;
        sheaf.set_restriction(1, 3, b_restriction_ab).unwrap();
        sheaf.set_restriction(1, 4, b_restriction_bc).unwrap();

        sheaf
    }

    #[test]
    fn test_sheaf_set_restriction() {
        let mut sheaf = create_test_graph();

        assert_eq!(sheaf.restrictions.len(), 2);

        let restriction = Matrix::<f64>::identity(1);
        let result = sheaf.set_restriction(99, 0, restriction.clone());
        assert!(matches!(result, Err(MathError::InvalidCellIdx)));
        let result = sheaf.set_restriction(0, 2, restriction);
        assert!(matches!(result, Err(MathError::NoRestrictionDefined)));
    }

    #[test]
    fn test_sheaf_k_coboundary() {
        let sheaf = create_test_graph();

        let result = sheaf.k_coboundary(1, 0);
        assert!(result.is_ok());

        let coboundary = result.unwrap();
        assert_eq!(coboundary.len(), 2);

        let (a_value, a_idx) = &coboundary[0];
        let (b_value, b_idx) = &coboundary[1];

        assert!((*a_idx == 3 && a_value[0] == 6.0) || (*b_idx == 3 && b_value[0] == 6.0));
        assert!((*a_idx == 3 && a_value[1] == 12.0) || (*b_idx == 3 && b_value[1] == 12.0));
        assert!((*a_idx == 4 && a_value[0] == 3.0) || (*b_idx == 4 && b_value[0] == 3.0));
        assert!((*a_idx == 4 && a_value[1] == 5.0) || (*b_idx == 4 && b_value[1] == 5.0));

        // Test error cases
        let result = sheaf.k_coboundary(99, 0);
        assert!(matches!(result, Err(MathError::InvalidCellIdx)));

        let result = sheaf.k_coboundary(0, 99);
        assert!(matches!(result, Err(MathError::InvalidDataIdx)));
    }

    #[test]
    fn test_sheaf_complex_field() {
        let mut sheaf: CellularSheaf<Complex<f64>, String, TestOpenSet> = CellularSheaf::init();
        let vertex = TestVertex {
            point: TestPoint("A".to_string()),
        };

        let mut sections = Sections::<Complex<f64>>::new(2);
        sections.add_section(vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)]);

        let result = sheaf.attach(Box::new(vertex), Some(sections), 0);
        assert!(result.is_ok());
        assert_eq!(sheaf.section_spaces[0].data[0][0], Complex::new(1.0, 1.0));
        assert_eq!(sheaf.section_spaces[0].data[0][1], Complex::new(2.0, 2.0));
    }

    #[test]
    fn test_sheaf_with_bases() {
        let mut sheaf: CellularSheaf<f64, String, TestOpenSet> = CellularSheaf::init();
        let vertex = TestVertex {
            point: TestPoint("A".to_string()),
        };

        let mut sections = Sections::<f64>::new(2);
        sections.add_section(vec![1.0, 0.0]);
        sections.add_section(vec![0.0, 1.0]);

        sections.add_bases(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

        let result = sheaf.attach(Box::new(vertex), Some(sections), 0);
        assert!(result.is_ok());
        assert!(sheaf.section_spaces[0].bases.is_some());
        let bases = sheaf.section_spaces[0].bases.as_ref().unwrap();
        assert_eq!(bases.len(), 2);
        assert_eq!(bases[0], vec![1.0, 0.0]);
        assert_eq!(bases[1], vec![0.0, 1.0]);
    }
}
