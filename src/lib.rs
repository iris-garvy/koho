pub mod cw;
pub mod error;
pub mod sheaf;

#[cfg(test)]
mod cw_tests {
    use std::collections::HashSet;
    use std::hash::Hash;

    use crate::cw::{KCell, Skeleton};
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
        _id: String,
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
        _edges: Vec<String>,
        _id: String,
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
            _id: "Edge_AB".to_string(),
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
            _edges: vec![
                "Edge_AB".to_string(),
                "Edge_BC".to_string(),
                "Edge_CA".to_string(),
            ],
            _id: "Face_ABC".to_string(),
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
            _id: "Edge_AB".to_string(),
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
            _id: "Edge_AB".to_string(),
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
            _id: "Edge_AB".to_string(),
            removed: HashSet::new(),
        };
        let edge2 = TestEdge {
            start: TestPoint("B".to_string()),
            end: TestPoint("C".to_string()),
            _id: "Edge_BC".to_string(),
            removed: HashSet::new(),
        };
        let edge3 = TestEdge {
            start: TestPoint("C".to_string()),
            end: TestPoint("A".to_string()),
            _id: "Edge_CA".to_string(),
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
            _edges: vec![
                "Edge_AB".to_string(),
                "Edge_BC".to_string(),
                "Edge_CA".to_string(),
            ],
            _id: "Face_ABC".to_string(),
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
    //! Tests for the `CellularSheaf` layer – sections, restrictions,
    //! (co)boundaries, Laplacians, and glue checking.
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashSet;

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct TestPoint(String);

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestOpenSet {
        points: HashSet<TestPoint>,
    }

    impl crate::sheaf::OpenSet for TestOpenSet {
        type Point = TestPoint;

        fn union(&self, other: Self) -> Self {
            let mut points = self.points.clone();
            points.extend(other.points);
            Self { points }
        }
        fn intersection(&self, other: Self) -> Self {
            let mut pts = HashSet::new();
            for p in &self.points {
                if other.points.contains(p) {
                    pts.insert(p.clone());
                }
            }
            Self { points: pts }
        }
        fn from(iter: Box<dyn Iterator<Item = Self::Point>>) -> Self {
            Self {
                points: iter.collect(),
            }
        }
    }

    impl IntoIterator for TestOpenSet {
        type Item = TestPoint;
        type IntoIter = std::collections::hash_set::IntoIter<TestPoint>;
        fn into_iter(self) -> Self::IntoIter {
            self.points.into_iter()
        }
    }

    /// 0-cell
    struct TestVertex {
        point: TestPoint,
    }
    impl crate::cw::KCell<String, TestOpenSet> for TestVertex {
        fn points(&self) -> HashSet<&TestPoint> {
            HashSet::from([&self.point])
        }
        fn dimension(&self) -> usize {
            0
        }
        fn boundary(&mut self) -> HashSet<TestPoint> {
            HashSet::new()
        }
        fn attach(
            &self,
            _p: &TestPoint,
            _skel: &mut crate::cw::Skeleton<String, TestOpenSet>,
        ) -> TestPoint {
            self.point.clone()
        }
        fn remove(&mut self, _set: HashSet<TestPoint>) -> bool {
            false
        }
    }

    /// 1-cell
    struct TestEdge {
        start: TestPoint,
        end: TestPoint,
        _id: String,
        removed: HashSet<TestPoint>,
    }
    impl crate::cw::KCell<String, TestOpenSet> for TestEdge {
        fn points(&self) -> HashSet<&TestPoint> {
            HashSet::from([&self.start, &self.end])
        }
        fn dimension(&self) -> usize {
            1
        }
        fn boundary(&mut self) -> HashSet<TestPoint> {
            HashSet::from([self.start.clone(), self.end.clone()])
        }
        fn attach(
            &self,
            p: &TestPoint,
            _skel: &mut crate::cw::Skeleton<String, TestOpenSet>,
        ) -> TestPoint {
            if p == &self.start {
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
    fn sheaf_init_is_empty() {
        let sheaf: crate::sheaf::CellularSheaf<String, TestOpenSet> =
            crate::sheaf::CellularSheaf::init(DType::F32, Device::Cpu);
        assert_eq!(sheaf.cw.cells.len(), 0);
        assert_eq!(sheaf.section_spaces.len(), 0);
        assert_eq!(sheaf.restrictions.len(), 0);
    }

    #[test]
    fn attach_cells_and_sections() {
        let mut sheaf =
            crate::sheaf::CellularSheaf::<String, TestOpenSet>::init(DType::F32, Device::Cpu);

        // ---- vertex ------------------------------------------------------
        let vertex = TestVertex {
            point: TestPoint("A".into()),
        };
        let mut v_sec = crate::sheaf::Sections::new(1, &Device::Cpu, DType::F32).unwrap();
        let v_data = Tensor::from_vec(vec![1f32], &[1, 1], &Device::Cpu).unwrap();
        v_sec.add_section_data(v_data.clone()).unwrap();
        sheaf.attach(Box::new(vertex), Some(v_sec), 1).unwrap();
        assert_eq!(sheaf.section_spaces.len(), 1);

        // ---- edge --------------------------------------------------------
        let edge = TestEdge {
            start: TestPoint("A".into()),
            end: TestPoint("B".into()),
            _id: "edge_AB".into(),
            removed: HashSet::new(),
        };
        let mut e_sec = crate::sheaf::Sections::new(1, &Device::Cpu, DType::F32).unwrap();
        let e_data = Tensor::from_vec(vec![0f32], &[1, 1], &Device::Cpu).unwrap();
        e_sec.add_section_data(e_data.clone()).unwrap();
        sheaf.attach(Box::new(edge), Some(e_sec), 1).unwrap();

        assert_eq!(sheaf.cw.dimension, 1);
        assert_eq!(sheaf.section_spaces.len(), 2);
    }

    #[test]
    fn invalid_indices_error_out() {
        let mut sheaf =
            crate::sheaf::CellularSheaf::<String, TestOpenSet>::init(DType::F32, Device::Cpu);

        // Requesting anything on an empty sheaf should bomb
        assert!(matches!(
            sheaf.update(0, Tensor::zeros(&[1, 1], DType::F32, &Device::Cpu).unwrap()),
            Err(crate::error::MathError::InvalidCellIdx)
        ));
        assert!(matches!(
            sheaf.k_coboundary(0),
            Err(crate::error::MathError::InvalidCellIdx)
        ));
    }

    // todo!() need to implement tests for the laplacian in cell complexes with three levels
}
