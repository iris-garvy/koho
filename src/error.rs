#[derive(Clone, Debug)]
pub enum Error {
    DimensionMismatch,
    CWUninitialized,
    InvalidCellIdx,
    InvalidDataIdx,
    NullMatrix,
    NoPointFound,
    NotAdjacent,
    NoRestrictionDefined,
    BadField,
    ImproperBases,
    BadCochain,
}
