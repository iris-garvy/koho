pub mod cw;
pub mod error;
pub mod sheaf;

pub fn add_vectors<T>(a: &[T], b: &[T]) -> Vec<T>
where
    T: std::ops::Add<Output = T> + Copy,
{
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_cw() {}
}
