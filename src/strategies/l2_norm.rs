//! L2 norm-based column permutation strategy.

use super::QuantizationStrategy;
use crate::utils::{apply_column_permutation, build_column_permutation, column_l2_norms};
use anyhow::Result;

pub struct L2NormStrategy;

impl L2NormStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl QuantizationStrategy for L2NormStrategy {
    fn apply_permutation(
        &self,
        data: &[f32],
        rows: usize,
        k: usize,
        _tensor_name: &str,
    ) -> Result<(Vec<f32>, Option<Vec<usize>>)> {
        let norms = column_l2_norms(rows, k, data);
        let perm = build_column_permutation(&norms);
        let permuted = apply_column_permutation(rows, k, data, &perm);
        Ok((permuted, Some(perm)))
    }

    fn name(&self) -> &'static str {
        "L2Norm"
    }
}
