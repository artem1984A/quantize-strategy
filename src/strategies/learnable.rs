//! Learnable permutation strategies using gradient descent.

use super::QuantizationStrategy;
use anyhow::Result;

pub struct LearnableStrategy {
    learning_rate: f64,
    iterations: usize,
}

impl LearnableStrategy {
    pub fn new(learning_rate: f64, iterations: usize) -> Self {
        Self { learning_rate, iterations }
    }
}

impl QuantizationStrategy for LearnableStrategy {
    fn apply_permutation(
        &self,
        data: &[f32],
        rows: usize,
        k: usize,
        _tensor_name: &str,
    ) -> Result<(Vec<f32>, Option<Vec<usize>>)> {
        // TODO: Implement Gumbel-Softmax learnable permutation
        // For now, fall back to L2 norm
        use crate::utils::{column_l2_norms, build_column_permutation, apply_column_permutation};
        
        let norms = column_l2_norms(rows, k, data);
        let perm = build_column_permutation(&norms);
        let permuted = apply_column_permutation(rows, k, data, &perm);
        Ok((permuted, Some(perm)))
    }

    fn name(&self) -> &'static str {
        "Learnable"
    }
}