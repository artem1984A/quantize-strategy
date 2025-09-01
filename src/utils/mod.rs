//! Utility functions for quantization operations.

pub mod permutation;
pub mod tensor_ops;

pub use permutation::{apply_column_permutation, build_column_permutation, column_l2_norms};
pub use tensor_ops::tensor_to_f32;

pub fn is_target_weight(name: &str, skip_patterns: &[String]) -> bool {
    if !name.ends_with(".weight") {
        return false;
    }
    !skip_patterns.iter().any(|pattern| name.contains(pattern))
}
