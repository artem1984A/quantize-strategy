//! Core quantization types and functionality.

pub mod header;
pub mod io;
pub mod validation;

pub use header::{Q8KHeader, DTYPE_Q8K, MAGIC_Q8K, VERSION};
pub use io::{load_perm, load_q8k_tensor, write_perm, write_q8k};
pub use validation::{validate_quantization, validate_quantization_direct};

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub strategy_type: crate::strategies::StrategyType,
    pub use_permutation: bool,
    pub skip_patterns: Vec<String>,
    pub output_dir: PathBuf,
    pub attention_aware: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            strategy_type: crate::strategies::StrategyType::L2Norm,
            use_permutation: false,
            skip_patterns: vec!["embed_tokens".to_string(), "norm".to_string()],
            output_dir: PathBuf::from("./quantized"),
            attention_aware: false,
        }
    }
}

#[derive(Debug)]
pub struct QuantizationResult {
    pub quantized_tensors: usize,
    pub skipped_tensors: usize,
    pub total_time_seconds: f32,
    pub mse_stats: Vec<(String, f32, f32)>,
}
