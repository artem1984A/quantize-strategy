// //! Advanced quantization strategies for neural network models.

pub mod core;
pub mod strategies;
pub mod utils;

// Import for cleaner type references
use candle_core::quantized::k_quants::BlockQ8K;

// Re-export commonly used types
pub use core::{
    QuantizationConfig, QuantizationResult, Q8KHeader, 
    MAGIC_Q8K, VERSION, DTYPE_Q8K
};

pub use strategies::{
    QuantizationStrategy, StrategyType,
    L2NormStrategy, AttentionAwareStrategy
};

pub use utils::{
    tensor_to_f32, apply_column_permutation, 
    column_l2_norms, build_column_permutation
};

use anyhow::Result;
use std::path::Path;

/// High-level API for quantizing safetensors files
pub fn quantize_safetensors(
    input_path: &Path,
    config: QuantizationConfig,
) -> Result<QuantizationResult> {
    strategies::run_quantization(input_path, config)
}

/// Load a quantized .q8k tensor for inference
pub fn load_quantized_tensor(
    path: &Path,
) -> Result<(Vec<BlockQ8K>, usize, usize, Option<Vec<usize>>)> {
    core::io::load_q8k_tensor(path)
}