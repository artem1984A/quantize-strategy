//! Quantization strategies.

pub mod attention_aware;
pub mod l2_norm;
// pub mod qr_pivot;    
// pub mod learnable;    

pub use attention_aware::AttentionAwareStrategy;
pub use l2_norm::L2NormStrategy;

use crate::core::{QuantizationConfig, QuantizationResult};
use anyhow::Result;
use std::path::Path;

use candle_core::quantized::k_quants::{BlockQ8K, QK_K};
use candle_core::quantized::GgmlType;

#[derive(Debug, Clone)]
pub enum StrategyType {
    L2Norm,
    AttentionAware,
    QRPivot,
    Learnable {
        learning_rate: f64,
        iterations: usize,
    },
}

pub trait QuantizationStrategy {
    /// Apply permutation strategy to the given data
    fn apply_permutation(
        &self,
        data: &[f32],
        rows: usize,
        k: usize,
        tensor_name: &str,
    ) -> Result<(Vec<f32>, Option<Vec<usize>>)>;

    /// Get strategy name for logging
    fn name(&self) -> &'static str;
}

pub fn create_strategy(strategy_type: &StrategyType) -> Box<dyn QuantizationStrategy> {
    match strategy_type {
        StrategyType::L2Norm => Box::new(L2NormStrategy::new()),
        StrategyType::AttentionAware => Box::new(AttentionAwareStrategy::new()),
        StrategyType::QRPivot => {
            // Box::new(QRPivotStrategy::new())
            unimplemented!("QR Pivot strategy not yet implemented")
        }
        StrategyType::Learnable { .. } => {
            // Box::new(LearnableStrategy::new(learning_rate, iterations))
            unimplemented!("Learnable strategy not yet implemented")
        }
    }
}

pub fn run_quantization(
    input_path: &Path,
    config: QuantizationConfig,
) -> Result<QuantizationResult> {
    use crate::core::io::{write_perm, write_q8k};
    use crate::core::validation::{validate_quantization, validate_quantization_direct};
    use crate::utils::{is_target_weight, tensor_to_f32};
    use safetensors::SafeTensors;
    use std::{fs, time::Instant};

    let start_time = Instant::now();
    fs::create_dir_all(&config.output_dir)?;

    let bytes = fs::read(input_path)?;
    let st = SafeTensors::deserialize(&bytes)?;

    let mut quantized_count = 0;
    let mut skipped_count = 0;
    let mut mse_stats = Vec::new();

    let strategy = if config.use_permutation {
        Some(create_strategy(&config.strategy_type))
    } else {
        None
    };

    println!("Tensors: {}", st.len());

    for name in st.names() {
        let tensor = st.tensor(name)?;
        let shape = tensor.shape();

        if shape.len() != 2 || !is_target_weight(name, &config.skip_patterns) {
            skipped_count += 1;
            continue;
        }

        let (rows, k) = (shape[0], shape[1]);
        if k % QK_K != 0 {
            println!("skip (k % {QK_K} != 0): {name} [{rows} x {k}]");
            skipped_count += 1;
            continue;
        }

        println!("quantizing {name} ({rows} x {k})");

        // Load weights to f32
        let data_f32 = tensor_to_f32(tensor.data(), tensor.dtype())?;

        // Apply permutation strategy if enabled
        let (data_for_quant, maybe_perm) = if let Some(ref strat) = strategy {
            strat.apply_permutation(&data_f32, rows, k, name)?
        } else {
            (data_f32, None)
        };

        // Quantize to BlockQ8K
        let blocks = quantize_rows_q8k(rows, k, &data_for_quant)?;

        // Validate quantization quality

        let mse_matmul = validate_quantization(&data_for_quant, &blocks, k)?;
        let mse_direct = validate_quantization_direct(&data_for_quant, &blocks, k)?;
        // Compare and log the results
        let diff = (mse_matmul - mse_direct).abs();
        if diff > 1e-6 {
            println!("    [INFO] Validation methods differ by {:.8e}", diff);
        }
        mse_stats.push((name.to_string(), mse_matmul, mse_direct));

        println!(
            "  MSE (matmul): {:.6e}, MSE (direct): {:.6e}",
            mse_matmul, mse_direct
        );

        let diff = (mse_matmul - mse_direct).abs();
        if diff > 1e-6 {
            println!("    [INFO] Validation methods differ by {:.8e}", diff);
        }
        if mse_matmul > 1e-2 || mse_direct > 1e-2 {
            println!("    [WARN] High MSE detected - quantization may be lossy");
        }

        // Write quantized data
        let out_path = config.output_dir.join(format!("{}.q8k", name));
        write_q8k(&out_path, rows, k, &blocks)?;

        // Write permutation if used
        if let Some(perm) = maybe_perm {
            write_perm(&out_path, &perm)?;
        }

        quantized_count += 1;
    }

    Ok(QuantizationResult {
        quantized_tensors: quantized_count,
        skipped_tensors: skipped_count,
        total_time_seconds: start_time.elapsed().as_secs_f32(),
        mse_stats,
    })
}

fn quantize_rows_q8k(rows: usize, k: usize, data: &[f32]) -> Result<Vec<BlockQ8K>> {
    use anyhow::bail;

    if k % QK_K != 0 {
        bail!("inner dim {k} not multiple of {QK_K}");
    }
    let blocks_per_row = k / QK_K;
    let mut blocks = vec![BlockQ8K::zeros(); rows * blocks_per_row];
    for r in 0..rows {
        let row = &data[r * k..(r + 1) * k];
        let dst = &mut blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
        BlockQ8K::from_float(row, dst)?;
    }
    Ok(blocks)
}
