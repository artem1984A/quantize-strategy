//! CLI interface for Q8K quantization with advanced strategies.

use anyhow::{Context, Result};
use quantize_strategy::{quantize_safetensors, QuantizationConfig, StrategyType};
use std::path::PathBuf;

fn main() -> Result<()> {
    // Parse arguments
    let mut args = std::env::args().skip(1);
    let in_file: PathBuf = args
        .next()
        .context("Usage: quantize_q8k <input.safetensors> <output_dir>")?
        .into();
    let out_dir: PathBuf = args
        .next()
        .context("Usage: quantize_q8k <input.safetensors> <output_dir>")?
        .into();

    // Configuration from environment
    let use_permutation = std::env::var("CANDLE_Q8K_PERMUTE")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let strategy_name =
        std::env::var("CANDLE_Q8K_STRATEGY").unwrap_or_else(|_| "l2_norm".to_string());

    let strategy_type = match strategy_name.as_str() {
        "attention_aware" => StrategyType::AttentionAware,
        "qr_pivot" => StrategyType::QRPivot,
        "learnable" => StrategyType::Learnable {
            learning_rate: 0.01,
            iterations: 1000,
        },
        _ => StrategyType::L2Norm,
    };

    let config = QuantizationConfig {
        strategy_type,
        use_permutation,
        output_dir: out_dir.clone(),
        attention_aware: strategy_name == "attention_aware",
        ..Default::default()
    };

    // Print configuration
    println!("Input  : {}", in_file.display());
    println!("Output : {}", out_dir.display());
    println!("Permute: {}", if use_permutation { "on" } else { "off" });
    println!("Strategy: {}", strategy_name);

    // Run quantization
    let result = quantize_safetensors(&in_file, config)?;

    // Print results
    println!(
        "Done in {:.2}s. Quantized: {}, skipped: {}",
        result.total_time_seconds, result.quantized_tensors, result.skipped_tensors
    );

    if !result.mse_stats.is_empty() {
        println!("\nMSE Statistics:");
        for (name, mse_matmul, mse_direct) in result.mse_stats.iter() {
            println!(
                "  {}: matmul={:.6e}, direct={:.6e}",
                name, mse_matmul, mse_direct
            );
        }
    }

    Ok(())
}
