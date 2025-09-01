//! Quantization quality validation functions.

use anyhow::Result;
use candle_core::quantized::k_quants::{matmul, BlockQ8K};
use candle_core::Device;

pub fn validate_quantization(original: &[f32], blocks: &[BlockQ8K], k: usize) -> Result<f32> {
    let rows = original.len() / k;
    let _device = Device::Cpu;
    let test_input = vec![1.0f32; k]; // Simple test vector

    // Expected output: multiply original weights by test vector
    let mut expected_output = vec![0f32; rows];
    for row in 0..rows {
        let row_data = &original[row * k..(row + 1) * k];
        expected_output[row] = row_data.iter().sum(); // Sum since test_input is all 1s
    }

    // Actual output: multiply quantized weights by test vector
    let mut actual_output = vec![0f32; rows];
    matmul::<BlockQ8K>((1, k, rows), &test_input, blocks, &mut actual_output)
        .map_err(|e| anyhow::anyhow!("matmul failed: {}", e))?;

    // Calculate MSE between expected and actual outputs
    let mut mse = 0f32;
    for (expected, actual) in expected_output.iter().zip(actual_output.iter()) {
        let diff = expected - actual;
        mse += diff * diff;
    }
    mse /= rows as f32;

    Ok(mse)
}

pub fn validate_quantization_direct(
    original: &[f32],
    blocks: &[BlockQ8K],
    k: usize,
) -> Result<f32> {
    let rows = original.len() / k;

    // Use a different test pattern to validate reconstruction quality
    let mut test_input = vec![0f32; k];
    for i in 0..k {
        test_input[i] = (i as f32 + 1.0) / k as f32; // Gradient from 0 to 1
    }

    // Expected output: multiply original weights by test vector
    let mut expected_output = vec![0f32; rows];
    for row in 0..rows {
        let row_data = &original[row * k..(row + 1) * k];
        expected_output[row] = row_data
            .iter()
            .zip(test_input.iter())
            .map(|(a, b)| a * b)
            .sum();
    }

    // Actual output: multiply quantized weights by test vector
    let mut actual_output = vec![0f32; rows];
    matmul::<BlockQ8K>((1, k, rows), &test_input, blocks, &mut actual_output)
        .map_err(|e| anyhow::anyhow!("direct validation matmul failed: {}", e))?;

    // Calculate MSE between expected and actual outputs
    let mut mse = 0f32;
    for (expected, actual) in expected_output.iter().zip(actual_output.iter()) {
        let diff = expected - actual;
        mse += diff * diff;
    }
    mse /= rows as f32;

    Ok(mse)
}
