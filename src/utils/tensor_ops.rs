//! Tensor conversion operations.

use anyhow::{bail, Result};
use half::{bf16, f16};
use safetensors::tensor::Dtype;

pub fn tensor_to_f32(bytes: &[u8], dtype: Dtype) -> Result<Vec<f32>> {
    Ok(match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        Dtype::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f16::from_bits(bits).to_f32()
            })
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                bf16::from_bits(bits).to_f32()
            })
            .collect(),
        other => bail!("unsupported dtype {other:?}"),
    })
}
