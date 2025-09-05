//! QR decomposition with column pivoting for optimal quantization permutation.

use super::QuantizationStrategy;
use anyhow::Result;

pub struct QRPivotStrategy {
    regularization: f32,
}

impl QRPivotStrategy {
    pub fn new(regularization: f32) -> Self {
        Self { regularization }
    }

    /// Compute QR decomposition with column pivoting
    /// Returns the optimal column permutation
    fn qr_column_pivoting(&self, data: &[f32], rows: usize, k: usize) -> Result<Vec<usize>> {
        // Initialize permutation as identity
        let mut perm: Vec<usize> = (0..k).collect();
        // Copy data for in-place QR computation
        let mut a = data.to_vec();
        let mut col_norms_sq = vec![0.0f32; k];
        // Initialize column norms
        for j in 0..k {
            for i in 0..rows {
                let val = a[i * k + j];
                col_norms_sq[j] += val * val;
            }
        }
        //Partial QR optimization
        let qr_steps = match k {
            k if k <= 64 => k,              // Tiny matrices: full QR
            k if k <= 256 => (k * 3) / 4,   // Small matrices: 75% QR
            k if k <= 512 => k / 2,         // Medium matrices: 50% QR
            k if k <= 1024 => k / 3,        // Large matrices: 33% QR
            k if k <= 2048 => k / 4,        // Very large matrices: 25% QR (512 cols)
            _ => (k / 8).max(256).min(512), // Huge matrices: 1/8 QR, 256-512 range
        };
        // QR with column pivoting - ONLY for qr_steps columns
        for step in 0..qr_steps.min(rows) {
            // Find column with largest remaining norm
            let mut max_norm_sq = col_norms_sq[step];
            let mut max_col = step;
            for j in (step + 1)..k {
                if col_norms_sq[j] > max_norm_sq {
                    max_norm_sq = col_norms_sq[j];
                    max_col = j;
                }
            }
            // Swap columns if needed
            if max_col != step {
                self.swap_columns(&mut a, rows, k, step, max_col);
                perm.swap(step, max_col);
                col_norms_sq.swap(step, max_col);
            }
            // Skip if column is effectively zero
            if col_norms_sq[step].sqrt() < self.regularization {
                continue;
            }
            // Apply Householder reflection to current column
            self.apply_householder_step(&mut a, &mut col_norms_sq, rows, k, step)?;
        }
        // optimization: For remaining columns, just sort by L2 norm (much faster)
        if qr_steps < k {
            println!("  Sorting remaining {} columns by L2 norm", k - qr_steps);
            let mut remaining: Vec<(usize, f32)> =
                (qr_steps..k).map(|j| (j, col_norms_sq[j])).collect();
            remaining.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (i, (orig_idx, _)) in remaining.iter().enumerate() {
                perm[qr_steps + i] = *orig_idx;
            }
        }
        Ok(perm)
    }

    /// Swap two columns in the matrix
    fn swap_columns(&self, a: &mut [f32], rows: usize, k: usize, col1: usize, col2: usize) {
        if col1 == col2 {
            return;
        }

        for i in 0..rows {
            let idx1 = i * k + col1;
            let idx2 = i * k + col2;
            a.swap(idx1, idx2);
        }
    }

    /// Apply one step of Householder QR factorization
    fn apply_householder_step(
        &self,
        a: &mut [f32],
        col_norms_sq: &mut [f32],
        rows: usize,
        k: usize,
        step: usize,
    ) -> Result<()> {
        if step >= rows || step >= k {
            return Ok(());
        }

        // Compute Householder vector for column 'step', starting from row 'step'
        let col_start = step;
        let n_rows_remaining = rows - step;

        if n_rows_remaining == 0 {
            return Ok(());
        }

        // Extract the column segment we're working on
        let mut x = vec![0.0f32; n_rows_remaining];
        for i in 0..n_rows_remaining {
            x[i] = a[(step + i) * k + step];
        }

        // Compute Householder vector
        let alpha = if x[0] >= 0.0 {
            -x.iter().map(|&v| v * v).sum::<f32>().sqrt()
        } else {
            x.iter().map(|&v| v * v).sum::<f32>().sqrt()
        };

        if alpha.abs() < self.regularization {
            return Ok(());
        }

        x[0] -= alpha;
        let norm_sq = x.iter().map(|&v| v * v).sum::<f32>();

        if norm_sq < self.regularization {
            return Ok(());
        }

        // Apply Householder reflection to remaining columns
        for j in (step + 1)..k {
            // Compute dot product with column j
            let mut dot = 0.0f32;
            for i in 0..n_rows_remaining {
                dot += x[i] * a[(step + i) * k + j];
            }

            let factor = 2.0 * dot / norm_sq;

            // Apply reflection to column j
            for i in 0..n_rows_remaining {
                a[(step + i) * k + j] -= factor * x[i];
            }

            // Update column norm (for numerical stability)
            if j < col_norms_sq.len() {
                col_norms_sq[j] = 0.0;
                for i in (step + 1)..rows {
                    let val = a[i * k + j];
                    col_norms_sq[j] += val * val;
                }
            }
        }

        Ok(())
    }
}

impl QuantizationStrategy for QRPivotStrategy {
    fn apply_permutation(
        &self,
        data: &[f32],
        rows: usize,
        k: usize,
        tensor_name: &str,
    ) -> Result<(Vec<f32>, Option<Vec<usize>>)> {
        // For small matrices, fall back to L2 norm (QR overhead not worth it)
        if rows < 32 || k < 32 {
            use crate::utils::{
                apply_column_permutation, build_column_permutation, column_l2_norms,
            };
            let norms = column_l2_norms(rows, k, data);
            let perm = build_column_permutation(&norms);
            let permuted = apply_column_permutation(rows, k, data, &perm);
            return Ok((permuted, Some(perm)));
        }

        // Apply QR column pivoting
        match self.qr_column_pivoting(data, rows, k) {
            Ok(perm) => {
                use crate::utils::apply_column_permutation;
                let permuted = apply_column_permutation(rows, k, data, &perm);

                // Log for debugging
                if tensor_name.contains("layers.0.") {
                    println!(
                        "  QR pivot strategy applied to {}: perm[0..5] = {:?}",
                        tensor_name,
                        &perm[..5.min(perm.len())]
                    );
                }

                Ok((permuted, Some(perm)))
            }
            Err(e) => {
                // Fall back to L2 norm on QR failure
                println!(
                    "  QR pivoting failed for {}, falling back to L2: {}",
                    tensor_name, e
                );
                use crate::utils::{
                    apply_column_permutation, build_column_permutation, column_l2_norms,
                };
                let norms = column_l2_norms(rows, k, data);
                let perm = build_column_permutation(&norms);
                let permuted = apply_column_permutation(rows, k, data, &perm);
                Ok((permuted, Some(perm)))
            }
        }
    }

    fn name(&self) -> &'static str {
        "QRPivot"
    }
}
