//! Permutation utility functions.

pub fn column_l2_norms(rows: usize, k: usize, data: &[f32]) -> Vec<f32> {
    let mut sums: Vec<f64> = vec![0.0; k];
    for r in 0..rows {
        let row = &data[r * k..(r + 1) * k];
        for (j, &v) in row.iter().enumerate() {
            let fv = v as f64;
            sums[j] += fv * fv;
        }
    }
    sums.into_iter().map(|s| (s.sqrt()) as f32).collect()
}

pub fn build_column_permutation(norms: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..norms.len()).collect();
    idx.sort_by(|&a, &b| {
        norms[b]
            .partial_cmp(&norms[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

pub fn apply_column_permutation(rows: usize, k: usize, data: &[f32], perm: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        let src = &data[r * k..(r + 1) * k];
        let dst = &mut out[r * k..(r + 1) * k];
        for j in 0..k {
            dst[j] = src[perm[j]];
        }
    }
    out
}
