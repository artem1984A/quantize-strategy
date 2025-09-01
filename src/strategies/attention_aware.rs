//! Attention layer-aware permutation strategy - copied from working quantize_q8k_copy2.rs

use super::QuantizationStrategy;
use crate::utils::{apply_column_permutation, build_column_permutation, column_l2_norms};
use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Mutex;

static LAYER_PERM_CACHE: Lazy<std::sync::OnceLock<Mutex<LayerPermCache>>> =
    Lazy::new(|| std::sync::OnceLock::new());

static ATTN_PROJ_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$").unwrap());

#[derive(Debug, Clone)]
struct LayerPermCache {
    map: HashMap<u32, Vec<usize>>,
}

impl LayerPermCache {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    fn get_or_compute(&mut self, layer_id: u32, rows: usize, k: usize, data: &[f32]) -> Vec<usize> {
        if let Some(p) = self.map.get(&layer_id) {
            return p.clone();
        }
        let norms = column_l2_norms(rows, k, data);
        let perm = build_column_permutation(&norms);
        self.map.insert(layer_id, perm.clone());
        perm
    }
}

pub struct AttentionAwareStrategy;

impl AttentionAwareStrategy {
    pub fn new() -> Self {
        Self
    }

    fn parse_attention_proj(name: &str) -> Option<(u32, &'static str)> {
        if let Some(caps) = ATTN_PROJ_RE.captures(name) {
            let layer_id: u32 = caps[1].parse().ok()?;
            let kind = match &caps[2] {
                "q" => "q",
                "k" => "k",
                "v" => "v",
                "o" => "o",
                _ => return None,
            };
            Some((layer_id, kind))
        } else {
            None
        }
    }
}

impl QuantizationStrategy for AttentionAwareStrategy {
    fn apply_permutation(
        &self,
        data: &[f32],
        rows: usize,
        k: usize,
        tensor_name: &str,
    ) -> Result<(Vec<f32>, Option<Vec<usize>>)> {
        if let Some((layer_id, kind)) = Self::parse_attention_proj(tensor_name) {
            if kind == "o" {
                // Don't permute output projection
                Ok((data.to_vec(), None))
            } else {
                // Share permutation across q/k/v in same layer
                let perm = {
                    let mut cache = LAYER_PERM_CACHE
                        .get_or_init(|| Mutex::new(LayerPermCache::new()))
                        .lock()
                        .unwrap();
                    cache.get_or_compute(layer_id, rows, k, data)
                };
                let permuted = apply_column_permutation(rows, k, data, &perm);
                Ok((permuted, Some(perm)))
            }
        } else {
            // Non-attention layers: use L2 norm strategy
            let norms = column_l2_norms(rows, k, data);
            let perm = build_column_permutation(&norms);
            let permuted = apply_column_permutation(rows, k, data, &perm);
            Ok((permuted, Some(perm)))
        }
    }

    fn name(&self) -> &'static str {
        "AttentionAware"
    }
}
