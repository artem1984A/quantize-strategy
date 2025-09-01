//! File I/O operations for Q8K format.

use super::header::{Q8KHeader, DTYPE_Q8K, MAGIC_Q8K, VERSION};
use anyhow::{bail, Result};
use candle_core::quantized::k_quants::{BlockQ8K, QK_K};
use candle_core::quantized::GgmlType;
use std::fs;
use std::io::{BufWriter, Write};
use std::mem;
use std::path::Path;

pub fn write_q8k(path: &Path, rows: usize, k: usize, blocks: &[BlockQ8K]) -> Result<()> {
    let header = Q8KHeader {
        magic: MAGIC_Q8K,
        version: VERSION,
        out: rows as u32,
        k: k as u32,
        blocks_per_row: (k / QK_K) as u32,
        dtype: DTYPE_Q8K,
    };
    let mut w = BufWriter::new(fs::File::create(path)?);
    w.write_all(bytemuck::bytes_of(&header))?;
    let raw = unsafe {
        std::slice::from_raw_parts(
            blocks.as_ptr() as *const u8,
            blocks.len() * mem::size_of::<BlockQ8K>(),
        )
    };
    w.write_all(raw)?;
    w.flush()?;
    Ok(())
}

pub fn write_perm(path_q8k: &Path, perm: &[usize]) -> Result<()> {
    let mut p = path_q8k.to_path_buf();
    p.set_extension("perm");
    let mut w = BufWriter::new(fs::File::create(&p)?);
    const MAGIC_PERM: u32 = 0x4D52_4550; // "PERM"
    w.write_all(&MAGIC_PERM.to_le_bytes())?;
    w.write_all(&(perm.len() as u32).to_le_bytes())?;
    for &u in perm {
        w.write_all(&(u as u32).to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

pub fn load_perm(path_q8k: &Path) -> Result<Option<Vec<usize>>> {
    let mut p = path_q8k.to_path_buf();
    p.set_extension("perm");
    if !p.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&p)?;
    if bytes.len() < 8 {
        bail!("perm file too small: {}", p.display());
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    const MAGIC_PERM: u32 = 0x4D52_4550;
    if magic != MAGIC_PERM {
        bail!("bad perm magic in {}", p.display());
    }
    let k = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let expect = 8 + 4 * k;
    if bytes.len() != expect {
        bail!(
            "perm size mismatch {} (got {}, expect {})",
            p.display(),
            bytes.len(),
            expect
        );
    }
    let mut perm = Vec::with_capacity(k);
    for i in 0..k {
        let off = 8 + 4 * i;
        let idx = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        perm.push(idx);
    }
    Ok(Some(perm))
}

pub fn load_q8k_tensor(path: &Path) -> Result<(Vec<BlockQ8K>, usize, usize, Option<Vec<usize>>)> {
    let data = fs::read(path)?;
    if data.len() < mem::size_of::<Q8KHeader>() {
        bail!("file too small: {}", path.display());
    }

    let hdr = *bytemuck::from_bytes::<Q8KHeader>(&data[..mem::size_of::<Q8KHeader>()]);
    if hdr.magic != MAGIC_Q8K {
        bail!("bad magic in {}", path.display());
    }
    if hdr.dtype != DTYPE_Q8K {
        bail!("unexpected dtype in {}", path.display());
    }

    let total_blocks = (hdr.out as usize) * (hdr.blocks_per_row as usize);
    let expected = mem::size_of::<Q8KHeader>() + total_blocks * mem::size_of::<BlockQ8K>();
    if data.len() != expected {
        bail!("size mismatch in {}", path.display());
    }

    let mut blocks = vec![BlockQ8K::zeros(); total_blocks];
    let raw = &data[mem::size_of::<Q8KHeader>()..];
    unsafe {
        std::ptr::copy_nonoverlapping(raw.as_ptr(), blocks.as_mut_ptr() as *mut u8, raw.len());
    }

    let perm = load_perm(path).ok().flatten();

    Ok((blocks, hdr.out as usize, hdr.k as usize, perm))
}
