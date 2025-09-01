//! Q8K file format header definitions.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct Q8KHeader {
    pub magic: u32,
    pub version: u32,
    pub out: u32,
    pub k: u32,
    pub blocks_per_row: u32,
    pub dtype: u32,
}

pub const MAGIC_Q8K: u32 = 0x4B51_3838; // "KQ88" little-endian
pub const VERSION: u32 = 1;
pub const DTYPE_Q8K: u32 = 0x18; // BlockQ8K format identifier
