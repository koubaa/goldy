//! Backend-selected matrix multiply recorded as a scheme node.

use super::matmul_kernel::{
    GemmBf16Kernel, GemvBf16Kernel, GemvBlockKernel, GemvF32IndexedKernel, GemvF32Kernel, MatMulF32Kernel,
    FLAG_ACCUMULATE, FLAG_TRANSPOSE_A, FLAG_TRANSPOSE_B, GEMV_ROWS_PER_GROUP,
};
use crate::backend::{BufferHandle, ComputePipelineHandle};
use crate::compute::ComputePipeline;
use crate::error::GoldyError;
use crate::runtime::Runtime;
use crate::scheme::{Scheme, SchemeBindable};
use crate::task_graph::NodeAccess;
use crate::types::BackendType;
use std::sync::Arc;

/// Element type for [`MatMulDesc`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MatMulDType {
    /// IEEE-754 binary32 for A, B, and C.
    #[default]
    F32,
    /// Packed BF16 `A`, f32 `B` and `C`. The product promotes each element of `A`.
    BF16,
    /// Block-scaled packed `A`. The block shape and code format live on [`MatMulBlock`].
    BlockScaled,
}

/// Code stored in a block-scaled matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedElem {
    /// Two E2M1 values per byte. The even column is the low nibble.
    E2M1,
    /// One E4M3 value per byte.
    E4M3,
}

/// Scale stored beside a block-scaled matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedScale {
    /// OCP E4M3, one byte per block.
    E4M3,
    /// Power-of-two byte, value `2^(s - 127)`.
    E8M0,
}

/// Block-scaled `A`: codes, one scale per block, optional f32 per-slab multiplier.
///
/// `block` is `[block_rows, block_cols]`. The supported layouts are `[1, 16]`
/// with [`PackedElem::E2M1`] and [`PackedScale::E4M3`], and `[32, 32]` with
/// [`PackedElem::E4M3`] and [`PackedScale::E8M0`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatMulBlock {
    pub elems: PackedElem,
    pub scale: PackedScale,
    pub block_rows: u32,
    pub block_cols: u32,
}

/// Device-selected slab. Slab `i` is `indices[first + i]`.
///
/// An index `>= slabs` yields a zero row and does not read `A` (masked gather).
/// `batch` rows start at `first`. `x_stride == 0` reuses one right-hand side;
/// otherwise row `i` reads `x` at `i * x_stride`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatMulIndex {
    /// Number of slabs in `A`.
    pub slabs: u32,
    /// Elements from the start of one slab to the next.
    pub slab_stride: u32,
    /// When set, the f32 global-scale vector is indexed by the slab index.
    /// Otherwise element 0 is the multiplier for every slab.
    pub global_per_slab: bool,
    /// First element read from the index vector.
    pub first: u32,
    /// How many consecutive indices to gather. `1` is a single slab.
    pub batch: u32,
    /// Element stride between right-hand sides. `0` broadcasts one vector.
    pub x_stride: u32,
}

impl MatMulIndex {
    /// One index at `first`, one shared right-hand side.
    pub const fn single(first: u32, slabs: u32, slab_stride: u32) -> Self {
        Self {
            first,
            slabs,
            slab_stride,
            global_per_slab: false,
            batch: 1,
            x_stride: 0,
        }
    }
}

/// Row-major `C[m, n] = alpha * op(A)[m, k] @ op(B)[k, n] + beta * C[m, n]`.
///
/// `op(X)` is `X` or `Xᵀ` according to the matching transpose flag. Buffers are
/// addressed with [`MatMulView`] (element offset plus optional leading dimension).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatMulDesc {
    /// Rows of `C` and of `op(A)`.
    pub m: u32,
    /// Columns of `C` and of `op(B)`.
    pub n: u32,
    /// Inner dimension shared by `op(A)` and `op(B)`.
    pub k: u32,
    /// Element type of a dense `A`, or [`MatMulDType::BlockScaled`].
    pub dtype: MatMulDType,
    /// Present when `dtype` is [`MatMulDType::BlockScaled`].
    pub block: Option<MatMulBlock>,
    /// Present when `A` is selected by a device id instead of a static offset.
    pub index: Option<MatMulIndex>,
    /// Left-hand transpose.
    pub transpose_a: bool,
    /// Right-hand transpose.
    pub transpose_b: bool,
    /// Scale on the product. Native libraries honor this; the stdlib fallback
    /// currently requires `1.0`.
    pub alpha: f32,
    /// Scale on the existing `C` contents. `0.0` overwrites `C`. The stdlib
    /// product adds onto `C` when this is `1` and `alpha` is `1`.
    pub beta: f32,
}

impl Default for MatMulDesc {
    fn default() -> Self {
        Self {
            m: 0,
            n: 0,
            k: 0,
            dtype: MatMulDType::F32,
            block: None,
            index: None,
            transpose_a: false,
            transpose_b: false,
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

impl MatMulDesc {
    /// Packed row-major GEMM (`C = A @ B`).
    pub fn gemm(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            ..Self::default()
        }
    }

    /// Packed row-major GEMV (`y = A @ x`), i.e. `n = 1`.
    pub fn gemv(rows: u32, inner: u32) -> Self {
        Self::gemm(rows, 1, inner)
    }

    pub(crate) fn packed_lda(&self) -> u32 {
        if self.transpose_a {
            self.m
        } else {
            self.k
        }
    }

    pub(crate) fn packed_ldb(&self) -> u32 {
        if self.transpose_b {
            self.k
        } else {
            self.n
        }
    }

    pub(crate) fn packed_ldc(&self) -> u32 {
        self.n
    }

    pub(crate) fn flags(&self) -> u32 {
        let mut flags = 0u32;
        if self.transpose_a {
            flags |= FLAG_TRANSPOSE_A;
        }
        if self.transpose_b {
            flags |= FLAG_TRANSPOSE_B;
        }
        if self.alpha == 1.0 && self.beta == 1.0 {
            flags |= FLAG_ACCUMULATE;
        }
        flags
    }

    pub(crate) fn validate(&self) -> Result<(), GoldyError> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(GoldyError::Validation(
                "matmul: m, n, and k must be greater than zero".into(),
            ));
        }
        match self.dtype {
            MatMulDType::F32 | MatMulDType::BF16 => {
                if self.block.is_some() {
                    return Err(GoldyError::Validation(
                        "matmul: block metadata is only valid for a block-scaled matrix".into(),
                    ));
                }
            }
            MatMulDType::BlockScaled => {
                let Some(block) = self.block else {
                    return Err(GoldyError::Validation(
                        "matmul: block-scaled matrix is missing its block descriptor".into(),
                    ));
                };
                let nvfp4 = block.elems == PackedElem::E2M1
                    && block.scale == PackedScale::E4M3
                    && block.block_rows == 1
                    && block.block_cols == 16;
                let mxfp8 = block.elems == PackedElem::E4M3
                    && block.scale == PackedScale::E8M0
                    && block.block_rows == 32
                    && block.block_cols == 32;
                if !nvfp4 && !mxfp8 {
                    return Err(GoldyError::Validation(
                        "matmul: block-scaled layout must be E2M1/E4M3 [1, 16] or E4M3/E8M0 [32, 32]".into(),
                    ));
                }
                if !self.k.is_multiple_of(block.block_cols) || !self.m.is_multiple_of(block.block_rows) {
                    return Err(GoldyError::Validation(
                        "matmul: rows and columns must be multiples of the block".into(),
                    ));
                }
            }
        }
        if self.dtype != MatMulDType::F32 && (self.transpose_a || self.transpose_b) {
            return Err(GoldyError::Validation(
                "matmul: BF16 and block-scaled matrices are not transposed".into(),
            ));
        }
        if self.dtype == MatMulDType::BlockScaled && self.n != 1 {
            return Err(GoldyError::Validation(
                "matmul: block-scaled stdlib kernel is a GEMV (n = 1)".into(),
            ));
        }
        if let Some(index) = self.index {
            if index.batch == 0 {
                return Err(GoldyError::Validation("matmul: gather batch is 0".into()));
            }
            if index.batch > 65535 {
                return Err(GoldyError::Validation(
                    "matmul: gather batch exceeds the 65535 workgroup limit".into(),
                ));
            }
            if index.first.checked_add(index.batch).is_none() {
                return Err(GoldyError::Validation(
                    "matmul: first index plus batch overflows u32".into(),
                ));
            }
            let gathered = index.batch != 1 || index.x_stride != 0;
            let f32_gemv = self.dtype == MatMulDType::F32 && self.n == 1 && !self.transpose_a && !self.transpose_b;
            if gathered && !f32_gemv {
                return Err(GoldyError::Validation(
                    "matmul: a run of slab indices or a strided right-hand side is only an f32 GEMV".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn requires_stdlib_identity_epilogue(&self) -> bool {
        self.alpha == 1.0 && self.beta == 0.0
    }
}

impl std::hash::Hash for MatMulDesc {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.m.hash(state);
        self.n.hash(state);
        self.k.hash(state);
        self.dtype.hash(state);
        self.block.hash(state);
        self.index.hash(state);
        self.transpose_a.hash(state);
        self.transpose_b.hash(state);
        self.alpha.to_bits().hash(state);
        self.beta.to_bits().hash(state);
    }
}

/// Element window into a MatMul operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct MatMulView {
    /// Element offset from the start of the bound buffer/parcel.
    pub offset_elements: u64,
    /// Row stride in elements. `None` means packed row-major for the operand's
    /// stored shape (`k`/`m` for A, `n`/`k` for B, `n` for C).
    pub leading_dim: Option<u32>,
}

impl MatMulView {
    /// Packed layout starting at element `offset`.
    pub const fn offset(offset_elements: u64) -> Self {
        Self {
            offset_elements,
            leading_dim: None,
        }
    }

    /// Packed layout at element 0.
    pub const fn packed() -> Self {
        Self::offset(0)
    }

    /// Row-major view with an explicit leading dimension.
    pub const fn strided(offset_elements: u64, leading_dim: u32) -> Self {
        Self {
            offset_elements,
            leading_dim: Some(leading_dim),
        }
    }
}

/// Buffer binding stored on [`NodeKind::MatMul`](crate::task_graph::NodeKind::MatMul).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MatMulOperand {
    pub buffer: BufferHandle,
    pub offset_elements: u64,
    pub leading_dim: u32,
}

/// `GOLDY_MATMUL` override. Unset uses [`MatMulPolicy::Default`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatMulPolicy {
    /// Per-hardware routing from [`MatMulTuning`].
    Default,
    /// Library for every shape (cuBLAS / MPS).
    Library,
    /// Goldy stdlib kernels only.
    Fallback,
}

fn env_policy() -> MatMulPolicy {
    match std::env::var("GOLDY_MATMUL") {
        Ok(v) => match v.to_ascii_lowercase().as_str() {
            "fallback" | "stdlib" | "goldy" => MatMulPolicy::Fallback,
            "library" | "native" | "cublas" | "mps" => MatMulPolicy::Library,
            _ => MatMulPolicy::Default,
        },
        Err(_) => MatMulPolicy::Default,
    }
}

pub(crate) fn backend_has_native(backend: BackendType) -> bool {
    matches!(backend, BackendType::Cuda | BackendType::Metal)
}

/// Goldy stdlib kernel that realizes a non-native MatMul node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MatMulFallback {
    /// General `matmul_f32`: packed operands, one thread per output.
    Gemm = 0,
    /// `gemv_f32`: `n = 1`, no transposes, any leading dimension.
    Gemv = 1,
    /// `gemv_bf16`: packed BF16 `A`, f32 vector, 32 lanes per row.
    GemvBf16 = 2,
    /// `gemm_bf16`: packed BF16 `A`, f32 `B`, one thread per output.
    GemmBf16 = 3,
    /// `gemv_f32_indexed`: device slab id, no copied tile.
    GemvIndexed = 4,
    /// `gemv_block`: block-scaled `A`, 32 lanes per row.
    GemvBlock = 5,
}

impl MatMulFallback {
    pub(crate) fn for_node(desc: &MatMulDesc, a: &MatMulOperand, b: &MatMulOperand, c: &MatMulOperand) -> Self {
        match desc.dtype {
            MatMulDType::BlockScaled => Self::GemvBlock,
            MatMulDType::BF16 => {
                if desc.n == 1 && !desc.transpose_a && !desc.transpose_b {
                    Self::GemvBf16
                } else {
                    Self::GemmBf16
                }
            }
            MatMulDType::F32 if desc.index.is_some() && desc.n == 1 => Self::GemvIndexed,
            MatMulDType::F32 => {
                if gemv_user_slots(desc, a, b, c).is_ok() {
                    Self::Gemv
                } else {
                    Self::Gemm
                }
            }
        }
    }
}

/// GPU family the default matmul routing is tuned for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatMulHardware {
    Cuda,
    /// Apple GPU; `generation` is the M-series number parsed from the adapter name.
    AppleSilicon {
        generation: Option<u32>,
    },
    Other,
}

impl MatMulHardware {
    pub(crate) fn detect(backend: BackendType, adapter_name: &str) -> Self {
        match backend {
            BackendType::Cuda => Self::Cuda,
            BackendType::Metal => Self::AppleSilicon {
                generation: apple_m_generation(adapter_name),
            },
            _ => Self::Other,
        }
    }
}

/// `"Apple M2 Pro"` → `2`.
fn apple_m_generation(adapter_name: &str) -> Option<u32> {
    let rest = adapter_name.strip_prefix("Apple M")?;
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Where the backend library loses to the Goldy stdlib kernels on one GPU family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatMulTuning {
    /// Stdlib-eligible GEMVs (`n = 1`) run on `gemv_f32`.
    pub stdlib_gemv: bool,
    /// GEMMs with `n` below this run on `matmul_f32` when its constraints hold.
    pub library_min_n: u32,
}

/// Apple M1 (T8103), from `llama3.goldy/tools/gemv_bench`: `MPSMatrixMultiplication`
/// costs about `n × 0.7 ms` at 768×768 for `n ≤ 3` (6–18× slower than `gemv_f32` at
/// `n = 1`) and switches to a real GEMM at `n = 4`, where it beats `matmul_f32`.
const APPLE_M1: MatMulTuning = MatMulTuning {
    stdlib_gemv: true,
    library_min_n: 4,
};

impl MatMulTuning {
    pub(crate) fn for_hardware(hardware: MatMulHardware) -> Self {
        match hardware {
            // cuBLAS picks split-K `gemvx` plus a reduce launch for decode-sized GEMVs,
            // which the single-pass stdlib GEMV beats.
            MatMulHardware::Cuda => Self {
                stdlib_gemv: true,
                library_min_n: 1,
            },
            MatMulHardware::AppleSilicon { generation: Some(1) } => APPLE_M1,
            // Unmeasured generations start from M1; retune with `tools/gemv_bench gemm`.
            MatMulHardware::AppleSilicon { .. } => APPLE_M1,
            MatMulHardware::Other => Self {
                stdlib_gemv: false,
                library_min_n: 1,
            },
        }
    }

    /// Hardware defaults with `GOLDY_MATMUL_GEMV=stdlib|library` and
    /// `GOLDY_MATMUL_LIBRARY_MIN_N=<n>` applied.
    pub(crate) fn resolve(hardware: MatMulHardware) -> Self {
        let mut tuning = Self::for_hardware(hardware);
        if let Ok(v) = std::env::var("GOLDY_MATMUL_GEMV") {
            match v.to_ascii_lowercase().as_str() {
                "stdlib" | "goldy" | "fallback" => tuning.stdlib_gemv = true,
                "library" | "native" => tuning.stdlib_gemv = false,
                _ => {}
            }
        }
        if let Some(n) = std::env::var("GOLDY_MATMUL_LIBRARY_MIN_N")
            .ok()
            .and_then(|v| v.parse().ok())
        {
            tuning.library_min_n = n;
        }
        tuning
    }
}

/// Whether a matmul runs on the backend library rather than a Goldy stdlib kernel.
///
/// `stdlib_gemm_ok` says `matmul_f32` can realize a node that is not a stdlib GEMV
/// (identity epilogue, packed operands); the default policy never routes a node to
/// a stdlib kernel that cannot run it.
pub(crate) fn use_native(
    backend: BackendType,
    adapter_name: &str,
    desc: &MatMulDesc,
    fallback: MatMulFallback,
    stdlib_gemm_ok: bool,
) -> bool {
    let tuning = MatMulTuning::resolve(MatMulHardware::detect(backend, adapter_name));
    use_native_with(env_policy(), tuning, backend, desc, fallback, stdlib_gemm_ok)
}

fn use_native_with(
    policy: MatMulPolicy,
    tuning: MatMulTuning,
    backend: BackendType,
    desc: &MatMulDesc,
    fallback: MatMulFallback,
    stdlib_gemm_ok: bool,
) -> bool {
    if !backend_has_native(backend) || desc.dtype != MatMulDType::F32 || desc.index.is_some() {
        return false;
    }
    match policy {
        MatMulPolicy::Fallback => false,
        MatMulPolicy::Library => true,
        MatMulPolicy::Default => match fallback {
            MatMulFallback::Gemv => !tuning.stdlib_gemv,
            MatMulFallback::Gemm => !(stdlib_gemm_ok && desc.n < tuning.library_min_n),
            MatMulFallback::GemvBf16
            | MatMulFallback::GemmBf16
            | MatMulFallback::GemvIndexed
            | MatMulFallback::GemvBlock => false,
        },
    }
}

pub(crate) fn fallback_workgroups(desc: &MatMulDesc, fallback: MatMulFallback) -> (u32, u32, u32) {
    match fallback {
        MatMulFallback::Gemm | MatMulFallback::GemmBf16 => {
            let threads = desc.m.saturating_mul(desc.n);
            (threads.div_ceil(256).max(1), 1, 1)
        }
        MatMulFallback::Gemv | MatMulFallback::GemvBf16 | MatMulFallback::GemvBlock => {
            (desc.m.div_ceil(GEMV_ROWS_PER_GROUP).max(1), 1, 1)
        }
        MatMulFallback::GemvIndexed => {
            let batch = desc.index.map(|i| i.batch.max(1)).unwrap_or(1);
            (desc.m.div_ceil(GEMV_ROWS_PER_GROUP).max(1), batch, 1)
        }
    }
}

pub(crate) fn fallback_user_slots(
    desc: &MatMulDesc,
    fallback: MatMulFallback,
    a: &MatMulOperand,
    b: &MatMulOperand,
    c: &MatMulOperand,
) -> Result<Vec<u32>, GoldyError> {
    match fallback {
        MatMulFallback::Gemm => Ok(gemm_user_slots(desc, a, b, c)?.to_vec()),
        MatMulFallback::Gemv => Ok(gemv_user_slots(desc, a, b, c)?.to_vec()),
        MatMulFallback::GemvBf16 | MatMulFallback::GemmBf16 => Ok(bf16_user_slots(desc, a, b, c)?),
        MatMulFallback::GemvIndexed => Ok(indexed_f32_user_slots(desc, a, b, c)?),
        MatMulFallback::GemvBlock => Ok(block_user_slots(desc, a, b, c)?),
    }
}

fn fit_offset(name: &str, v: u64) -> Result<u32, GoldyError> {
    u32::try_from(v).map_err(|_| GoldyError::Validation(format!("matmul: {name} offset {v} does not fit in u32")))
}

fn gemm_user_slots(
    desc: &MatMulDesc,
    a: &MatMulOperand,
    b: &MatMulOperand,
    c: &MatMulOperand,
) -> Result<[u32; 7], GoldyError> {
    Ok([
        desc.m,
        desc.n,
        desc.k,
        fit_offset("A", a.offset_elements)?,
        fit_offset("B", b.offset_elements)?,
        fit_offset("C", c.offset_elements)?,
        desc.flags(),
    ])
}

/// `gemv_f32` scalars: `m, k, lda, x_stride, y_stride, a_off, x_off, y_off`.
fn gemv_user_slots(
    desc: &MatMulDesc,
    a: &MatMulOperand,
    b: &MatMulOperand,
    c: &MatMulOperand,
) -> Result<[u32; 8], GoldyError> {
    if desc.n != 1 || desc.transpose_a || desc.transpose_b || !desc.requires_stdlib_identity_epilogue() {
        return Err(GoldyError::Validation(
            "matmul: gemv kernel needs n = 1, no transposes, alpha = 1, beta = 0".into(),
        ));
    }
    Ok([
        desc.m,
        desc.k,
        a.leading_dim,
        b.leading_dim,
        c.leading_dim,
        fit_offset("A", a.offset_elements)?,
        fit_offset("B", b.offset_elements)?,
        fit_offset("C", c.offset_elements)?,
    ])
}

pub(crate) fn packed_leading_dim(desc: &MatMulDesc, which: OperandKind) -> u32 {
    match which {
        OperandKind::A => desc.packed_lda(),
        OperandKind::B => desc.packed_ldb(),
        OperandKind::C => desc.packed_ldc(),
    }
}

#[derive(Clone, Copy)]
pub(crate) enum OperandKind {
    A,
    B,
    C,
}

/// Builder returned by [`Scheme::matmul`](crate::Scheme::matmul).
pub struct MatMulBuilder<'a> {
    scheme: &'a mut Scheme,
    label: crate::SchemeLabel,
    desc: MatMulDesc,
    a: Option<BoundOperand>,
    b: Option<BoundOperand>,
    c: Option<BoundOperand>,
    extras: Vec<BoundOperand>,
}

pub(crate) struct BoundOperand {
    pub operand: MatMulOperand,
    pub resource: crate::task_graph::ResourceId,
    pub slot: u32,
    /// The operand's buffer as a semantic parcel, when `slot` is its whole-buffer view.
    pub parcel: Option<crate::semantic_fusion::SiteParcel>,
}

impl<'a> MatMulBuilder<'a> {
    pub(crate) fn new(scheme: &'a mut Scheme, label: crate::SchemeLabel, desc: MatMulDesc) -> Self {
        Self {
            scheme,
            label,
            desc,
            a: None,
            b: None,
            c: None,
            extras: Vec::new(),
        }
    }

    /// Left-hand matrix `A`.
    #[allow(private_bounds)]
    pub fn a(mut self, buf: &impl SchemeBindable, view: MatMulView) -> Self {
        match bind_operand(
            self.scheme,
            buf,
            NodeAccess::Read,
            view,
            packed_leading_dim(&self.desc, OperandKind::A),
        ) {
            Ok(bound) => self.a = Some(bound),
            Err(msg) => self.scheme.push_record_error(msg),
        }
        self
    }

    /// Right-hand matrix or vector `B`.
    #[allow(private_bounds)]
    pub fn b(mut self, buf: &impl SchemeBindable, view: MatMulView) -> Self {
        match bind_operand(
            self.scheme,
            buf,
            NodeAccess::Read,
            view,
            packed_leading_dim(&self.desc, OperandKind::B),
        ) {
            Ok(bound) => self.b = Some(bound),
            Err(msg) => self.scheme.push_record_error(msg),
        }
        self
    }

    /// An extra read-only buffer after `C`. Block-scaled and indexed kernels bind
    /// scales, a global multiplier, or the id vector here, in kernel order.
    #[allow(private_bounds)]
    pub fn extra(mut self, buf: &impl SchemeBindable, view: MatMulView) -> Self {
        match bind_operand(self.scheme, buf, NodeAccess::Read, view, 1) {
            Ok(bound) => self.extras.push(bound),
            Err(msg) => self.scheme.push_record_error(msg),
        }
        self
    }

    /// Output matrix or vector `C`.
    #[allow(private_bounds)]
    pub fn out(mut self, buf: &impl SchemeBindable, view: MatMulView) -> Self {
        let access = if self.desc.beta == 0.0 {
            NodeAccess::Overwrite
        } else {
            NodeAccess::ReadWrite
        };
        match bind_operand(
            self.scheme,
            buf,
            access,
            view,
            packed_leading_dim(&self.desc, OperandKind::C),
        ) {
            Ok(bound) => self.c = Some(bound),
            Err(msg) => self.scheme.push_record_error(msg),
        }
        self
    }

    /// Append the MatMul node. Realization (cuBLAS / MPS / stdlib) happens on first submit.
    pub fn record(mut self) {
        if let Err(msg) = self.desc.validate() {
            self.scheme.push_record_error(msg.to_string());
            return;
        }
        let Some(a) = self.a else {
            self.scheme
                .push_record_error(format!("matmul `{}`: missing .a()", self.label));
            return;
        };
        let Some(b) = self.b else {
            self.scheme
                .push_record_error(format!("matmul `{}`: missing .b()", self.label));
            return;
        };
        let Some(c) = self.c else {
            self.scheme
                .push_record_error(format!("matmul `{}`: missing .out()", self.label));
            return;
        };
        let fallback = MatMulFallback::for_node(&self.desc, &a.operand, &b.operand, &c.operand);
        let operands = [
            ("A", &a.operand, packed_leading_dim(&self.desc, OperandKind::A)),
            ("B", &b.operand, packed_leading_dim(&self.desc, OperandKind::B)),
            ("C", &c.operand, packed_leading_dim(&self.desc, OperandKind::C)),
        ];
        let stdlib_gemm_ok = self.desc.requires_stdlib_identity_epilogue()
            && operands.iter().all(|(_, op, packed)| op.leading_dim == *packed);
        let native = use_native(
            self.scheme.backend_type(),
            &self.scheme.adapter_name(),
            &self.desc,
            fallback,
            stdlib_gemm_ok,
        );
        let stdlib_epilogue = self.desc.alpha == 1.0
            && (self.desc.beta == 0.0
                || (self.desc.beta == 1.0 && self.desc.dtype == MatMulDType::F32 && self.desc.index.is_none()));
        if !native && !stdlib_epilogue {
            self.scheme.push_record_error(format!(
                "matmul `{}`: stdlib fallback requires alpha=1 and beta=0, or beta=1 on an f32 product",
                self.label
            ));
            return;
        }
        if fallback == MatMulFallback::GemvIndexed && self.desc.dtype == MatMulDType::F32 {
            let index_off = self
                .extras
                .first()
                .map(|extra| extra.operand.offset_elements)
                .unwrap_or(0);
            let words = match indexed_gather_words(&self.desc, &a.operand, &b.operand, &c.operand, index_off) {
                Ok(words) => words,
                Err(err) => {
                    self.scheme.push_record_error(err.to_string());
                    return;
                }
            };
            let params = match self.scheme.alloc_matmul_params(&words) {
                Ok(buf) => buf,
                Err(err) => {
                    self.scheme.push_record_error(err.to_string());
                    return;
                }
            };
            match bind_operand(self.scheme, &params, NodeAccess::Read, MatMulView::packed(), 1) {
                Ok(bound) => self.extras.push(bound),
                Err(msg) => {
                    self.scheme.push_record_error(msg);
                    return;
                }
            }
            self.scheme.retain_matmul_params(params);
        }
        if !native && fallback == MatMulFallback::Gemm {
            for (name, op, packed) in operands {
                if op.leading_dim != packed {
                    self.scheme.push_record_error(format!(
                        "matmul `{}`: stdlib fallback requires packed leading dim {packed} for {name}, got {}",
                        self.label, op.leading_dim
                    ));
                    return;
                }
            }
        }
        let c_access = if self.desc.beta == 0.0 {
            NodeAccess::Overwrite
        } else {
            NodeAccess::ReadWrite
        };
        let site = || {
            let operands = [
                (&a.operand, a.parcel?),
                (&b.operand, b.parcel?),
                (&c.operand, c.parcel?),
            ];
            let ids = match fallback {
                MatMulFallback::GemvIndexed => self
                    .extras
                    .first()
                    .and_then(|extra| extra.parcel.map(|parcel| (&extra.operand, parcel))),
                _ => None,
            };
            match native {
                true => crate::semantic_fusion::library_matmul(&self.desc, fallback, operands),
                false => crate::semantic_fusion::matmul(&self.desc, fallback, operands, ids),
            }
        };
        let site = site();
        self.scheme.push_matmul_node(
            self.label,
            self.desc,
            a,
            b,
            c,
            self.extras,
            c_access,
            native,
            fallback,
            site,
        );
    }
}

fn bind_operand(
    scheme: &mut Scheme,
    bindable: &impl SchemeBindable,
    access: NodeAccess,
    view: MatMulView,
    packed_ld: u32,
) -> Result<BoundOperand, String> {
    let descriptor_access = match access {
        NodeAccess::Read => crate::types::ResourceAccess::Read,
        NodeAccess::Write | NodeAccess::Overwrite => crate::types::ResourceAccess::Write,
        NodeAccess::ReadWrite => crate::types::ResourceAccess::ReadWrite,
    };
    let (resource_identity, slot) = bindable.resolve(scheme, descriptor_access);
    let slot = slot.ok_or_else(|| format!("matmul: resource has no descriptor for {access:?} access"))?;
    let (resource, maybe_stamp) =
        resource_identity.ok_or_else(|| "matmul: bindable has no resource identity".to_string())?;
    if let Some(stamp) = maybe_stamp {
        scheme.register_stamp(resource, stamp);
    }
    let (buffer, base_bytes) = match resource {
        crate::task_graph::ResourceId::Buffer(h) => (h, 0u64),
        crate::task_graph::ResourceId::BufferRange { parent, offset, .. } => (parent, offset),
        _ => return Err("matmul: operands must be buffer parcels".into()),
    };
    if base_bytes % 4 != 0 {
        return Err(format!("matmul: buffer range offset {base_bytes} is not f32-aligned"));
    }
    let offset_elements = base_bytes / 4 + view.offset_elements;
    let leading_dim = view.leading_dim.unwrap_or(packed_ld);
    if leading_dim == 0 {
        return Err("matmul: leading dimension must be greater than zero".into());
    }
    let parcel = bindable
        .buffer_parcel()
        .and_then(|p| crate::semantic_fusion::SiteParcel::of(&p))
        .filter(|p| p.buffer == buffer && (p.srv == Some(slot) || p.uav == Some(slot)));
    Ok(BoundOperand {
        operand: MatMulOperand {
            buffer,
            offset_elements,
            leading_dim,
        },
        resource,
        slot,
        parcel,
    })
}

pub(crate) struct StdlibMatMulPipelines {
    kernel: crate::kernel::PreparedKernel,
}

impl StdlibMatMulPipelines {
    pub(crate) fn select(&self, policy: crate::ArithmeticPolicy) -> anyhow::Result<Arc<ComputePipeline>> {
        self.kernel.try_pipeline_arc_for(policy)
    }
}

pub(crate) fn prepare_stdlib(
    runtime: &Runtime,
    kind: MatMulFallback,
    policy: crate::ArithmeticPolicy,
) -> anyhow::Result<StdlibMatMulPipelines> {
    macro_rules! prepare {
        ($kernel:ty) => {{
            let kernel = <$kernel>::prepare_for_policy(runtime, policy).map_err(|e| anyhow::anyhow!("{e}"))?;
            StdlibMatMulPipelines {
                kernel: kernel.into_prepared(),
            }
        }};
    }
    Ok(match kind {
        MatMulFallback::Gemm => prepare!(MatMulF32Kernel),
        MatMulFallback::Gemv => prepare!(GemvF32Kernel),
        MatMulFallback::GemvBf16 => prepare!(GemvBf16Kernel),
        MatMulFallback::GemmBf16 => prepare!(GemmBf16Kernel),
        MatMulFallback::GemvIndexed => prepare!(GemvF32IndexedKernel),
        MatMulFallback::GemvBlock => prepare!(GemvBlockKernel),
    })
}

fn bf16_user_slots(
    desc: &MatMulDesc,
    a: &MatMulOperand,
    _b: &MatMulOperand,
    _c: &MatMulOperand,
) -> Result<Vec<u32>, GoldyError> {
    let (slabs, first, stride, a_off) = index_slots(desc, a)?;
    if desc.n == 1 {
        Ok(vec![desc.m, desc.k, a_off, stride, slabs, first])
    } else {
        Ok(vec![desc.m, desc.n, desc.k, a_off, stride, slabs, first])
    }
}

fn indexed_f32_user_slots(
    desc: &MatMulDesc,
    a: &MatMulOperand,
    b: &MatMulOperand,
    c: &MatMulOperand,
) -> Result<Vec<u32>, GoldyError> {
    // Sizes travel in a parameter buffer. Confirm they fit, then leave the push block empty.
    indexed_gather_words(desc, a, b, c, 0)?;
    Ok(vec![])
}

/// `m, k, lda, slab_stride, slabs, first, batch, x_stride, a_base, x_base, y_base, index_base, x_elem, y_elem`.
pub(crate) fn indexed_gather_words(
    desc: &MatMulDesc,
    a: &MatMulOperand,
    b: &MatMulOperand,
    c: &MatMulOperand,
    index_off: u64,
) -> Result<[u32; 14], GoldyError> {
    let index = desc
        .index
        .ok_or_else(|| GoldyError::Validation("matmul: indexed GEMV is missing its index".into()))?;
    Ok([
        desc.m,
        desc.k,
        a.leading_dim,
        index.slab_stride,
        index.slabs,
        index.first,
        index.batch,
        index.x_stride,
        fit_offset("A", a.offset_elements)?,
        fit_offset("x", b.offset_elements)?,
        fit_offset("out", c.offset_elements)?,
        fit_offset("indices", index_off)?,
        b.leading_dim.max(1),
        c.leading_dim.max(1),
    ])
}

fn block_user_slots(
    desc: &MatMulDesc,
    a: &MatMulOperand,
    _b: &MatMulOperand,
    _c: &MatMulOperand,
) -> Result<Vec<u32>, GoldyError> {
    let block = desc
        .block
        .ok_or_else(|| GoldyError::Validation("matmul: missing block descriptor".into()))?;
    let (slabs, first, _stride, _off) = index_slots(desc, a)?;
    let elem = match block.elems {
        PackedElem::E2M1 => 0,
        PackedElem::E4M3 => 1,
    };
    let scale = match block.scale {
        PackedScale::E4M3 => 0,
        PackedScale::E8M0 => 1,
    };
    let per_slab = u32::from(desc.index.is_some_and(|i| i.global_per_slab));
    let rows = desc.index.map(|i| i.slab_stride).unwrap_or(desc.m);
    Ok(vec![
        desc.m,
        desc.k,
        elem | (scale << 1),
        block.block_cols,
        slabs,
        first,
        rows,
        per_slab,
    ])
}

fn index_slots(desc: &MatMulDesc, a: &MatMulOperand) -> Result<(u32, u32, u32, u32), GoldyError> {
    match desc.index {
        Some(index) => Ok((index.slabs, index.first, index.slab_stride, 0)),
        None => Ok((0, 0, 0, fit_offset("A", a.offset_elements)?)),
    }
}

/// Node payload stored in the task graph.
#[derive(Debug, Clone)]
pub(crate) struct MatMulNode {
    pub desc: MatMulDesc,
    pub a: MatMulOperand,
    pub b: MatMulOperand,
    pub c: MatMulOperand,
    pub resource_slots: Vec<u32>,
    pub native: bool,
    pub fallback: MatMulFallback,
    pub fallback_pipeline: Option<ComputePipelineHandle>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gathered_batch_is_only_an_f32_gemv() {
        let mut desc = MatMulDesc::gemv(4, 8);
        desc.dtype = MatMulDType::BF16;
        desc.index = Some(MatMulIndex {
            first: 0,
            slabs: 4,
            slab_stride: 32,
            global_per_slab: false,
            batch: 3,
            x_stride: 0,
        });
        assert!(desc.validate().is_err());
        desc.dtype = MatMulDType::F32;
        desc.index.as_mut().unwrap().batch = 65536;
        assert!(desc.validate().is_err());
        desc.index.as_mut().unwrap().batch = 2;
        desc.index.as_mut().unwrap().first = u32::MAX;
        assert!(desc.validate().is_err());
        desc.index.as_mut().unwrap().first = 0;
        assert!(desc.validate().is_ok());
    }

    fn operand(offset_elements: u64, leading_dim: u32) -> MatMulOperand {
        MatMulOperand {
            buffer: BufferHandle::default(),
            offset_elements,
            leading_dim,
        }
    }

    #[test]
    fn gemv_fallback_covers_strided_identity_gemv_only() {
        let (a, b, c) = (operand(4, 9), operand(0, 1), operand(2, 3));
        let gemv = MatMulDesc::gemv(5, 7);
        assert_eq!(MatMulFallback::for_node(&gemv, &a, &b, &c), MatMulFallback::Gemv);
        assert_eq!(gemv_user_slots(&gemv, &a, &b, &c).unwrap(), [5, 7, 9, 1, 3, 4, 0, 2]);

        let gemm = MatMulDesc::gemm(5, 2, 7);
        let transposed = MatMulDesc {
            transpose_a: true,
            ..gemv
        };
        let scaled = MatMulDesc { beta: 1.0, ..gemv };
        for desc in [gemm, transposed, scaled] {
            assert_eq!(
                MatMulFallback::for_node(&desc, &a, &b, &c),
                MatMulFallback::Gemm,
                "{desc:?}"
            );
        }
        let far = operand(u64::from(u32::MAX) + 1, 9);
        assert_eq!(MatMulFallback::for_node(&gemv, &far, &b, &c), MatMulFallback::Gemm);
    }

    #[test]
    fn default_routing_follows_hardware_tuning() {
        use MatMulFallback::{Gemm, Gemv};
        use MatMulPolicy::{Default, Fallback, Library};
        let route = |policy, backend, name: &str, n, fallback, stdlib_ok| {
            let tuning = MatMulTuning::for_hardware(MatMulHardware::detect(backend, name));
            use_native_with(
                policy,
                tuning,
                backend,
                &MatMulDesc::gemm(768, n, 768),
                fallback,
                stdlib_ok,
            )
        };
        let (cuda, metal, m1) = (BackendType::Cuda, BackendType::Metal, "Apple M1");
        assert!(!route(Default, cuda, "RTX", 1, Gemv, true));
        assert!(route(Default, cuda, "RTX", 2, Gemm, true));
        assert!(!route(Default, metal, m1, 1, Gemv, true));
        assert!(!route(Default, metal, m1, 3, Gemm, true));
        assert!(route(Default, metal, m1, 3, Gemm, false), "stdlib cannot run it");
        assert!(route(Default, metal, m1, 4, Gemm, true));
        assert!(!route(Default, metal, "Apple M4 Max", 1, Gemv, true));
        assert!(route(Library, metal, m1, 1, Gemv, true));
        assert!(!route(Fallback, cuda, "RTX", 64, Gemm, true));
        assert!(!route(Library, BackendType::Vulkan, "any", 64, Gemm, true));
    }

    #[test]
    fn apple_generation_parses_adapter_names() {
        assert_eq!(apple_m_generation("Apple M1"), Some(1));
        assert_eq!(apple_m_generation("Apple M2 Pro"), Some(2));
        assert_eq!(apple_m_generation("Apple M10 Ultra"), Some(10));
        assert_eq!(apple_m_generation("AMD Radeon Pro 5500M"), None);
    }
}
