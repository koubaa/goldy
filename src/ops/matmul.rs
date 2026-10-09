//! Backend-selected matrix multiply recorded as a scheme node.

use super::matmul_kernel::{GemvF32Kernel, MatMulF32Kernel, FLAG_TRANSPOSE_A, FLAG_TRANSPOSE_B, GEMV_ROWS_PER_GROUP};
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
    /// IEEE-754 binary32. The only type in the first slice.
    #[default]
    F32,
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
    /// Input and output element type.
    pub dtype: MatMulDType,
    /// Left-hand transpose.
    pub transpose_a: bool,
    /// Right-hand transpose.
    pub transpose_b: bool,
    /// Scale on the product. Native libraries honor this; the stdlib fallback
    /// currently requires `1.0`.
    pub alpha: f32,
    /// Scale on the existing `C` contents. `0.0` means `C` is overwritten.
    /// The stdlib fallback currently requires `0.0`.
    pub beta: f32,
}

impl Default for MatMulDesc {
    fn default() -> Self {
        Self {
            m: 0,
            n: 0,
            k: 0,
            dtype: MatMulDType::F32,
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
        flags
    }

    pub(crate) fn validate(&self) -> Result<(), GoldyError> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return Err(GoldyError::Validation(
                "matmul: m, n, and k must be greater than zero".into(),
            ));
        }
        if self.dtype != MatMulDType::F32 {
            return Err(GoldyError::Validation("matmul: only F32 is supported".into()));
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
}

impl MatMulFallback {
    pub(crate) fn for_node(desc: &MatMulDesc, a: &MatMulOperand, b: &MatMulOperand, c: &MatMulOperand) -> Self {
        if gemv_user_slots(desc, a, b, c).is_ok() {
            Self::Gemv
        } else {
            Self::Gemm
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
    if !backend_has_native(backend) {
        return false;
    }
    match policy {
        MatMulPolicy::Fallback => false,
        MatMulPolicy::Library => true,
        MatMulPolicy::Default => match fallback {
            MatMulFallback::Gemv => !tuning.stdlib_gemv,
            MatMulFallback::Gemm => !(stdlib_gemm_ok && desc.n < tuning.library_min_n),
        },
    }
}

pub(crate) fn fallback_workgroups(desc: &MatMulDesc, fallback: MatMulFallback) -> (u32, u32, u32) {
    match fallback {
        MatMulFallback::Gemm => {
            let threads = desc.m.saturating_mul(desc.n);
            (threads.div_ceil(256).max(1), 1, 1)
        }
        MatMulFallback::Gemv => (desc.m.div_ceil(GEMV_ROWS_PER_GROUP).max(1), 1, 1),
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
    pub fn record(self) {
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
        if !native && !self.desc.requires_stdlib_identity_epilogue() {
            self.scheme.push_record_error(format!(
                "matmul `{}`: stdlib fallback requires alpha=1 and beta=0",
                self.label
            ));
            return;
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
            match native {
                true => crate::semantic_fusion::library_matmul(&self.desc, fallback, operands),
                false => crate::semantic_fusion::matmul(&self.desc, fallback, operands),
            }
        };
        let site = site();
        self.scheme
            .push_matmul_node(self.label, self.desc, a, b, c, c_access, native, fallback, site);
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

pub(crate) fn prepare_stdlib(runtime: &Runtime, kind: MatMulFallback) -> anyhow::Result<Arc<ComputePipeline>> {
    match kind {
        MatMulFallback::Gemm => {
            let kernel = MatMulF32Kernel::prepare(runtime).map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(kernel.pipeline_arc())
        }
        MatMulFallback::Gemv => {
            let kernel = GemvF32Kernel::prepare(runtime).map_err(|e| anyhow::anyhow!("{e}"))?;
            Ok(kernel.pipeline_arc())
        }
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
