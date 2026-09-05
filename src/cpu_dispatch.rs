//! CPU dispatches: serial host functions recorded as [`crate::Scheme`] nodes.
//!
//! A CPU dispatch is the host-side counterpart of a compute dispatch. Instead of a
//! shader entry point it runs a stateless Rust function whose parameter list *is* the
//! virtual main: every buffer parcel bound with
//! [`SchemeCpuNodeBuilder::with_parcel`](crate::scheme::SchemeCpuNodeBuilder::with_parcel)
//! arrives as a whole `&[T]` / `&mut [T]` slice (in declaration order), followed by
//! the scalar parameters declared with `with_param`. There is no thread id and no
//! workgroup: the function sees the complete parcel and runs once per submission.
//!
//! ```rust,ignore
//! scheme
//!     .cpu_node("integrate")
//!     .with_parcel(&velocities, NodeAccess::Read)
//!     .with_parcel(&positions, NodeAccess::ReadWrite)
//!     .with_param(dt.to_bits())
//!     .dispatch(|vel: &[f32], pos: &mut [f32], dt: f32| {
//!         for (p, v) in pos.iter_mut().zip(vel) {
//!             *p += v * dt;
//!         }
//!     })?;
//! ```
//!
//! # Execution model
//!
//! Host visibility is a property of the node, not of the bound parcels: the parcels
//! keep their device-resident allocation and the runtime stages them around the host
//! call. For each submission, the submit engine lowers the node into
//!
//! 1. a device→host copy of every parcel the function may read (`Read`, `Write`,
//!    `ReadWrite`) into scheme-owned readback staging,
//! 2. a CPU wait for that copy (and for every cross-context dependency the node has),
//! 3. the host call over the staged bytes,
//! 4. a host→device copy from scheme-owned upload staging back into every parcel the
//!    function may write (`Write`, `ReadWrite`, `Overwrite`).
//!
//! `Overwrite` skips the download: the `&mut [T]` slice arrives zeroed and its final
//! contents replace the parcel. `Write` downloads first so untouched elements keep
//! their previous values.
//!
//! Because step 2 is a fence wait, a CPU dispatch is a full pipeline drain: every
//! GPU node it depends on has completed before it runs, and every GPU node that
//! depends on it starts only after its upload copy. This makes CPU dispatches far
//! more expensive than GPU dispatches — they exist so a host program can move into a
//! scheme one node at a time, not as a steady-state design. Backends with unified
//! memory may later elide the staging copies; the node's contract does not change.
//!
//! The function must be `Fn + Send + Sync + 'static`: it may not hold exclusive
//! mutable state between submissions. Everything it needs comes through its
//! parameters.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use bytemuck::Pod;

use crate::backend::{BufferHandle, GpuCommand};
use crate::buffer::Allocation;
use crate::parcel::Parcel;
use crate::task_graph::{
    BarrierSet, BarrierUsage, NodeAccess, NodeAccessUnion, ResourceId, SlotUsageSet, UsageKindFlags,
};
use crate::timeline::TimelineValue;
use crate::{Context, GoldyError};

/// Largest element alignment a CPU dispatch slice parameter may require.
///
/// Staged bytes are stored in 16-byte aligned blocks, so any `Pod` element type with
/// alignment up to this value can be viewed in place without a copy.
pub const MAX_CPU_ARG_ALIGN: usize = 16;

/// Shape of one virtual-main parameter, reported by [`CpuArg::KIND`].
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArgKind {
    /// A whole-parcel slice.
    Parcel {
        elem_size: usize,
        elem_align: usize,
        /// `&mut [T]` (true) or `&[T]` (false).
        mutable: bool,
    },
    /// A `u32` wire word from `with_param`.
    Scalar,
}

/// Raw argument handed to [`CpuArg::bind`] at execution time.
#[doc(hidden)]
pub enum CpuArgView<'a> {
    /// Staged parcel bytes, 16-byte aligned, length validated against the element size.
    Bytes(&'a mut [u8]),
    /// A scalar wire word.
    Scalar(u32),
}

/// One parameter type of a CPU dispatch virtual main.
///
/// Implemented for `&[T]` and `&mut [T]` where `T: bytemuck::Pod`, and for the scalar
/// wire types `u32`, `i32`, `f32`, and `bool`.
pub trait CpuArg {
    /// The parameter type as seen by the function for a call of lifetime `'a`.
    type Item<'a>;

    #[doc(hidden)]
    const KIND: CpuArgKind;

    #[doc(hidden)]
    fn bind<'a>(view: &'a mut CpuArgView<'_>) -> Self::Item<'a>;
}

impl<T: Pod> CpuArg for &[T] {
    type Item<'a> = &'a [T];

    const KIND: CpuArgKind = CpuArgKind::Parcel {
        elem_size: std::mem::size_of::<T>(),
        elem_align: std::mem::align_of::<T>(),
        mutable: false,
    };

    fn bind<'a>(view: &'a mut CpuArgView<'_>) -> &'a [T] {
        match view {
            CpuArgView::Bytes(bytes) => bytemuck::cast_slice(&**bytes),
            CpuArgView::Scalar(_) => panic!("cpu dispatch: slice parameter bound to a scalar"),
        }
    }
}

impl<T: Pod> CpuArg for &mut [T] {
    type Item<'a> = &'a mut [T];

    const KIND: CpuArgKind = CpuArgKind::Parcel {
        elem_size: std::mem::size_of::<T>(),
        elem_align: std::mem::align_of::<T>(),
        mutable: true,
    };

    fn bind<'a>(view: &'a mut CpuArgView<'_>) -> &'a mut [T] {
        match view {
            CpuArgView::Bytes(bytes) => bytemuck::cast_slice_mut(&mut **bytes),
            CpuArgView::Scalar(_) => panic!("cpu dispatch: slice parameter bound to a scalar"),
        }
    }
}

macro_rules! impl_scalar_cpu_arg {
    ($ty:ty, |$w:ident| $conv:expr) => {
        impl CpuArg for $ty {
            type Item<'a> = $ty;

            const KIND: CpuArgKind = CpuArgKind::Scalar;

            fn bind<'a>(view: &'a mut CpuArgView<'_>) -> $ty {
                match view {
                    CpuArgView::Scalar($w) => {
                        let $w = *$w;
                        $conv
                    }
                    CpuArgView::Bytes(_) => panic!("cpu dispatch: scalar parameter bound to a parcel"),
                }
            }
        }
    };
}

impl_scalar_cpu_arg!(u32, |w| w);
impl_scalar_cpu_arg!(i32, |w| w as i32);
impl_scalar_cpu_arg!(f32, |w| f32::from_bits(w));
impl_scalar_cpu_arg!(bool, |w| w != 0);

/// A function usable as a CPU dispatch virtual main.
///
/// Implemented for every `Fn(A0, .., An) + Send + Sync + 'static` whose parameters
/// are [`CpuArg`]s (up to 16 parameters). Parcel parameters must come first, in the
/// order of the `with_parcel` calls, followed by scalar parameters in the order of
/// the `with_param` calls. `Marker` is inferred; never name it.
pub trait CpuMain<Marker>: Send + Sync + 'static {
    /// Parameter shapes in declaration order.
    #[doc(hidden)]
    fn signature() -> Vec<CpuArgKind>;

    /// Call the function over bound arguments (one view per parameter).
    #[doc(hidden)]
    fn invoke(&self, args: &mut [CpuArgView<'_>]);
}

macro_rules! impl_cpu_main {
    ($(($P:ident, $v:ident)),*) => {
        impl<F, $($P),*> CpuMain<($($P,)*)> for F
        where
            F: Fn($($P),*) + for<'a> Fn($($P::Item<'a>),*) + Send + Sync + 'static,
            $($P: CpuArg,)*
        {
            fn signature() -> Vec<CpuArgKind> {
                vec![$($P::KIND),*]
            }

            #[allow(unused_variables, unused_mut, clippy::too_many_arguments)]
            fn invoke(&self, args: &mut [CpuArgView<'_>]) {
                // Calling `self` directly is ambiguous between the two `Fn` bounds;
                // route through an inner fn that only sees the instantiated one.
                #[allow(clippy::too_many_arguments)]
                fn call_inner<$($P),*>(f: &impl Fn($($P),*), $($v: $P),*) {
                    f($($v),*)
                }
                let mut it = args.iter_mut();
                $(let $v = it.next().expect("cpu dispatch: fewer bound arguments than parameters");)*
                assert!(it.next().is_none(), "cpu dispatch: more bound arguments than parameters");
                call_inner(self, $($P::bind($v)),*);
            }
        }
    };
}

impl_cpu_main!();
impl_cpu_main!((A0, a0));
impl_cpu_main!((A0, a0), (A1, a1));
impl_cpu_main!((A0, a0), (A1, a1), (A2, a2));
impl_cpu_main!((A0, a0), (A1, a1), (A2, a2), (A3, a3));
impl_cpu_main!((A0, a0), (A1, a1), (A2, a2), (A3, a3), (A4, a4));
impl_cpu_main!((A0, a0), (A1, a1), (A2, a2), (A3, a3), (A4, a4), (A5, a5));
impl_cpu_main!((A0, a0), (A1, a1), (A2, a2), (A3, a3), (A4, a4), (A5, a5), (A6, a6));
impl_cpu_main!(
    (A0, a0),
    (A1, a1),
    (A2, a2),
    (A3, a3),
    (A4, a4),
    (A5, a5),
    (A6, a6),
    (A7, a7)
);
impl_cpu_main!(
    (A0, a0),
    (A1, a1),
    (A2, a2),
    (A3, a3),
    (A4, a4),
    (A5, a5),
    (A6, a6),
    (A7, a7),
    (A8, a8)
);
impl_cpu_main!(
    (A0, a0),
    (A1, a1),
    (A2, a2),
    (A3, a3),
    (A4, a4),
    (A5, a5),
    (A6, a6),
    (A7, a7),
    (A8, a8),
    (A9, a9)
);
impl_cpu_main!(
    (A0, a0),
    (A1, a1),
    (A2, a2),
    (A3, a3),
    (A4, a4),
    (A5, a5),
    (A6, a6),
    (A7, a7),
    (A8, a8),
    (A9, a9),
    (A10, a10)
);
impl_cpu_main!(
    (A0, a0),
    (A1, a1),
    (A2, a2),
    (A3, a3),
    (A4, a4),
    (A5, a5),
    (A6, a6),
    (A7, a7),
    (A8, a8),
    (A9, a9),
    (A10, a10),
    (A11, a11)
);
impl_cpu_main!(
    (A0, a0),
    (A1, a1),
    (A2, a2),
    (A3, a3),
    (A4, a4),
    (A5, a5),
    (A6, a6),
    (A7, a7),
    (A8, a8),
    (A9, a9),
    (A10, a10),
    (A11, a11),
    (A12, a12)
);
impl_cpu_main!(
    (A0, a0),
    (A1, a1),
    (A2, a2),
    (A3, a3),
    (A4, a4),
    (A5, a5),
    (A6, a6),
    (A7, a7),
    (A8, a8),
    (A9, a9),
    (A10, a10),
    (A11, a11),
    (A12, a12),
    (A13, a13)
);
impl_cpu_main!(
    (A0, a0),
    (A1, a1),
    (A2, a2),
    (A3, a3),
    (A4, a4),
    (A5, a5),
    (A6, a6),
    (A7, a7),
    (A8, a8),
    (A9, a9),
    (A10, a10),
    (A11, a11),
    (A12, a12),
    (A13, a13),
    (A14, a14)
);
impl_cpu_main!(
    (A0, a0),
    (A1, a1),
    (A2, a2),
    (A3, a3),
    (A4, a4),
    (A5, a5),
    (A6, a6),
    (A7, a7),
    (A8, a8),
    (A9, a9),
    (A10, a10),
    (A11, a11),
    (A12, a12),
    (A13, a13),
    (A14, a14),
    (A15, a15)
);

// ---- Record-time validation ------------------------------------------------

/// Check a virtual-main signature against the recorded bindings and params.
///
/// Returns, per parcel parameter, nothing but an error when the shape does not match;
/// parcel byte sizes are supplied in binding order.
pub(crate) fn validate_signature(
    label: &str,
    signature: &[CpuArgKind],
    bindings: &[(NodeAccess, u64)],
    param_count: usize,
) -> Result<(), GoldyError> {
    let err = |msg: String| GoldyError::Backend(anyhow::anyhow!("cpu_node({label}): {msg}"));

    let parcel_params = signature
        .iter()
        .take_while(|k| matches!(k, CpuArgKind::Parcel { .. }))
        .count();
    let scalar_params = signature.len() - parcel_params;
    if signature[parcel_params..]
        .iter()
        .any(|k| matches!(k, CpuArgKind::Parcel { .. }))
    {
        return Err(err(
            "slice parameters must precede scalar parameters in the virtual main".into(),
        ));
    }
    if parcel_params != bindings.len() {
        return Err(err(format!(
            "virtual main takes {parcel_params} slice parameter(s) but {} parcel(s) were bound",
            bindings.len()
        )));
    }
    if scalar_params != param_count {
        return Err(err(format!(
            "virtual main takes {scalar_params} scalar parameter(s) but {param_count} with_param value(s) were given"
        )));
    }
    for (idx, (kind, (access, byte_size))) in signature.iter().zip(bindings).enumerate() {
        let CpuArgKind::Parcel {
            elem_size,
            elem_align,
            mutable,
        } = *kind
        else {
            unreachable!("parcel prefix established above");
        };
        if elem_size == 0 {
            return Err(err(format!("parameter {idx}: zero-sized element type")));
        }
        if elem_align > MAX_CPU_ARG_ALIGN {
            return Err(err(format!(
                "parameter {idx}: element alignment {elem_align} exceeds {MAX_CPU_ARG_ALIGN}"
            )));
        }
        if byte_size % elem_size as u64 != 0 {
            return Err(err(format!(
                "parameter {idx}: parcel byte size {byte_size} is not a multiple of the element size {elem_size}"
            )));
        }
        if mutable != access.writes() {
            let want = if access.writes() { "&mut [T]" } else { "&[T]" };
            return Err(err(format!(
                "parameter {idx}: bound with {access:?} access but the virtual main does not take {want}"
            )));
        }
    }
    Ok(())
}

// ---- Execution record ------------------------------------------------------

/// 16-byte aligned block so staged bytes can be viewed as any `Pod` slice in place.
#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
struct AlignedBlock([u8; 16]);

unsafe impl bytemuck::Zeroable for AlignedBlock {}
unsafe impl bytemuck::Pod for AlignedBlock {}

/// Zeroed, 16-byte aligned byte storage of exactly `len` bytes.
struct AlignedBytes {
    blocks: Vec<AlignedBlock>,
    len: usize,
}

impl AlignedBytes {
    fn zeroed(len: usize) -> Self {
        let blocks = vec![AlignedBlock::default(); len.div_ceil(16)];
        Self { blocks, len }
    }

    fn as_mut_bytes(&mut self) -> &mut [u8] {
        &mut bytemuck::cast_slice_mut::<AlignedBlock, u8>(&mut self.blocks)[..self.len]
    }

    fn as_bytes(&self) -> &[u8] {
        &bytemuck::cast_slice::<AlignedBlock, u8>(&self.blocks)[..self.len]
    }
}

/// One bound parcel of a CPU dispatch, with the staging it is routed through.
pub(crate) struct CpuBindingExec {
    pub resource: ResourceId,
    /// Parent buffer of the parcel and the parcel's byte window inside it.
    pub buffer: BufferHandle,
    pub offset: u64,
    pub byte_size: u64,
    pub access: NodeAccess,
    /// Device→host staging; `None` for `Overwrite` (nothing to read).
    pub readback: Option<BufferHandle>,
    /// Host→device staging (`CPU_WRITABLE` transient parcel); `None` for `Read`.
    pub upload: Option<Parcel>,
    /// Keeps the parent allocation alive while the scheme references its handle.
    pub _keepalive: Arc<Allocation>,
}

impl fmt::Debug for CpuBindingExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuBindingExec")
            .field("resource", &self.resource)
            .field("access", &self.access)
            .field("byte_size", &self.byte_size)
            .field("downloads", &self.readback.is_some())
            .field("uploads", &self.upload.is_some())
            .finish()
    }
}

/// Side-table entry for one [`crate::task_graph::NodeKind::CpuDispatch`] node.
pub(crate) struct CpuDispatchExec {
    pub label: &'static str,
    main: Box<dyn Fn(&mut [CpuArgView<'_>]) + Send + Sync>,
    pub bindings: Vec<CpuBindingExec>,
    pub params: Vec<u32>,
    /// Timeline of the last submission whose copies referenced this node's staging.
    ready_after: AtomicU64,
}

impl fmt::Debug for CpuDispatchExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuDispatchExec")
            .field("label", &self.label)
            .field("bindings", &self.bindings)
            .field("params", &self.params)
            .field("ready_after", &self.ready_after.load(Ordering::Relaxed))
            .finish()
    }
}

fn transfer_usage(access: NodeAccessUnion) -> SlotUsageSet {
    SlotUsageSet {
        access,
        kinds: UsageKindFlags::TRANSFER,
    }
}

impl CpuDispatchExec {
    pub(crate) fn new<M, F: CpuMain<M>>(
        label: &'static str,
        main: F,
        bindings: Vec<CpuBindingExec>,
        params: Vec<u32>,
    ) -> Self {
        Self {
            label,
            main: Box::new(move |args| main.invoke(args)),
            bindings,
            params,
            ready_after: AtomicU64::new(0),
        }
    }

    /// Timeline value the staging buffers must be settled past before host reuse.
    pub(crate) fn ready_after(&self) -> TimelineValue {
        self.ready_after.load(Ordering::Acquire)
    }

    /// Record that a submission at `tv` referenced this node's staging.
    pub(crate) fn stamp(&self, ctx: crate::backend::ContextHandle, tv: TimelineValue) {
        self.ready_after.fetch_max(tv, Ordering::AcqRel);
        for b in &self.bindings {
            if let Some(upload) = &b.upload {
                upload.mark_referenced(ctx, tv);
            }
        }
    }

    /// Barrier + copies moving every readable binding from its parcel into readback staging.
    ///
    /// `wave_barriers` are the schedule's `barriers_before` for the CPU wave; the
    /// producer side is kept and the consumer side is narrowed to a transfer read,
    /// which is what the copies actually do.
    pub(crate) fn download_commands(&self, wave_barriers: &BarrierSet) -> Vec<GpuCommand> {
        let mut commands = Vec::new();
        let downloaded: Vec<BufferHandle> = self
            .bindings
            .iter()
            .filter(|b| b.readback.is_some())
            .map(|b| b.buffer)
            .collect();
        if downloaded.is_empty() {
            return commands;
        }
        let buffers: Vec<(BufferHandle, BarrierUsage)> = wave_barriers
            .buffers
            .iter()
            .filter(|(h, _)| downloaded.contains(h))
            .map(|(h, usage)| {
                (
                    *h,
                    BarrierUsage {
                        src: usage.src,
                        dst: transfer_usage(NodeAccessUnion::ReadOnly),
                    },
                )
            })
            .collect();
        if !buffers.is_empty() {
            commands.push(GpuCommand::ResourceBarrier {
                buffers,
                textures: Vec::new(),
            });
        }
        for b in &self.bindings {
            if let Some(staging) = b.readback {
                commands.push(GpuCommand::CopyBuffer {
                    src: b.buffer,
                    src_offset: b.offset,
                    dst: staging,
                    dst_offset: 0,
                    size: b.byte_size,
                });
            }
        }
        commands
    }

    /// Barrier + copies moving every writable binding from upload staging into its parcel.
    ///
    /// The producer side of each barrier is the schedule's, widened with the transfer
    /// read performed by [`Self::download_commands`] when the binding was downloaded.
    pub(crate) fn upload_commands(&self, wave_barriers: &BarrierSet) -> Vec<GpuCommand> {
        let mut commands = Vec::new();
        let uploaded: Vec<BufferHandle> = self
            .bindings
            .iter()
            .filter(|b| b.upload.is_some())
            .map(|b| b.buffer)
            .collect();
        if uploaded.is_empty() {
            return commands;
        }
        let mut buffers: Vec<(BufferHandle, BarrierUsage)> = Vec::new();
        for b in &self.bindings {
            if b.upload.is_none() || buffers.iter().any(|(h, _)| *h == b.buffer) {
                continue;
            }
            let mut src = wave_barriers
                .buffers
                .iter()
                .find(|(h, _)| *h == b.buffer)
                .map(|(_, usage)| usage.src)
                .unwrap_or_default();
            if b.readback.is_some() {
                src.merge(NodeAccess::Read, UsageKindFlags::TRANSFER);
            }
            buffers.push((
                b.buffer,
                BarrierUsage {
                    src,
                    dst: transfer_usage(NodeAccessUnion::Write),
                },
            ));
        }
        commands.push(GpuCommand::ResourceBarrier {
            buffers,
            textures: Vec::new(),
        });
        for b in &self.bindings {
            if let Some(upload) = &b.upload {
                let src = upload.buffer_handle().expect("upload staging is a whole buffer");
                commands.push(GpuCommand::CopyBuffer {
                    src,
                    src_offset: 0,
                    dst: b.buffer,
                    dst_offset: b.offset,
                    size: b.byte_size,
                });
            }
        }
        commands
    }

    /// Run the host side: read staged bytes, call the virtual main, write upload staging.
    ///
    /// The caller must have waited for the download copies (and every dependency of
    /// the node) before calling this, and for [`Self::ready_after`] so the upload
    /// staging is no longer read by the previous submission.
    pub(crate) fn run_host(&self, context: &Context) -> Result<(), GoldyError> {
        let _tz = crate::tracy_zone!("goldy.cpu_dispatch.run_host");
        let mut storage: Vec<AlignedBytes> = Vec::with_capacity(self.bindings.len());
        {
            let backend = context.device().inner.backend.lock().unwrap();
            for b in &self.bindings {
                let mut bytes = AlignedBytes::zeroed(b.byte_size as usize);
                if let Some(staging) = b.readback {
                    backend
                        .read_readback_buffer(staging, bytes.as_mut_bytes())
                        .map_err(|e| context.classify(e))?;
                }
                storage.push(bytes);
            }
        }

        {
            let _tz = crate::tracy_zone!("goldy.cpu_dispatch.invoke");
            let mut views: Vec<CpuArgView<'_>> = storage
                .iter_mut()
                .map(|s| CpuArgView::Bytes(s.as_mut_bytes()))
                .chain(self.params.iter().map(|&p| CpuArgView::Scalar(p)))
                .collect();
            (self.main)(&mut views);
        }

        for (b, bytes) in self.bindings.iter().zip(&storage) {
            if let Some(upload) = &b.upload {
                upload
                    .write_bytes(0, bytes.as_bytes())
                    .map_err(|e| context.classify(e))?;
            }
        }
        Ok(())
    }

    /// Release staging once no submission references it any more.
    pub(crate) fn release(self, context: &Context) {
        let ready_after = self.ready_after();
        if ready_after > 0 {
            let _ = context.wait_until(ready_after);
        }
        {
            let mut backend = context.device().inner.backend.lock().unwrap();
            for b in &self.bindings {
                if let Some(staging) = b.readback {
                    backend.free_readback_buffer(staging);
                }
            }
        }
        for b in self.bindings {
            if let Some(mut upload) = b.upload {
                let ready_after = upload.last_referenced();
                upload.release_bookkeeping();
                context.with_transient_pool(|pool| pool.return_buffer_parcel(upload, ready_after));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds<M, F: CpuMain<M>>(_: &F) -> Vec<CpuArgKind> {
        F::signature()
    }

    #[test]
    fn signature_reports_parameter_shapes() {
        let f = |_a: &[f32], _b: &mut [u32], _dt: f32, _flag: bool| {};
        assert_eq!(
            kinds(&f),
            vec![
                CpuArgKind::Parcel {
                    elem_size: 4,
                    elem_align: 4,
                    mutable: false
                },
                CpuArgKind::Parcel {
                    elem_size: 4,
                    elem_align: 4,
                    mutable: true
                },
                CpuArgKind::Scalar,
                CpuArgKind::Scalar,
            ]
        );
        assert!(kinds(&|| {}).is_empty());
    }

    #[test]
    fn invoke_binds_slices_and_scalars_in_order() {
        let f = |src: &[f32], dst: &mut [f32], scale: f32, off: i32| {
            for (d, s) in dst.iter_mut().zip(src) {
                *d = s * scale + off as f32;
            }
        };
        let mut src = AlignedBytes::zeroed(12);
        src.as_mut_bytes()
            .copy_from_slice(bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]));
        let mut dst = AlignedBytes::zeroed(12);
        let mut views = vec![
            CpuArgView::Bytes(src.as_mut_bytes()),
            CpuArgView::Bytes(dst.as_mut_bytes()),
            CpuArgView::Scalar(2.0f32.to_bits()),
            CpuArgView::Scalar((-1i32) as u32),
        ];
        f.invoke(&mut views);
        drop(views);
        let out: &[f32] = bytemuck::cast_slice(dst.as_bytes());
        assert_eq!(out, &[1.0, 3.0, 5.0]);
    }

    #[test]
    fn validate_signature_checks_shape() {
        let ok = kinds(&|_a: &[f32], _b: &mut [u32], _x: u32| {});
        assert!(validate_signature("t", &ok, &[(NodeAccess::Read, 16), (NodeAccess::Write, 8)], 1).is_ok());

        // Wrong parcel count.
        assert!(validate_signature("t", &ok, &[(NodeAccess::Read, 16)], 1).is_err());
        // Wrong scalar count.
        assert!(validate_signature("t", &ok, &[(NodeAccess::Read, 16), (NodeAccess::Write, 8)], 0).is_err());
        // Mutability vs access mismatch.
        assert!(validate_signature("t", &ok, &[(NodeAccess::Write, 16), (NodeAccess::Write, 8)], 1).is_err());
        assert!(validate_signature("t", &ok, &[(NodeAccess::Read, 16), (NodeAccess::Read, 8)], 1).is_err());
        // Byte size not a multiple of the element size.
        assert!(validate_signature("t", &ok, &[(NodeAccess::Read, 6), (NodeAccess::Write, 8)], 1).is_err());
        // Scalars before slices.
        let bad_order = vec![
            CpuArgKind::Scalar,
            CpuArgKind::Parcel {
                elem_size: 4,
                elem_align: 4,
                mutable: false,
            },
        ];
        assert!(validate_signature("t", &bad_order, &[(NodeAccess::Read, 4)], 1).is_err());
    }
}
