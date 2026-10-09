//! What a fused kernel and the dispatches it replaces cost on a device.
//!
//! Lowering counts what a synthesized kernel does ([`Estimate`]); this model prices
//! the counts. The planner fuses the prefix of a run that saves the most estimated
//! time, and leaves a run unfused when no prefix saves any.

use crate::semantic_fusion::SemanticSite;
use goldy_shader_ir::algebra::{Estimate, OpKind};

/// How fast a device runs what an [`Estimate`] counts, and how fast its native
/// library runs matrix products; see [`crate::Scheme::set_fusion_cost_model`].
///
/// A kernel's time is its dispatch cost plus the slowest of its storage traffic, its
/// longest thread (its steps and the reads they wait on), its total work and its
/// matrix-unit work, each at its rate. The
/// estimate is coarse: it decides whether a fused kernel beats the dispatches it
/// replaces, and is meant to be wrong only where the two are close.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusionCostModel {
    /// Fixed cost of one dispatch, including the barrier the next dispatch waits on.
    pub dispatch_ns: f64,
    /// Storage bandwidth.
    pub bytes_per_ns: f64,
    /// One dependent step of one thread.
    pub step_ns: f64,
    /// One storage read a thread waits on, such as in each iteration of a loop.
    pub load_ns: f64,
    /// Operations over all threads.
    pub ops_per_ns: f64,
    /// Multiply-adds a synthesized kernel runs on matrix units.
    pub matrix_macs_per_ns: f64,
    /// Fixed cost of one native-library matrix product, which may take several
    /// dispatches (a split-summation product and the reduction of its parts).
    pub library_dispatch_ns: f64,
    /// Multiply-adds of a matrix product the native library runs.
    pub library_macs_per_ns: f64,
}

impl Default for FusionCostModel {
    /// A discrete desktop GPU of about 290 GB/s and 20 FP32 TFLOP/s, such as an
    /// RTX 4060 Ti, where a one-workgroup kernel takes about 1.4 µs, a read that
    /// hits the L2 cache about 90 ns, and a small cuBLAS product about 5 µs.
    fn default() -> Self {
        Self {
            dispatch_ns: 1500.0,
            bytes_per_ns: 288.0,
            step_ns: 1.5,
            load_ns: 90.0,
            ops_per_ns: 8000.0,
            matrix_macs_per_ns: 2000.0,
            library_dispatch_ns: 5000.0,
            library_macs_per_ns: 8000.0,
        }
    }
}

impl FusionCostModel {
    /// One dispatch of a kernel that does what `estimate` counts.
    pub fn kernel_ns(&self, estimate: &Estimate) -> f64 {
        let busy = [
            estimate.bytes as f64 / self.bytes_per_ns,
            estimate.serial as f64 * self.step_ns + estimate.loads as f64 * self.load_ns,
            estimate.work as f64 / self.ops_per_ns,
            estimate.matrix as f64 / self.matrix_macs_per_ns,
        ];
        self.dispatch_ns + busy.into_iter().fold(0.0, f64::max)
    }

    /// One native-library matrix product of `macs` multiply-adds over `bytes` of
    /// operands.
    pub fn library_ns(&self, macs: u64, bytes: u64) -> f64 {
        self.library_dispatch_ns + (bytes as f64 / self.bytes_per_ns).max(macs as f64 / self.library_macs_per_ns)
    }

    /// `site` as recorded: a native-library product at the library's rate, anything
    /// else as the kernel `alone` synthesizes for it, if it synthesizes.
    pub(crate) fn site_ns(&self, site: &SemanticSite, alone: Option<&Estimate>) -> f64 {
        if !site.exact {
            if let OpKind::Contraction(c) = &site.op.kind {
                let numel = |shape: &[u32]| shape.iter().map(|&e| u64::from(e)).product::<u64>();
                let macs = numel(&c.extents);
                let bytes = 4 * (numel(&c.lhs.shape) + numel(&c.rhs.shape) + numel(&c.out.shape));
                return self.library_ns(macs, bytes);
            }
        }
        alone.map_or(self.dispatch_ns, |e| self.kernel_ns(e))
    }
}
