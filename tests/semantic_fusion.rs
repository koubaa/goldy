//! Semantic fusion: runs of tensor operations and matrix products synthesized as one
//! kernel. Every trial records the same computation into a fusing scheme and an
//! unfused one over identical tensors, and requires every tensor to match byte for
//! byte after every frame.

#![cfg(all(feature = "gpu", feature = "tensor"))]

#[path = "common/submission.rs"]
mod submission;

use goldy::{
    compute, ArithmeticPolicy, BackendType, ContractionPrecision, FusionRegionStatus, FusionRejection, FusionSchedule,
    FusionTier, GoldyError, NodeId, RequestAdapterOptions, Runtime, RuntimeDescriptor, ScatterMode, Scheme, Tensor,
    TensorKernels, TensorRecorder, TensorShape,
};
use std::sync::Mutex;

static GPU: Mutex<()> = Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn runtime() -> Runtime {
    goldy::Instance::new()
        .expect("instance")
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_runtime(&RuntimeDescriptor::default())
        .expect("runtime")
}

/// Deterministic values in `[-1, 1)`.
fn data(seed: u32, n: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i.wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(40_503)).wrapping_mul(2_246_822_519);
            (h >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

enum Init {
    F32(TensorShape, Vec<f32>),
    I32(TensorShape, Vec<i32>),
    U32(TensorShape, Vec<u32>),
}

fn tensors(device: &Runtime, init: &[Init]) -> Vec<Tensor> {
    init.iter()
        .map(|i| match i {
            Init::F32(shape, v) => Tensor::from_f32(device, *shape, v),
            Init::I32(shape, v) => Tensor::from_i32(device, *shape, v),
            Init::U32(shape, v) => Tensor::from_u32(device, *shape, v),
        })
        .collect::<Result<_, _>>()
        .expect("tensor")
}

/// Records over the initial tensors; returns the tensors it allocates and the nodes a
/// trial may change.
type Recorded = Result<(Vec<Tensor>, Vec<NodeId>), GoldyError>;

/// The same recording in a fusing scheme and an unfused one.
struct Twin {
    _specialization: Option<goldy::test_support::SpecializationOverride>,
    /// Fusing first: each scheme, its tensors and the nodes `record` returned.
    runs: Vec<(Scheme, Vec<Tensor>, Vec<NodeId>)>,
    /// Whether each run specializes its kernels.
    specializes: Vec<bool>,
}

impl Twin {
    fn new(device: &Runtime, init: &[Init], record: impl Fn(&mut TensorRecorder<'_>, &[Tensor]) -> Recorded) -> Self {
        Self::build(device, init, record, &[(true, false), (false, false)])
    }

    /// Both schemes specialize their kernels, and a third unfused scheme keeps every
    /// generic kernel; [`Self::settle`] waits for the variants.
    fn specialized(
        device: &Runtime,
        init: &[Init],
        record: impl Fn(&mut TensorRecorder<'_>, &[Tensor]) -> Recorded,
    ) -> Self {
        Self::build(device, init, record, &[(true, true), (false, true), (false, false)])
    }

    /// One run per `(fusing, specializing)`.
    fn build(
        device: &Runtime,
        init: &[Init],
        record: impl Fn(&mut TensorRecorder<'_>, &[Tensor]) -> Recorded,
        modes: &[(bool, bool)],
    ) -> Self {
        let ctx = submission::submission_context(device);
        let kernels = TensorKernels::new(device).expect("tensor kernels");
        let runs = modes
            .iter()
            .map(|&(fusing, _)| {
                let mut scheme = Scheme::new(&ctx);
                scheme.set_automatic_fusion(fusing);
                let mut t = tensors(device, init);
                let (allocated, nodes) = record(&mut kernels.recorder(&mut scheme), &t).expect("record");
                t.extend(allocated);
                (scheme, t, nodes)
            })
            .collect();
        let mut twin = Self {
            _specialization: None,
            runs,
            specializes: modes.iter().map(|&(_, s)| s).collect(),
        };
        twin.pin(0);
        twin
    }

    /// Pin specialization as run `run` wants it.
    fn pin(&mut self, run: usize) {
        self._specialization = None;
        self._specialization = Some(if self.specializes[run] {
            goldy::test_support::SpecializationOverride::force_enabled()
        } else {
            goldy::test_support::SpecializationOverride::force_disabled()
        });
    }

    fn fused(&self) -> &Scheme {
        &self.runs[0].0
    }

    /// Submit `frames` frames, comparing the initial and the allocated tensors.
    fn frames(&mut self, frames: usize, what: &str) {
        for frame in 1..=frames {
            let mut parcels: Vec<Vec<Vec<u8>>> = Vec::new();
            for run in 0..self.runs.len() {
                self.pin(run);
                let (scheme, t, _) = &mut self.runs[run];
                let mut sub = scheme.submit().expect("submit");
                let got = t
                    .iter()
                    .map(|t| (&mut sub >> t.buffer()).take::<u8>().expect("take").to_vec())
                    .collect();
                drop(sub);
                goldy::test_support::wait_for_fusion_compiles(scheme);
                parcels.push(got);
            }
            self.pin(0);
            for (run, other) in parcels.iter().enumerate().skip(1) {
                for (j, (fused, unfused)) in parcels[0].iter().zip(other).enumerate() {
                    assert!(
                        fused == unfused,
                        "{what}, frame {frame}: tensor {j} differs in run {run}"
                    );
                }
            }
        }
    }

    /// Submit `frames` frames without comparing.
    fn submit(&mut self, frames: usize) {
        for _ in 0..frames {
            for run in 0..self.runs.len() {
                self.pin(run);
                let scheme = &mut self.runs[run].0;
                drop(scheme.submit().expect("submit"));
                goldy::test_support::wait_for_fusion_compiles(scheme);
            }
        }
        self.pin(0);
    }

    /// Submit until every specializing scheme runs its variants and nothing is left
    /// compiling.
    fn settle(&mut self) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        loop {
            self.submit(1);
            let settled = self
                .runs
                .iter()
                .zip(&self.specializes)
                .all(|((scheme, _, _), &specializes)| {
                    !scheme.compiles_pending() && (!specializes || scheme.replay_stats().specialization_promotions > 0)
                });
            if settled {
                return;
            }
            assert!(std::time::Instant::now() < deadline, "specialization did not settle");
            for (scheme, _, _) in &mut self.runs {
                goldy::test_support::wait_for_specialization_compiles(scheme);
            }
        }
    }

    /// Set user slot `slot` of returned node `node` in both schemes.
    fn set_param(&mut self, node: usize, slot: usize, value: u32) {
        for (scheme, _, nodes) in &mut self.runs {
            scheme.set_node_param(nodes[node], slot, value).expect("set_node_param");
        }
    }
}

/// Run `record` for `frames` frames fused and unfused, comparing the initial and the
/// allocated tensors; returns the fusing scheme.
fn fused_matches_unfused(
    device: &Runtime,
    init: &[Init],
    record: impl Fn(&mut TensorRecorder<'_>, &[Tensor]) -> Recorded,
    frames: usize,
) -> Scheme {
    let mut twin = Twin::new(device, init, record);
    twin.frames(frames, "steady");
    twin.runs.swap_remove(0).0
}

fn assert_one_semantic_region(scheme: &Scheme, nodes: usize) {
    let report = scheme.fusion_report();
    assert_eq!(report.regions.len(), 1, "{report:?}");
    let region = &report.regions[0];
    assert_eq!(region.tier, FusionTier::Semantic);
    assert_eq!(region.status, FusionRegionStatus::Promoted, "{report:?}");
    assert_eq!(region.nodes.len(), nodes);
    assert_eq!(scheme.executed_node_count(), 1);
}

/// The single fused region's structure.
fn structure(scheme: &Scheme) -> String {
    scheme.fusion_report().regions[0]
        .structure
        .clone()
        .expect("a semantic region has a structure")
}

/// Whether matrix products run on the Goldy kernels that semantic fusion describes.
/// Metal's library GEMM is a different kernel, so those tests stay on the other backends.
/// Indexed GEMV always uses `gemv_f32_indexed`, including on Metal.
fn stdlib_matmul(device: &Runtime) -> bool {
    device.backend_type() != BackendType::Metal
}

/// `gemv_f32` is what a chained GEMV site describes. Apple silicon runs it; another
/// Metal GPU may use the library GEMV.
fn stdlib_gemv(device: &Runtime) -> bool {
    if device.backend_type() != BackendType::Metal {
        return true;
    }
    device.adapter().name().contains("Apple")
}

/// The single fused region's schedule.
fn schedule(scheme: &Scheme) -> FusionSchedule {
    scheme.fusion_report().regions[0]
        .schedule
        .expect("a semantic region has a schedule")
}

/// The schedule of a matrix-vector product's region: lane groups of 32, four to a
/// workgroup, which exchange through the subgroup when a subgroup holds whole groups.
fn gemv_schedule(device: &Runtime) -> FusionSchedule {
    FusionSchedule::Lanes {
        lanes: 32,
        subgroup: device
            .capabilities()
            .subgroup_width
            .is_some_and(|w| w >= 32 && w.is_multiple_of(32) && 128u32.is_multiple_of(w)),
    }
}

const ROWS: u32 = 70;
const INNER: u32 = 100;

#[test]
fn gemv_into_gemv_over_a_short_rank() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !stdlib_gemv(&device) {
        return;
    }
    // The short vector is recomputed in every workgroup, so the consumer keeps its grid.
    let (m, rank, k) = (4u32, 4u32, 32u32);
    let init = [
        Init::F32(TensorShape::matrix(rank, k), data(1, rank * k)),
        Init::F32(TensorShape::vector(k), data(2, k)),
        Init::F32(TensorShape::vector(rank), vec![0.0; rank as usize]),
        Init::F32(TensorShape::matrix(m, rank), data(3, m * rank)),
        Init::F32(TensorShape::vector(m), vec![0.0; m as usize]),
    ];
    let scheme = fused_matches_unfused(
        &device,
        &init,
        |rec, t| {
            rec.matmul_into("down", t[0].view(), t[1].view(), t[2].view())?;
            rec.matmul_into("forget", t[3].view(), t[2].view(), t[4].view())?;
            Ok((Vec::new(), Vec::new()))
        },
        4,
    );
    assert_one_semantic_region(&scheme, 2);
    assert_eq!(schedule(&scheme), gemv_schedule(&device));
}

fn record_chain(rec: &mut TensorRecorder<'_>, t: &[Tensor]) -> Result<(Vec<Tensor>, Vec<NodeId>), GoldyError> {
    rec.matmul_into("down", t[0].view(), t[1].view(), t[2].view())?;
    rec.matmul_into("forget", t[3].view(), t[2].view(), t[4].view())?;
    Ok((Vec::new(), Vec::new()))
}

fn chain(m: u32, rank: u32, k: u32) -> [Init; 5] {
    [
        Init::F32(TensorShape::matrix(rank, k), data(1, rank * k)),
        Init::F32(TensorShape::vector(k), data(2, k)),
        Init::F32(TensorShape::vector(rank), vec![0.0; rank as usize]),
        Init::F32(TensorShape::matrix(m, rank), data(3, m * rank)),
        Init::F32(TensorShape::vector(m), vec![0.0; m as usize]),
    ]
}

/// Rows that are not a multiple of four, and a rank that is not a power of two.
#[test]
fn gemv_chain_covers_a_partial_workgroup() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !stdlib_gemv(&device) {
        return;
    }
    for (m, rank, k) in [(9u32, 5u32, 40u32), (13, 37, 32)] {
        let scheme = fused_matches_unfused(&device, &chain(m, rank, k), record_chain, 8);
        assert_one_semantic_region(&scheme, 2);
    }
}

/// A large consumer stays two dispatches. One workgroup must not walk every row.
#[test]
fn a_large_gemv_chain_stays_two_dispatches() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !stdlib_gemv(&device) {
        return;
    }
    let mut twin = Twin::new(&device, &chain(4096, 16, 4096), record_chain);
    twin.submit(2);
    assert!(
        twin.fused().fusion_report().regions.is_empty(),
        "{:?}",
        twin.fused().fusion_report()
    );
    assert_eq!(twin.fused().executed_node_count(), 2);
}

#[test]
fn contraction_with_a_pointwise_epilogue() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !stdlib_matmul(&device) {
        return;
    }
    // x += W @ h, with the residual updated in place.
    let init = [
        Init::F32(TensorShape::matrix(ROWS, INNER), data(1, ROWS * INNER)),
        Init::F32(TensorShape::vector(INNER), data(2, INNER)),
        Init::F32(TensorShape::vector(ROWS), data(3, ROWS)),
        Init::F32(TensorShape::vector(ROWS), vec![0.0; ROWS as usize]),
    ];
    let scheme = fused_matches_unfused(
        &device,
        &init,
        |rec, t| {
            rec.matmul_into("project", t[0].view(), t[1].view(), t[3].view())?;
            rec.add_into("residual", t[2].view(), t[3].view(), t[2].view())?;
            Ok((Vec::new(), Vec::new()))
        },
        5,
    );
    assert_one_semantic_region(&scheme, 2);
    assert_eq!(scheme.fusion_report().regions[0].forwarded, 1);
    assert_eq!(schedule(&scheme), gemv_schedule(&device));
    // The product stays a contraction, and the residual update reads it as an epilogue.
    assert_eq!(
        structure(&scheme),
        "contraction p2[i0] = sum{s1<100 by 32x2}(p0[i0, s1] * p1[s1])\n\
         output p3[i0] = p3[i0] + p2[i0]\n"
    );
}

#[test]
fn sibling_contractions_with_a_gated_epilogue() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !stdlib_matmul(&device) {
        return;
    }
    // out = silu(W1 @ x) * (W3 @ x), where silu(g) = g / (1 + exp(-g)).
    let init = [
        Init::F32(TensorShape::matrix(ROWS, INNER), data(1, ROWS * INNER)),
        Init::F32(TensorShape::matrix(ROWS, INNER), data(5, ROWS * INNER)),
        Init::F32(TensorShape::vector(INNER), data(2, INNER)),
        Init::F32(TensorShape::vector(ROWS), vec![0.0; ROWS as usize]),
        Init::F32(TensorShape::vector(ROWS), vec![0.0; ROWS as usize]),
    ];
    let scheme = fused_matches_unfused(
        &device,
        &init,
        |rec, t| {
            rec.matmul_into("gate", t[0].view(), t[2].view(), t[3].view())?;
            rec.matmul_into("up", t[1].view(), t[2].view(), t[4].view())?;
            let negated = rec.neg("neg", t[3].view())?;
            let exp = rec.exp("exp", negated.view())?;
            let denominator = rec.add_scalar("denominator", exp.view(), 1.0)?;
            let silu = rec.div("silu", t[3].view(), denominator.view())?;
            let out = rec.mul("gated", silu.view(), t[4].view())?;
            Ok((vec![negated, exp, denominator, silu, out], Vec::new()))
        },
        5,
    );
    assert_one_semantic_region(&scheme, 7);
    let structure = structure(&scheme);
    assert_eq!(
        structure.lines().filter(|l| l.starts_with("contraction")).count(),
        2,
        "{structure}"
    );
    assert!(
        structure.ends_with("shared rhs of p2 and rhs of p4\n"),
        "both products read the same input: {structure}"
    );
    // Both products share one strided loop and one exchange, and stay exact.
    assert_eq!(schedule(&scheme), gemv_schedule(&device));
}

#[test]
fn a_scale_updated_in_place() {
    let _gpu = gpu_lock();
    let device = runtime();
    // out = x / sqrt(mean(x * x) + eps), the scale refined in place in one element.
    let n = 100;
    let init = [
        Init::F32(TensorShape::vector(n), data(4, n)),
        Init::F32(TensorShape::vector(n), vec![0.0; n as usize]),
        Init::F32(TensorShape::vector(1), vec![0.0]),
        Init::F32(TensorShape::vector(n), vec![0.0; n as usize]),
    ];
    let scheme = fused_matches_unfused(
        &device,
        &init,
        |rec, t| {
            let (x, square, scale, out) = (t[0].view(), t[1].view(), t[2].view(), t[3].view());
            rec.mul_into("square", x, x, square)?;
            rec.mean_into("mean", square, 0, scale)?;
            rec.add_scalar_into("eps", scale, 1e-5, scale)?;
            rec.sqrt_into("sqrt", scale, scale)?;
            rec.reciprocal_into("scale", scale, scale)?;
            rec.mul_into("normalize", scale, x, out)?;
            Ok((Vec::new(), Vec::new()))
        },
        3,
    );
    assert_one_semantic_region(&scheme, 6);
}

#[test]
fn reductions_over_entries_that_hold_the_identity() {
    let _gpu = gpu_lock();
    let device = runtime();
    // Each row's last columns hold -1e30, the identity of its max, so they weigh
    // exp(-1e30 - max) = 0 in the row's sum: a fixed-width reduction over a padded axis.
    let (rows, cols, live) = (8, 12, 8);
    let mut x = data(9, rows * cols);
    for row in x.chunks_mut(cols as usize) {
        row[live as usize..].fill(-1.0e30);
    }
    let init = [
        Init::F32(TensorShape::matrix(rows, cols), x),
        Init::F32(TensorShape::matrix(rows, 1), vec![0.0; rows as usize]),
        Init::F32(TensorShape::matrix(rows, cols), vec![0.0; (rows * cols) as usize]),
        Init::F32(TensorShape::vector(rows), vec![0.0; rows as usize]),
        Init::F32(TensorShape::vector(rows), vec![0.0; rows as usize]),
        Init::F32(TensorShape::vector(rows), vec![0.0; rows as usize]),
    ];
    let scheme = fused_matches_unfused(
        &device,
        &init,
        |rec, t| {
            let (x, top, shifted) = (t[0].view(), t[1].view(), t[2].view());
            let (total, low, out) = (t[3].view(), t[4].view(), t[5].view());
            rec.max_reduce_into("max", x, 1, top)?;
            rec.sub_into("shift", x, top, shifted)?;
            rec.exp_into("exp", shifted, shifted)?;
            rec.sum_into("sum", shifted, 1, total)?;
            rec.min_reduce_into("min", x.narrow(1, 0, live)?, 1, low)?;
            rec.div_into("ratio", total, low, out)?;
            Ok((Vec::new(), Vec::new()))
        },
        3,
    );
    assert_one_semantic_region(&scheme, 6);
}

#[test]
fn tanh_and_sigmoid_fuse() {
    let _gpu = gpu_lock();
    let device = runtime();
    let mut values = data(7, 16);
    values.extend_from_slice(&[
        0.0,
        4.0,
        -4.0,
        25.0,
        -25.0,
        30.0,
        -30.0,
        100.0,
        -100.0,
        1.0e10,
        -1.0e10,
        f32::MAX,
        f32::MIN,
    ]);
    let n = values.len() as u32;
    let init = [Init::F32(TensorShape::vector(n), values)];
    let scheme = fused_matches_unfused(
        &device,
        &init,
        |rec, t| {
            let y = rec.tanh("tanh", t[0].view())?;
            let z = rec.sigmoid("sigmoid", y.view())?;
            let out = rec.mul_scalar("scale", z.view(), 2.0)?;
            Ok((vec![y, z, out], Vec::new()))
        },
        3,
    );
    assert_one_semantic_region(&scheme, 3);
    let structure = structure(&scheme);
    assert!(structure.contains("tanh("), "{structure}");
    assert!(structure.contains("sigmoid("), "{structure}");
}

#[test]
fn products_of_different_rows_side_by_side() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !stdlib_matmul(&device) {
        return;
    }
    // q = Wq @ x and k = Wk @ x into disjoint ranges of one buffer, as a query and
    // key projection: one dispatch runs both grids.
    let (q, k) = (48, 16);
    let init = [
        Init::F32(TensorShape::matrix(q, INNER), data(1, q * INNER)),
        Init::F32(TensorShape::matrix(k, INNER), data(5, k * INNER)),
        Init::F32(TensorShape::vector(INNER), data(2, INNER)),
        Init::F32(TensorShape::vector(q + k), vec![-3.0; (q + k) as usize]),
    ];
    let scheme = fused_matches_unfused(
        &device,
        &init,
        |rec, t| {
            rec.matmul_into("query", t[0].view(), t[2].view(), t[3].view().narrow(0, 0, 48)?)?;
            rec.matmul_into("key", t[1].view(), t[2].view(), t[3].view().narrow(0, 48, 16)?)?;
            Ok((Vec::new(), Vec::new()))
        },
        5,
    );
    assert_one_semantic_region(&scheme, 2);
    assert_eq!(schedule(&scheme), gemv_schedule(&device));
}

#[test]
fn normalized_input_of_a_product() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !stdlib_matmul(&device) {
        return;
    }
    // y = W @ (x / sqrt(mean(x * x) + eps) * g): an RMSNorm prologue, whose scale
    // every thread of the product computes once rather than once per term.
    let init = [
        Init::F32(TensorShape::matrix(ROWS, INNER), data(1, ROWS * INNER)),
        Init::F32(TensorShape::vector(INNER), data(2, INNER)),
        Init::F32(TensorShape::vector(INNER), data(3, INNER)),
        Init::F32(TensorShape::vector(ROWS), vec![0.0; ROWS as usize]),
    ];
    let scheme = fused_matches_unfused(
        &device,
        &init,
        |rec, t| {
            let square = rec.mul("square", t[1].view(), t[1].view())?;
            let mean = rec.mean("mean", square.view(), 0)?;
            let shifted = rec.add_scalar("eps", mean.view(), 1e-5)?;
            let root = rec.sqrt("sqrt", shifted.view())?;
            let scale = rec.reciprocal("scale", root.view())?;
            let normalized = rec.mul("normalize", t[1].view(), scale.view())?;
            let weighted = rec.mul("weight", normalized.view(), t[2].view())?;
            rec.matmul_into("project", t[0].view(), weighted.view(), t[3].view())?;
            Ok((
                vec![square, mean, shifted, root, scale, normalized, weighted],
                Vec::new(),
            ))
        },
        5,
    );
    assert_one_semantic_region(&scheme, 8);
    let structure = structure(&scheme);
    assert_eq!(
        structure.lines().filter(|l| l.starts_with("contraction")).count(),
        1,
        "{structure}"
    );
}

/// `C = A @ B` for `A` of `GEMM.0 × GEMM.2` and `B` of `GEMM.2 × GEMM.1`: no extent
/// a multiple of the 16-square matrix tile.
const GEMM: (u32, u32, u32) = (40, 72, 100);

fn gemm_init() -> [Init; 3] {
    let (m, n, k) = GEMM;
    [
        Init::F32(TensorShape::matrix(m, k), data(6, m * k)),
        Init::F32(TensorShape::matrix(k, n), data(7, k * n)),
        Init::F32(TensorShape::matrix(m, n), vec![0.0; (m * n) as usize]),
    ]
}

/// `C = A @ B`, then `max(C, 0)`.
fn gemm_relu(rec: &mut TensorRecorder<'_>, t: &[Tensor]) -> Result<(Vec<Tensor>, Vec<NodeId>), GoldyError> {
    rec.matmul_into("project", t[0].view(), t[1].view(), t[2].view())?;
    let relu = rec.max_scalar("relu", t[2].view(), 0.0)?;
    Ok((vec![relu], Vec::new()))
}

#[test]
fn matrix_product_with_an_epilogue_stays_exact_by_default() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !stdlib_matmul(&device) {
        return;
    }
    let scheme = fused_matches_unfused(&device, &gemm_init(), gemm_relu, 5);
    assert_eq!(scheme.contraction_precision(), ContractionPrecision::Exact);
    if library_gemm(&device) {
        // The library sums in its own order, which no exact fusion can reproduce.
        assert!(scheme.fusion_report().regions.is_empty());
        return;
    }
    assert_one_semantic_region(&scheme, 2);
    assert_eq!(schedule(&scheme), FusionSchedule::Threads);
}

/// Whether general matrix products run on the backend library (cuBLAS) by default.
fn library_gemm(device: &Runtime) -> bool {
    device.backend_type() == BackendType::Cuda
}

/// `x` rounded to the nearest f16, ties to even (normal range).
fn round_f16(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits((bits + 0x0fff + ((bits >> 13) & 1)) & !0x1fff)
}

#[test]
fn matrix_product_on_matrix_units_when_rounding_is_admitted() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !stdlib_matmul(&device) || !device.capabilities().matrix_multiply {
        return;
    }
    let _specialization = goldy::test_support::SpecializationOverride::force_disabled();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).expect("tensor kernels");
    let mut scheme = Scheme::new(&ctx);
    scheme.set_automatic_fusion(true);
    scheme.set_contraction_precision(ContractionPrecision::F16Factors);
    let init = gemm_init();
    let mut t = tensors(&device, &init);
    let (allocated, _) = gemm_relu(&mut kernels.recorder(&mut scheme), &t).expect("record");
    t.extend(allocated);

    let (m, n, k) = GEMM;
    let [Init::F32(_, a), Init::F32(_, b), _] = &init else {
        unreachable!("f32 operands")
    };
    let product = |round: fn(f32) -> f32| -> Vec<f64> {
        (0..m * n)
            .map(|e| {
                let (i, j) = (e / n, e % n);
                (0..k)
                    .map(|s| f64::from(round(a[(i * k + s) as usize])) * f64::from(round(b[(s * n + j) as usize])))
                    .sum()
            })
            .collect()
    };
    let (exact, rounded) = (product(|x| x), product(round_f16));
    let mut fused_frames = 0;
    for frame in 1..=4 {
        let mut sub = scheme.submit().expect("submit");
        let got: Vec<Vec<f32>> = t[2..]
            .iter()
            .map(|t| {
                (&mut sub >> t.buffer())
                    .take::<u8>()
                    .expect("take")
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            })
            .collect();
        drop(sub);
        goldy::test_support::wait_for_fusion_compiles(&mut scheme);
        let fused = scheme.executed_node_count() == 1;
        let (mut from_rounded, mut from_exact) = (0f64, 0f64);
        for (e, (&c, &relu)) in got[0].iter().zip(&got[1]).enumerate() {
            assert_eq!(relu, c.max(0.0), "frame {frame}: the epilogue reads the product");
            from_rounded = from_rounded.max((f64::from(c) - rounded[e]).abs());
            from_exact = from_exact.max((f64::from(c) - exact[e]).abs());
        }
        // Only f32 summation separates the fused result from the product of the rounded
        // factors; rounding the factors moves it far more.
        let rounding = exact
            .iter()
            .zip(&rounded)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        assert!(from_exact < 0.05, "frame {frame}: {from_exact}");
        if fused {
            fused_frames += 1;
            assert!(
                from_rounded < 1e-4 && from_rounded * 10.0 < rounding,
                "frame {frame}: {from_rounded} vs {rounding}"
            );
        }
    }
    assert!(fused_frames > 0);
    assert_one_semantic_region(&scheme, 2);
    assert_eq!(schedule(&scheme), FusionSchedule::Matrix);

    // Exact contractions again: the region replans onto the exact schedule, or, for a
    // library product, runs as recorded.
    scheme.set_contraction_precision(ContractionPrecision::Exact);
    for _ in 0..4 {
        drop(scheme.submit().expect("submit"));
        goldy::test_support::wait_for_fusion_compiles(&mut scheme);
    }
    if library_gemm(&device) {
        assert!(scheme.fusion_report().regions.is_empty());
        assert_eq!(scheme.executed_node_count(), 2);
    } else {
        assert_one_semantic_region(&scheme, 2);
        assert_eq!(schedule(&scheme), FusionSchedule::Threads);
    }
}

/// The run's rejection for costing more fused than as recorded.
fn cost_rejection(scheme: &Scheme) -> Option<(u64, u64)> {
    scheme.fusion_report().rejected.iter().find_map(|r| match r.reason {
        FusionRejection::Cost { fused_ns, unfused_ns } => Some((fused_ns, unfused_ns)),
        _ => None,
    })
}

#[test]
fn a_product_the_library_runs_faster_stays_with_it() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !library_gemm(&device) || !device.capabilities().matrix_multiply {
        return;
    }
    let _specialization = goldy::test_support::SpecializationOverride::force_disabled();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).expect("tensor kernels");
    let mut scheme = Scheme::new(&ctx);
    scheme.set_automatic_fusion(true);
    scheme.set_contraction_precision(ContractionPrecision::F16Factors);
    let n = 512;
    let init = [
        Init::F32(TensorShape::matrix(n, n), data(6, n * n)),
        Init::F32(TensorShape::matrix(n, n), data(7, n * n)),
        Init::F32(TensorShape::matrix(n, n), vec![0.0; (n * n) as usize]),
    ];
    let t = tensors(&device, &init);
    let (_relu, _) = gemm_relu(&mut kernels.recorder(&mut scheme), &t).expect("record");
    for _ in 0..3 {
        drop(scheme.submit().expect("submit"));
        goldy::test_support::wait_for_fusion_compiles(&mut scheme);
    }
    // One tile a subgroup on matrix units is far slower than the library at this size,
    // which saving the epilogue's dispatch does not repay.
    assert!(
        scheme.fusion_report().regions.is_empty(),
        "{:?}",
        scheme.fusion_report()
    );
    let (fused_ns, unfused_ns) = cost_rejection(&scheme).expect("a cost rejection");
    assert!(fused_ns > unfused_ns);
    assert_eq!(scheme.executed_node_count(), 2);

    // A device model whose matrix units match the library's rate fuses it again.
    let mut model = scheme.fusion_cost_model();
    model.matrix_macs_per_ns = model.library_macs_per_ns;
    scheme.set_fusion_cost_model(model);
    for _ in 0..3 {
        drop(scheme.submit().expect("submit"));
        goldy::test_support::wait_for_fusion_compiles(&mut scheme);
    }
    assert_one_semantic_region(&scheme, 2);
}

/// `s = sum(X, axis 1)`, then `y = W @ s`, for `X` of `ROWS × inner`.
fn row_sums_then_product(rec: &mut TensorRecorder<'_>, t: &[Tensor]) -> Result<(Vec<Tensor>, Vec<NodeId>), GoldyError> {
    let sums = rec.sum("row sums", t[0].view(), 1)?;
    rec.matmul_into("project", t[1].view(), sums.view(), t[2].view())?;
    Ok((vec![sums], Vec::new()))
}

#[test]
fn a_factor_recomputed_for_every_row_stays_unfused_when_it_costs_more() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !stdlib_matmul(&device) {
        return;
    }
    // Fused, every lane of every output row sums its own elements of `s`: `ROWS`
    // times the work, and a longest thread `inner` times longer.
    let init = |inner: u32| {
        [
            Init::F32(TensorShape::matrix(INNER, inner), data(1, INNER * inner)),
            Init::F32(TensorShape::matrix(ROWS, INNER), data(2, ROWS * INNER)),
            Init::F32(TensorShape::vector(ROWS), vec![0.0; ROWS as usize]),
        ]
    };
    let mut short = Twin::new(&device, &init(4), row_sums_then_product);
    if device.backend_type() == BackendType::Cuda {
        // NVRTC contracts the product's multiply and add into an FMA only when one basic
        // block holds both, and the recomputed sum's loop separates them.
        short.submit(3);
    } else {
        short.frames(3, "short rows");
    }
    assert_one_semantic_region(short.fused(), 2);
    let long = fused_matches_unfused(&device, &init(1024), row_sums_then_product, 3);
    assert!(long.fusion_report().regions.is_empty(), "{:?}", long.fusion_report());
    let (fused_ns, unfused_ns) = cost_rejection(&long).expect("a cost rejection");
    assert!(fused_ns > unfused_ns);
}

#[test]
fn contraction_relocated_into_a_selected_row() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !stdlib_matmul(&device) {
        return;
    }
    // cache[position] = W @ h, with the position read from a parcel on the device.
    let cache_rows = 6;
    let init = [
        Init::F32(TensorShape::matrix(ROWS, INNER), data(1, ROWS * INNER)),
        Init::F32(TensorShape::vector(INNER), data(2, INNER)),
        Init::F32(TensorShape::vector(ROWS), vec![0.0; ROWS as usize]),
        Init::F32(
            TensorShape::matrix(cache_rows, ROWS),
            vec![-7.0; (cache_rows * ROWS) as usize],
        ),
        Init::I32(TensorShape::vector(1), vec![4]),
    ];
    let scheme = fused_matches_unfused(
        &device,
        &init,
        |rec, t| {
            rec.matmul_into("project", t[0].view(), t[1].view(), t[2].view())?;
            let row = t[2].view().reshape(&[1, ROWS])?;
            let position = t[4]
                .view()
                .reshape(&[1, 1])?
                .broadcast_to(TensorShape::matrix(1, ROWS))?;
            rec.scatter("store", row, position, t[3].view(), 0, ScatterMode::UniqueWrite)?;
            Ok((Vec::new(), Vec::new()))
        },
        5,
    );
    assert_one_semantic_region(&scheme, 2);
}

#[test]
fn pointwise_tensor_chain() {
    let _gpu = gpu_lock();
    let device = runtime();
    let n = 1000;
    let init = [
        Init::F32(TensorShape::vector(n), data(4, n)),
        Init::F32(TensorShape::vector(n), vec![0.0; n as usize]),
        Init::F32(TensorShape::vector(n), vec![0.0; n as usize]),
    ];
    // out = max(exp(-x) - x, 0.25), through an intermediate the chain overwrites.
    let scheme = fused_matches_unfused(
        &device,
        &init,
        |rec, t| {
            let negated = rec.neg("neg", t[0].view())?;
            let exp = rec.exp("exp", negated.view())?;
            rec.copy("keep", exp.view(), t[1].view())?;
            let diff = rec.sub("sub", t[1].view(), t[0].view())?;
            let clamped = rec.max_scalar("clamp", diff.view(), 0.25)?;
            rec.copy("store", clamped.view(), t[2].view())?;
            Ok((vec![negated, exp, diff, clamped], Vec::new()))
        },
        5,
    );
    assert_one_semantic_region(&scheme, 6);
    assert!(!structure(&scheme).contains("contraction"));
}

#[compute(workgroup_size = [64, 1, 1])]
fn rectify(input: &[f32], output: goldy::gpu::Scattered<f32>, count: u32, gain: f32) {
    let i = goldy::gpu::global_id().x;
    if i < count {
        output[i] = goldy::gpu::max(input[i], 0.0) * gain;
    }
}

/// User slots of `rectify`.
const RECTIFY_COUNT: usize = 0;
const RECTIFY_GAIN: usize = 1;

#[test]
fn contraction_with_a_generated_epilogue() {
    let _gpu = gpu_lock();
    let device = runtime();
    if !stdlib_matmul(&device) {
        return;
    }
    // out[..count] = max(W @ h, 0) * gain, where the epilogue is a generated kernel.
    let init = [
        Init::F32(TensorShape::matrix(ROWS, INNER), data(1, ROWS * INNER)),
        Init::F32(TensorShape::vector(INNER), data(2, INNER)),
        Init::F32(TensorShape::vector(ROWS), vec![0.0; ROWS as usize]),
        Init::F32(TensorShape::vector(ROWS), vec![-7.0; ROWS as usize]),
    ];
    let mut twin = Twin::new(&device, &init, |rec, t| {
        rec.matmul_into("project", t[0].view(), t[1].view(), t[2].view())?;
        let device = rec.scheme().context().runtime().clone();
        let kernel = rectify::Kernel::prepare(&device)?;
        let node = kernel
            .invoke(t[2].buffer(), t[3].buffer(), ROWS - 13, 1.5)
            .over_1d(ROWS)
            .record(rec.scheme(), "rectify")?
            .node();
        Ok((Vec::new(), vec![node]))
    });
    twin.frames(5, "steady");
    assert_one_semantic_region(twin.fused(), 2);
    let fallbacks = twin.fused().replay_stats().fusion_fallbacks;

    // The gain binds a scalar parameter: the fused kernel reads the new value.
    twin.set_param(0, RECTIFY_GAIN, 0.75f32.to_bits());
    twin.frames(2, "after the gain changes");
    assert_one_semantic_region(twin.fused(), 2);
    assert_eq!(twin.fused().replay_stats().fusion_fallbacks, fallbacks);

    // The count is the lifted extent: the region is planned again for the new value.
    twin.set_param(0, RECTIFY_COUNT, ROWS);
    twin.frames(5, "after the count changes");
    assert_one_semantic_region(twin.fused(), 2);
    assert_eq!(twin.fused().replay_stats().fusion_fallbacks, fallbacks + 1);
}

#[goldy::compute(workgroup_size = [64, 1, 1])]
fn shifted_product(
    #[tensor(shape = [n])] x: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [n])] w: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [n])] y: goldy::gpu::TensorWrite<f32>,
) {
    let i = goldy::gpu::global_id().x;
    if i < y.len() {
        let shifted = x[i] + 1.5;
        y[i] = shifted * w[i];
    }
}

/// `shifted_product`'s tensors after `sum` (`a + b`) and before `halve`: `a, b, w,
/// x, y, z, scratch`.
fn described_init() -> [Init; 7] {
    let vector = TensorShape::vector(ROWS);
    let zeros = || Init::F32(vector, vec![0.0; ROWS as usize]);
    [
        Init::F32(vector, data(1, ROWS)),
        Init::F32(vector, data(2, ROWS)),
        Init::F32(vector, data(3, ROWS)),
        zeros(),
        zeros(),
        zeros(),
        zeros(),
    ]
}

/// `x = a + b`, then `shifted_product` described through `scratch` as `exact` says,
/// then `z = y * 0.5`.
fn record_described(rec: &mut TensorRecorder<'_>, t: &[Tensor], exact: bool) -> Recorded {
    rec.add_into("sum", t[0].view(), t[1].view(), t[3].view())?;
    let device = rec.scheme().context().runtime().clone();
    let kernel = shifted_product::Kernel::prepare(&device)?;
    let node = kernel
        .record(rec.scheme(), "shifted_product", t[3].view(), t[2].view(), t[4].view())?
        .over_1d(ROWS)
        .node();
    rec.describe(node, exact, |d| {
        d.add_scalar_into("shift", t[3].view(), 1.5, t[6].view())?;
        d.mul_into("product", t[6].view(), t[2].view(), t[4].view())
    })?;
    rec.mul_scalar_into("halve", t[4].view(), 0.5, t[5].view())?;
    Ok((Vec::new(), vec![node]))
}

/// A described kernel fuses with its neighbours as the operations that describe it,
/// and the region does not store what only the description writes.
#[test]
fn described_dispatch_fuses_as_its_description() {
    let _gpu = gpu_lock();
    let device = runtime();
    let scheme = fused_matches_unfused(&device, &described_init(), |rec, t| record_described(rec, t, true), 5);
    assert_one_semantic_region(&scheme, 3);
    let labels = &scheme.fusion_report().regions[0].labels;
    assert!(labels.iter().any(|l| l.to_string() == "shifted_product"), "{labels:?}");
}

/// Under recorded arithmetic, a description that does not round as its kernel does
/// stays out of regions unless contractions may reassociate; its kernel runs as recorded.
#[test]
fn inexact_description_needs_reassociation() {
    let _gpu = gpu_lock();
    let device = runtime();
    let scheme = fused_matches_unfused(&device, &described_init(), |rec, t| record_described(rec, t, false), 5);
    let report = scheme.fusion_report();
    assert!(
        report
            .regions
            .iter()
            .all(|r| r.labels.iter().all(|l| l.to_string() != "shifted_product")),
        "{report:?}"
    );
    assert_eq!(scheme.executed_node_count(), 3);
}

/// Relaxed arithmetic permits the reassociation an inexact description declares.
#[test]
fn relaxed_arithmetic_admits_an_inexact_description() {
    let _gpu = gpu_lock();
    let device = runtime();
    // `Twin` compares bytes, which relaxed arithmetic does not promise; this chain
    // (`(a + b + 1.5) * w * 0.5`) leaves a compiler nothing to reassociate or contract.
    let mut twin = Twin::new(&device, &described_init(), |rec, t| record_described(rec, t, false));
    for (scheme, _, _) in &mut twin.runs {
        scheme.set_arithmetic_policy(ArithmeticPolicy::Relaxed);
    }
    twin.frames(5, "relaxed");
    assert_one_semantic_region(twin.fused(), 3);
    let labels = &twin.fused().fusion_report().regions[0].labels;
    assert!(labels.iter().any(|l| l.to_string() == "shifted_product"), "{labels:?}");
}

#[goldy::compute(workgroup_size = [64, 1, 1])]
fn multiply_add(
    #[tensor(shape = [n])] x: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [n])] w: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [n])] b: goldy::gpu::Tensor<f32>,
    #[tensor(shape = [n])] y: goldy::gpu::TensorWrite<f32>,
) {
    let i = goldy::gpu::global_id().x;
    if i < y.len() {
        y[i] = x[i] * w[i] + b[i];
    }
}

/// `x = a + b`, then `multiply_add(x, w, a)` described exactly as a product and a sum
/// through `scratch`, then `z = y * 0.5`.
fn record_multiply_add(rec: &mut TensorRecorder<'_>, t: &[Tensor]) -> Recorded {
    rec.add_into("sum", t[0].view(), t[1].view(), t[3].view())?;
    let device = rec.scheme().context().runtime().clone();
    let kernel = multiply_add::Kernel::prepare(&device)?;
    let node = kernel
        .record(
            rec.scheme(),
            "multiply_add",
            t[3].view(),
            t[2].view(),
            t[0].view(),
            t[4].view(),
        )?
        .over_1d(ROWS)
        .node();
    rec.describe(node, true, |d| {
        d.mul_into("product", t[3].view(), t[2].view(), t[6].view())?;
        d.add_into("bias", t[6].view(), t[0].view(), t[4].view())
    })?;
    rec.mul_scalar_into("halve", t[4].view(), 0.5, t[5].view())?;
    Ok((Vec::new(), vec![node]))
}

/// A compiler that contracts may compute an exactly described `x * w + a` with one
/// rounding where its operations round twice, so under recorded arithmetic the
/// description joins a region only on a device that never contracts.
#[test]
fn a_described_multiply_add_fuses_only_where_nothing_contracts() {
    let _gpu = gpu_lock();
    let device = runtime();
    let never = device.capabilities().fp_contraction == goldy::FpContraction::Never;
    let scheme = fused_matches_unfused(&device, &described_init(), record_multiply_add, 5);
    let report = scheme.fusion_report();
    let joined = report
        .regions
        .iter()
        .any(|r| r.status == FusionRegionStatus::Promoted && r.labels.iter().any(|l| *l == "multiply_add"));
    assert_eq!(joined, never, "{report:?}");
    let declined = report
        .rejected
        .iter()
        .any(|r| r.reason == FusionRejection::Contraction { stage: 0 } && r.labels == ["multiply_add"]);
    assert_eq!(declined, !never, "{report:?}");
}

#[test]
fn describe_needs_composable_operations() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).expect("tensor kernels");
    let mut scheme = Scheme::new(&ctx);
    let t = tensors(&device, &described_init());
    let flags = Tensor::from_u32(&device, TensorShape::vector(4), &[0; 4]).expect("tensor");
    let kernel = shifted_product::Kernel::prepare(&device).expect("kernel");
    let node = kernel
        .record(&mut scheme, "shifted_product", t[3].view(), t[2].view(), t[4].view())
        .expect("record")
        .over_1d(ROWS)
        .node();
    let mut rec = kernels.recorder(&mut scheme);
    let err = rec
        .describe(node, true, |d| {
            d.fill("flags", flags.view(), goldy::TensorScalar::U32(1))?;
            d.mul_into("product", t[3].view(), t[2].view(), t[4].view())
        })
        .expect_err("a fill of integers has no site");
    assert!(err.to_string().contains("flags"), "{err}");
}

/// What a scheme keeps alive lives as long as it and every scheme that includes it.
#[test]
fn kept_values_live_as_long_as_their_schemes() {
    struct Flag(std::sync::Arc<std::sync::atomic::AtomicBool>);
    impl Drop for Flag {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut parent = Scheme::new(&ctx);
    let mut child = Scheme::new(&ctx);
    child.keep_alive(Flag(dropped.clone()));
    parent.include("child", &child).expect("include").finish();
    drop(child);
    assert!(!dropped.load(std::sync::atomic::Ordering::SeqCst));
    drop(parent);
    assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
}

/// Turning fusion off after a plan has promoted must re-record, not resubmit a
/// command buffer whose elided temporaries were just released.
#[test]
fn turning_fusion_off_after_promotion_resubmits() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).expect("tensor kernels");
    let mut scheme = Scheme::new(&ctx);
    scheme.set_automatic_fusion(true);
    let n = 64u32;
    let shape = TensorShape::vector(n);
    let left = data(1, n);
    let right = data(2, n);
    let a = Tensor::from_f32(&device, shape, &left).unwrap();
    let b = Tensor::from_f32(&device, shape, &right).unwrap();
    let out = Tensor::zeros(&device, shape, goldy::TensorDType::F32).unwrap();
    let scaled = {
        let mut rec = kernels.recorder(&mut scheme);
        let scaled = rec.mul("mul", a.view(), b.view()).unwrap();
        rec.add_into("add", scaled.view(), a.view(), out.view()).unwrap();
        scaled
    };
    let _keep = scaled;
    let mut promoted = false;
    for _ in 0..24 {
        drop(scheme.submit().expect("submit"));
        goldy::test_support::wait_for_fusion_compiles(&mut scheme);
        promoted = scheme
            .fusion_report()
            .regions
            .iter()
            .any(|region| region.status == FusionRegionStatus::Promoted);
        if promoted {
            break;
        }
    }
    assert!(promoted, "fusion did not promote");
    scheme.set_automatic_fusion(false);
    let mut sub = scheme.submit().expect("submit after fusion off");
    let got = (&mut sub >> out.buffer()).take::<f32>().expect("take").to_vec();
    drop(sub);
    for (i, (got, (x, y))) in got.iter().zip(left.iter().zip(&right)).enumerate() {
        let want = x * y + x;
        assert!((got - want).abs() < 1e-4, "index {i}: {got} != {want}");
    }
    assert!(scheme.replay_stats().fusion_fallbacks >= 1);
}

/// Arithmetic policy is part of fused-program identity: exact schemes share a
/// compile, while a relaxed scheme gets its own source and pipeline.
#[test]
fn fused_program_cache_is_keyed_by_arithmetic_policy() {
    let _gpu = gpu_lock();
    let device = runtime();
    let ctx = submission::submission_context(&device);
    let kernels = TensorKernels::new(&device).expect("tensor kernels");
    let n = 64u32;
    let shape = TensorShape::vector(n);
    let mut schemes = Vec::new();
    for arithmetic in [
        ArithmeticPolicy::MatchRecorded,
        ArithmeticPolicy::MatchRecorded,
        ArithmeticPolicy::Relaxed,
    ] {
        let mut scheme = Scheme::new(&ctx);
        assert_eq!(
            scheme.arithmetic_policy(),
            ArithmeticPolicy::MatchRecorded,
            "recorded arithmetic is the default"
        );
        scheme.set_automatic_fusion(true);
        if arithmetic == ArithmeticPolicy::MatchRecorded {
            scheme.set_arithmetic_policy(arithmetic);
        }
        let a = Tensor::from_f32(&device, shape, &data(1, n)).unwrap();
        let b = Tensor::from_f32(&device, shape, &data(2, n)).unwrap();
        let out = Tensor::zeros(&device, shape, goldy::TensorDType::F32).unwrap();
        let scaled = {
            let mut rec = kernels.recorder(&mut scheme);
            let scaled = rec.mul("mul", a.view(), b.view()).unwrap();
            rec.add_into("add", scaled.view(), a.view(), out.view()).unwrap();
            scaled
        };
        // Changing policy after recording must switch recorded generated variants
        // as well as re-key and rebuild any fused replacement.
        scheme.set_arithmetic_policy(arithmetic);
        assert_eq!(scheme.arithmetic_policy(), arithmetic);
        schemes.push((scheme, scaled, a, b, out));
    }
    for (scheme, _, _, _, _) in &mut schemes {
        for _ in 0..24 {
            drop(scheme.submit().expect("submit"));
            goldy::test_support::wait_for_fusion_compiles(scheme);
            if scheme
                .fusion_report()
                .regions
                .iter()
                .any(|region| region.status == FusionRegionStatus::Promoted)
            {
                break;
            }
        }
    }
    assert!(
        schemes.iter().all(|(scheme, _, _, _, _)| {
            scheme
                .fusion_report()
                .regions
                .iter()
                .any(|region| region.status == FusionRegionStatus::Promoted)
        }),
        "every scheme should promote"
    );
    let ids: Vec<_> = schemes
        .iter()
        .map(|(scheme, _, _, _, _)| scheme.fusion_report().regions[0].kernel)
        .collect();
    assert_eq!(ids[0], ids[1], "exact schemes share one program");
    assert_ne!(ids[0], ids[2], "relaxed arithmetic is a distinct program");
    assert_eq!(goldy::test_support::fusion_compiles_started(&device), 2);
}

fn indexed_record(
    slot: u32,
) -> impl Fn(&mut TensorRecorder<'_>, &[Tensor]) -> Result<(Vec<Tensor>, Vec<NodeId>), GoldyError> {
    move |rec, t| {
        rec.matmul_indexed_into("gather", t[0].view(), t[1].view(), t[2].view(), slot, t[3].view())?;
        rec.mul_scalar_into("scale", t[3].view(), 2.0, t[4].view())?;
        Ok((Vec::new(), Vec::new()))
    }
}

/// Indexed GEMV plus a scale matches the unfused kernel byte for byte, including
/// an index past the slab count and a `k` that is not a multiple of 64.
#[test]
fn indexed_gemv_scale_matches_unfused() {
    let _gpu = gpu_lock();
    let device = runtime();
    let (slabs, rows, k, slot) = (3u32, 5u32, 40u32, 1u32);
    let mut ids = [0u32, 3, 2];
    let init = [
        Init::F32(
            TensorShape::from_dims(&[slabs, rows, k]).unwrap(),
            data(1, slabs * rows * k),
        ),
        Init::F32(TensorShape::vector(k), data(2, k)),
        Init::U32(TensorShape::vector(3), ids.to_vec()),
        Init::F32(TensorShape::vector(rows), vec![0.0; rows as usize]),
        Init::F32(TensorShape::vector(rows), vec![0.0; rows as usize]),
    ];
    let scheme = fused_matches_unfused(&device, &init, indexed_record(slot), 4);
    assert_one_semantic_region(&scheme, 2);

    // A second id, including u32::MAX, rewritten on the device without re-recording.
    ids = [u32::MAX, 1, 0];
    let mut twin = Twin::new(&device, &init, indexed_record(slot));
    twin.frames(2, "first ids");
    for (scheme, tensors, _) in &mut twin.runs {
        tensors[2]
            .buffer()
            .write(0, bytemuck::cast_slice(&ids))
            .expect("rewrite ids");
        let _ = scheme;
    }
    twin.frames(2, "rewritten ids");
    assert_one_semantic_region(twin.fused(), 2);
}

/// A nonzero weight offset is part of the address, and fused matches unfused.
#[test]
fn indexed_gemv_honors_a_weight_offset() {
    let _gpu = gpu_lock();
    let device = runtime();
    let (n_slabs, rows, k, slot, pad) = (2u32, 6u32, 20u32, 0u32, 7u32);
    let slab_bytes = data(4, n_slabs * rows * k);
    let mut weight = vec![0.0; pad as usize];
    weight.extend_from_slice(&slab_bytes);
    let init = [
        Init::F32(TensorShape::vector(pad + n_slabs * rows * k), weight),
        Init::F32(TensorShape::vector(k), data(5, k)),
        Init::U32(TensorShape::vector(1), vec![1]),
        Init::F32(TensorShape::vector(rows), vec![0.0; rows as usize]),
        Init::F32(TensorShape::vector(rows), vec![0.0; rows as usize]),
    ];
    let scheme = fused_matches_unfused(
        &device,
        &init,
        move |rec, t| {
            let mut desc = goldy::MatMulDesc::gemv(rows, k);
            desc.index = Some(goldy::ops::MatMulIndex {
                batch: 1,
                x_stride: 0,
                first: slot,
                slabs: n_slabs,
                slab_stride: rows * k,
                global_per_slab: false,
            });
            rec.scheme()
                .matmul("gather", desc)
                .a(t[0].buffer(), goldy::MatMulView::offset(u64::from(pad)))
                .b(t[1].buffer(), goldy::MatMulView::packed())
                .out(t[3].buffer(), goldy::MatMulView::packed())
                .extra(t[2].buffer(), goldy::MatMulView::packed())
                .record();
            rec.mul_scalar_into("scale", t[3].view(), 2.0, t[4].view())?;
            Ok((Vec::new(), Vec::new()))
        },
        3,
    );
    assert_one_semantic_region(&scheme, 2);
}

/// A run of slab indices is one contraction. It matches the unfused kernel with the
/// 64-wide main loop, a row count that is not a multiple of 4, a nonzero first index,
/// a weight offset, and a narrowed output. Rewriting the index vector replays it.
#[test]
fn gathered_batch_matches_unfused_and_replays() {
    let _gpu = gpu_lock();
    let device = runtime();
    let (n_slabs, rows, k, first, batch, pad, out_off) = (4u32, 5u32, 80u32, 1u32, 2u32, 3u32, 2u32);
    let slab_elems = n_slabs * rows * k;
    let mut weight = vec![0.0; pad as usize];
    weight.extend_from_slice(&data(4, slab_elems));
    let out_len = batch * rows;
    let out = vec![-1.0; (out_off + out_len + 2) as usize];
    let mut ids = vec![9u32, 1, 3, 0];
    let init = [
        Init::F32(TensorShape::vector(pad + slab_elems), weight),
        Init::F32(TensorShape::vector(k), data(5, k)),
        Init::U32(TensorShape::vector(4), ids.clone()),
        Init::F32(TensorShape::vector(out.len() as u32), out),
        Init::F32(
            TensorShape::from_dims(&[batch, rows]).unwrap(),
            vec![0.0; out_len as usize],
        ),
    ];
    let record = move |rec: &mut TensorRecorder<'_>, t: &[Tensor]| {
        let mut desc = goldy::MatMulDesc::gemv(rows, k);
        desc.index = Some(goldy::ops::MatMulIndex {
            first,
            slabs: n_slabs,
            slab_stride: rows * k,
            global_per_slab: false,
            batch,
            x_stride: 0,
        });
        rec.scheme()
            .matmul("gather", desc)
            .a(t[0].buffer(), goldy::MatMulView::offset(u64::from(pad)))
            .b(t[1].buffer(), goldy::MatMulView::packed())
            .out(t[3].buffer(), goldy::MatMulView::offset(u64::from(out_off)))
            .extra(t[2].buffer(), goldy::MatMulView::packed())
            .record();
        let window = t[3].view().narrow(0, out_off, out_len)?.reshape(&[batch, rows])?;
        rec.mul_scalar_into("scale", window, 2.0, t[4].view())?;
        Ok((Vec::new(), Vec::new()))
    };
    let mut twin = Twin::new(&device, &init, record);
    twin.frames(3, "gathered batch");
    assert_one_semantic_region(twin.fused(), 2);
    let (scheme, tensors, _) = &mut twin.runs[0];
    let mut sub = scheme.submit().expect("submit");
    let got = (&mut sub >> tensors[3].buffer()).take::<f32>().expect("take").to_vec();
    drop(sub);
    assert!(got[..out_off as usize].iter().all(|v| *v == -1.0), "prefix {got:?}");
    assert!(
        got[(out_off + out_len) as usize..].iter().all(|v| *v == -1.0),
        "suffix {got:?}"
    );
    assert!(got[out_off as usize..(out_off + out_len) as usize]
        .iter()
        .any(|v| *v != -1.0 && *v != 0.0));

    ids = vec![0, u32::MAX, 2, 1];
    for (_, tensors, _) in &mut twin.runs {
        tensors[2]
            .buffer()
            .write(0, bytemuck::cast_slice(&ids))
            .expect("rewrite indices");
    }
    twin.frames(2, "rewritten indices");
    assert_one_semantic_region(twin.fused(), 2);
}

/// Each index reads its own right-hand side, including a stride longer than `k`,
/// and a narrowed `x`.
#[test]
fn gathered_batch_honors_strided_inputs() {
    let _gpu = gpu_lock();
    let device = runtime();
    let (n_slabs, rows, k, batch, x_stride, x_off) = (3u32, 5u32, 80u32, 2u32, 84u32, 3u32);
    let x_need = (batch - 1) * x_stride + k;
    let mut xs = vec![0.0; (x_off + x_need) as usize];
    for (i, v) in data(6, x_need).into_iter().enumerate() {
        xs[x_off as usize + i] = v;
    }
    let init = [
        Init::F32(
            TensorShape::from_dims(&[n_slabs, rows, k]).unwrap(),
            data(7, n_slabs * rows * k),
        ),
        Init::F32(TensorShape::vector(xs.len() as u32), xs),
        Init::U32(TensorShape::vector(2), vec![2, 0]),
        Init::F32(
            TensorShape::from_dims(&[batch, rows]).unwrap(),
            vec![0.0; (batch * rows) as usize],
        ),
        Init::F32(
            TensorShape::from_dims(&[batch, rows]).unwrap(),
            vec![0.0; (batch * rows) as usize],
        ),
    ];
    let scheme = fused_matches_unfused(
        &device,
        &init,
        move |rec, t| {
            let mut desc = goldy::MatMulDesc::gemv(rows, k);
            desc.index = Some(goldy::ops::MatMulIndex {
                first: 0,
                slabs: n_slabs,
                slab_stride: rows * k,
                global_per_slab: false,
                batch,
                x_stride,
            });
            rec.scheme()
                .matmul("gather", desc)
                .a(t[0].buffer(), goldy::MatMulView::packed())
                .b(t[1].buffer(), goldy::MatMulView::strided(u64::from(x_off), 1))
                .out(t[3].buffer(), goldy::MatMulView::packed())
                .extra(t[2].buffer(), goldy::MatMulView::packed())
                .record();
            let view = t[3].view().reshape(&[batch, rows])?;
            rec.mul_scalar_into("scale", view, 3.0, t[4].view())?;
            Ok((Vec::new(), Vec::new()))
        },
        3,
    );
    assert_one_semantic_region(&scheme, 2);
}

/// Gathered products over the two halves of each slab of a stacked weight, and their
/// elementwise product, are one kernel that matches the three dispatches.
#[test]
fn gathered_halves_and_their_product_match_unfused() {
    let _gpu = gpu_lock();
    let device = runtime();
    let (n_slabs, rows, k, batch) = (4u32, 6u32, 40u32, 3u32);
    let out = || Init::F32(TensorShape::vector(batch * rows), vec![0.0; (batch * rows) as usize]);
    let init = [
        Init::F32(
            TensorShape::from_dims(&[n_slabs, 2 * rows, k]).unwrap(),
            data(13, n_slabs * 2 * rows * k),
        ),
        Init::F32(TensorShape::vector(k), data(14, k)),
        Init::U32(TensorShape::vector(batch), vec![3, 0, 2]),
        out(),
        out(),
        out(),
    ];
    let scheme = fused_matches_unfused(
        &device,
        &init,
        move |rec, t| {
            let half = |h: u32| t[0].view().narrow(1, h * rows, rows);
            rec.matmul_indexed_batch_into("first", half(0)?, t[1].view(), t[2].view(), 0, batch, 0, t[3].view())?;
            rec.matmul_indexed_batch_into("second", half(1)?, t[1].view(), t[2].view(), 0, batch, 0, t[4].view())?;
            let rows_of = |i: usize| t[i].view().reshape(&[batch, rows]);
            rec.mul_into("product", rows_of(3)?, rows_of(4)?, rows_of(5)?)?;
            Ok((Vec::new(), Vec::new()))
        },
        3,
    );
    assert_one_semantic_region(&scheme, 3);
}

/// Adding the rows of a gathered batch reads every slab at each output coordinate,
/// while a thread holds only its own slab index. The adds must not inline the
/// contraction at another batch position, and fused matches unfused.
#[test]
fn gathered_rows_summed_match_unfused() {
    let _gpu = gpu_lock();
    let device = runtime();
    let (n_slabs, rows, k, batch) = (4u32, 6u32, 40u32, 3u32);
    let init = [
        Init::F32(
            TensorShape::from_dims(&[n_slabs, rows, k]).unwrap(),
            data(8, n_slabs * rows * k),
        ),
        Init::F32(TensorShape::vector(k), data(9, k)),
        Init::U32(TensorShape::vector(batch), vec![2, 0, 3]),
        Init::F32(
            TensorShape::from_dims(&[batch, rows]).unwrap(),
            vec![0.0; (batch * rows) as usize],
        ),
        Init::F32(TensorShape::vector(rows), vec![0.0; rows as usize]),
    ];
    let record = move |rec: &mut TensorRecorder<'_>, t: &[Tensor]| {
        let mut desc = goldy::MatMulDesc::gemv(rows, k);
        desc.index = Some(goldy::ops::MatMulIndex {
            first: 0,
            slabs: n_slabs,
            slab_stride: rows * k,
            global_per_slab: false,
            batch,
            x_stride: 0,
        });
        rec.scheme()
            .matmul("gather", desc)
            .a(t[0].buffer(), goldy::MatMulView::packed())
            .b(t[1].buffer(), goldy::MatMulView::packed())
            .out(t[3].buffer(), goldy::MatMulView::packed())
            .extra(t[2].buffer(), goldy::MatMulView::packed())
            .record();
        rec.fill("clear", t[4].view(), goldy::TensorScalar::F32(0.0))?;
        for b in 0..batch {
            let row = t[3].view().narrow(0, b, 1)?.reshape(&[rows])?;
            rec.add_into("sum", t[4].view(), row, t[4].view())?;
        }
        Ok((Vec::new(), Vec::new()))
    };
    let mut twin = Twin::new(&device, &init, record);
    twin.frames(3, "summed rows");
    let (scheme, tensors, _) = &mut twin.runs[0];
    let mut sub = scheme.submit().expect("submit");
    let got = (&mut sub >> tensors[4].buffer()).take::<f32>().expect("take").to_vec();
    drop(sub);
    assert!(got.iter().any(|v| *v != 0.0), "{got:?}");
}

/// Run `record` with specialization on until every scheme runs its variants, then
/// compare `frames` frames; returns the fusing scheme.
fn specialized_matches_unfused(
    device: &Runtime,
    init: &[Init],
    record: impl Fn(&mut TensorRecorder<'_>, &[Tensor]) -> Recorded,
    frames: usize,
) -> Scheme {
    let mut twin = Twin::specialized(device, init, record);
    twin.settle();
    twin.frames(frames, "specialized");
    twin.runs.swap_remove(0).0
}

/// Specialization bakes a fused kernel's scalars as literals. A divisor that becomes
/// a constant still divides on its own instead of folding into the function that
/// reads the quotient.
#[test]
fn specialized_scalar_divisor_matches_unfused() {
    let _gpu = gpu_lock();
    let device = runtime();
    let n = 256u32;
    let x: Vec<f32> = data(11, n).into_iter().map(|v| v * 40.0).collect();
    let init = [
        Init::F32(TensorShape::vector(n), x),
        Init::F32(TensorShape::vector(n), vec![0.0; n as usize]),
        Init::F32(TensorShape::vector(n), vec![0.0; n as usize]),
        Init::F32(TensorShape::vector(n), vec![0.0; n as usize]),
    ];
    let scheme = specialized_matches_unfused(
        &device,
        &init,
        |rec, t| {
            rec.div_scalar_into("divide", t[0].view(), 25.0, t[1].view())?;
            rec.tanh_into("tanh", t[1].view(), t[2].view())?;
            rec.mul_scalar_into("scale", t[2].view(), 25.0, t[3].view())?;
            Ok((Vec::new(), Vec::new()))
        },
        3,
    );
    assert_one_semantic_region(&scheme, 3);
}

/// Scalar offsets, divisors and factors in a row, each baked as a literal: no two
/// constants combine across the operations.
#[test]
fn specialized_scalar_chain_matches_unfused() {
    let _gpu = gpu_lock();
    let device = runtime();
    let n = 256u32;
    let x: Vec<f32> = data(12, n).into_iter().map(|v| v * 3.0).collect();
    let mut init = vec![Init::F32(TensorShape::vector(n), x)];
    for _ in 0..7 {
        init.push(Init::F32(TensorShape::vector(n), vec![0.0; n as usize]));
    }
    let scheme = specialized_matches_unfused(
        &device,
        &init,
        |rec, t| {
            rec.add_scalar_into("offset", t[0].view(), 0.1, t[1].view())?;
            rec.add_scalar_into("offset", t[1].view(), 0.7, t[2].view())?;
            rec.sub_scalar_into("offset", t[2].view(), 0.3, t[3].view())?;
            rec.div_scalar_into("divide", t[3].view(), 3.0, t[4].view())?;
            rec.div_scalar_into("divide", t[4].view(), 7.0, t[5].view())?;
            rec.mul_scalar_into("scale", t[5].view(), 0.1, t[6].view())?;
            rec.mul_scalar_into("scale", t[6].view(), 0.3, t[7].view())?;
            Ok((Vec::new(), Vec::new()))
        },
        3,
    );
    assert_one_semantic_region(&scheme, 7);
}

fn sum_inputs(rec: &mut TensorRecorder<'_>, t: &[Tensor]) -> Result<(Vec<Tensor>, Vec<NodeId>), GoldyError> {
    let last = t.len() - 1;
    rec.add_into("sum", t[0].view(), t[1].view(), t[last].view())?;
    for i in 2..last {
        rec.add_into("sum", t[last].view(), t[i].view(), t[last].view())?;
    }
    Ok((Vec::new(), Vec::new()))
}

/// Eighteen inputs plus the accumulator is inside the 24-resource dispatch cap
/// and matches the unfused adds byte for byte.
#[test]
fn eighteen_inputs_fuse_into_one_kernel() {
    let _gpu = gpu_lock();
    let device = runtime();
    let n = 32u32;
    let inputs = 18u32;
    let mut init = Vec::new();
    for i in 0..inputs {
        init.push(Init::F32(TensorShape::vector(n), data(i + 1, n)));
    }
    init.push(Init::F32(TensorShape::vector(n), vec![0.0; n as usize]));
    let scheme = fused_matches_unfused(&device, &init, sum_inputs, 3);
    assert_eq!(scheme.executed_node_count(), 1, "{:?}", scheme.fusion_report());
}
