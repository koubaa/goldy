//! Semantic fusion: runs of tensor operations and matrix products synthesized as one
//! kernel. Every trial records the same computation into a fusing scheme and an
//! unfused one over identical tensors, and requires every tensor to match byte for
//! byte after every frame.

#![cfg(all(feature = "gpu", feature = "tensor"))]

#[path = "common/submission.rs"]
mod submission;

use goldy::{
    compute, BackendType, ContractionPrecision, FusionRegionStatus, FusionSchedule, FusionTier, GoldyError, NodeId,
    RequestAdapterOptions, Runtime, RuntimeDescriptor, ScatterMode, Scheme, Tensor, TensorKernels, TensorRecorder,
    TensorShape,
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
}

fn tensors(device: &Runtime, init: &[Init]) -> Vec<Tensor> {
    init.iter()
        .map(|i| match i {
            Init::F32(shape, v) => Tensor::from_f32(device, *shape, v),
            Init::I32(shape, v) => Tensor::from_i32(device, *shape, v),
        })
        .collect::<Result<_, _>>()
        .expect("tensor")
}

/// Records over the initial tensors; returns the tensors it allocates and the nodes a
/// trial may change.
type Record = fn(&mut TensorRecorder<'_>, &[Tensor]) -> Result<(Vec<Tensor>, Vec<NodeId>), GoldyError>;

/// The same recording in a fusing scheme and an unfused one.
struct Twin {
    _specialization: goldy::test_support::SpecializationOverride,
    /// Fusing first: each scheme, its tensors and the nodes `record` returned.
    runs: Vec<(Scheme, Vec<Tensor>, Vec<NodeId>)>,
}

impl Twin {
    fn new(device: &Runtime, init: &[Init], record: Record) -> Self {
        let specialization = goldy::test_support::SpecializationOverride::force_disabled();
        let ctx = submission::submission_context(device);
        let kernels = TensorKernels::new(device).expect("tensor kernels");
        let runs = [true, false]
            .into_iter()
            .map(|fusing| {
                let mut scheme = Scheme::new(&ctx);
                scheme.set_automatic_fusion(fusing);
                let mut t = tensors(device, init);
                let (allocated, nodes) = record(&mut kernels.recorder(&mut scheme), &t).expect("record");
                t.extend(allocated);
                (scheme, t, nodes)
            })
            .collect();
        Self {
            _specialization: specialization,
            runs,
        }
    }

    fn fused(&self) -> &Scheme {
        &self.runs[0].0
    }

    /// Submit `frames` frames, comparing the initial and the allocated tensors.
    fn frames(&mut self, frames: usize, what: &str) {
        for frame in 1..=frames {
            let parcels: Vec<Vec<Vec<u8>>> = self
                .runs
                .iter_mut()
                .map(|(scheme, t, _)| {
                    let mut sub = scheme.submit().expect("submit");
                    let got = t
                        .iter()
                        .map(|t| (&mut sub >> t.buffer()).take::<u8>().expect("take").to_vec())
                        .collect();
                    drop(sub);
                    goldy::test_support::wait_for_fusion_compiles(scheme);
                    got
                })
                .collect();
            for (j, (fused, unfused)) in parcels[0].iter().zip(&parcels[1]).enumerate() {
                assert!(fused == unfused, "{what}, frame {frame}: tensor {j} differs");
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
fn fused_matches_unfused(device: &Runtime, init: &[Init], record: Record, frames: usize) -> Scheme {
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
fn stdlib_matmul(device: &Runtime) -> bool {
    device.backend_type() != BackendType::Metal
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
