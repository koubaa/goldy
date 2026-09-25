//! Kernel fusion: every fused dispatch is compared against its unfused sequence, and
//! value forwarding against conservative fusion, on every bound parcel, not only the
//! final output. Automatic fusion of retained schemes is checked frame by frame.

#![cfg(feature = "gpu")]
#![allow(clippy::too_many_arguments)]

#[path = "common/submission.rs"]
mod submission;

use goldy::{
    compute, AccessKind, BackendType, Buffer, BufferKind, FusedKernel, FusionError, FusionRegionStatus,
    FusionRejection, Instance, Invocation, RequestAdapterOptions, Runtime, RuntimeDescriptor, Scheme,
};
use std::sync::Arc;

/// Threads per stage; not a multiple of the workgroup so bounds guards matter.
const N: u32 = 1000;
/// Parcel length; elements past `N` must keep their sentinel.
const LEN: usize = 1024;
const SENTINEL: f32 = -7.0;

#[compute(workgroup_size = [64, 1, 1])]
fn scale(input: &[f32], output: goldy::gpu::Scattered<f32>, count: u32, factor: f32) {
    let i = goldy::gpu::global_id().x;
    if i >= count {
        return;
    }
    output[i] = input[i] * factor;
}

#[compute(workgroup_size = [64, 1, 1])]
fn bias(input: &[f32], output: goldy::gpu::Scattered<f32>, count: u32, bias: f32) {
    let i = goldy::gpu::global_id().x;
    if i < count {
        output[i] = input[i] + bias;
    }
}

#[compute(workgroup_size = [64, 1, 1])]
fn square(input: &[f32], output: goldy::gpu::Scattered<f32>, count: u32) {
    let i = goldy::gpu::global_id().x;
    if i < count {
        let v = input[i];
        output[i] = v * v;
    }
}

#[compute(workgroup_size = [64, 1, 1])]
fn cube(input: &[f32], output: goldy::gpu::Scattered<f32>, count: u32) {
    let i = goldy::gpu::global_id().x;
    if i < count {
        let v = input[i];
        output[i] = v * v * v;
    }
}

#[compute(workgroup_size = [64, 1, 1])]
fn add(a: &[f32], b: &[f32], output: goldy::gpu::Scattered<f32>, count: u32) {
    let i = goldy::gpu::global_id().x;
    if i < count {
        output[i] = a[i] + b[i];
    }
}

#[compute(workgroup_size = [64, 1, 1])]
fn scale_in_place(data: &mut [f32], count: u32, factor: f32) {
    let i = goldy::gpu::global_id().x;
    if i < count {
        data[i] = data[i] * factor;
    }
}

#[compute(workgroup_size = [64, 1, 1])]
fn clamp_in_place(data: &mut [f32], count: u32, low: f32, high: f32) {
    let i = goldy::gpu::global_id().x;
    if i < count {
        data[i] = goldy::gpu::min(goldy::gpu::max(data[i], low), high);
    }
}

/// One partial sum per workgroup, through workgroup-shared memory and barriers.
#[compute(workgroup_size = [64, 1, 1])]
fn block_sum(data: &[f32], sums: goldy::gpu::Scattered<f32>, count: u32) {
    let mut scratch = goldy::gpu::workgroup_array::<f32, 64>();
    let i = goldy::gpu::global_id().x;
    let mut v = 0.0;
    if i < count {
        v = data[i];
    }
    let total = goldy::gpu::workgroup_sum::<64>(v, scratch);
    if goldy::gpu::local_id().x == 0 {
        sums[goldy::gpu::workgroup_id().x] = total;
    }
}

/// Reads its neighbour's element: not fusible after a producer of `input`.
#[compute(workgroup_size = [64, 1, 1])]
fn shift(input: &[f32], output: goldy::gpu::Scattered<f32>, count: u32) {
    let i = goldy::gpu::global_id().x;
    if i + 1 < count {
        output[i] = input[i + 1];
    }
}

/// `enabled` gates a loop; baking it to zero removes the loop from the fused program.
#[compute(workgroup_size = [64, 1, 1])]
fn damp(input: &[f32], output: goldy::gpu::Scattered<f32>, count: u32, enabled: u32, rounds: u32) {
    let i = goldy::gpu::global_id().x;
    if i < count {
        let mut v = input[i];
        if enabled != 0 {
            for _r in 0..rounds {
                v = v * 0.5 + 1.0;
            }
        }
        output[i] = v;
    }
}

#[compute(workgroup_size = [128, 1, 1])]
fn bias_wide(input: &[f32], output: goldy::gpu::Scattered<f32>, count: u32, bias: f32) {
    let i = goldy::gpu::global_id().x;
    if i < count {
        output[i] = input[i] + bias;
    }
}

fn input_data() -> Vec<f32> {
    (0..LEN).map(|i| (i % 17) as f32 - 8.0).collect()
}

fn sentinel() -> Vec<f32> {
    vec![SENTINEL; LEN]
}

fn buffers(device: &Runtime, init: &[Vec<f32>]) -> anyhow::Result<Vec<Buffer>> {
    init.iter()
        .map(|d| device.acquire_buffer_with_data(d, BufferKind::Scattered))
        .collect()
}

/// Submit what `record` records and read every parcel in `parcels` back.
fn submit(
    device: &Runtime,
    parcels: &[Buffer],
    record: impl FnOnce(&mut Scheme) -> anyhow::Result<()>,
) -> anyhow::Result<(usize, Vec<Vec<u8>>)> {
    let ctx = device.create_context()?;
    let mut scheme = Scheme::new(&ctx);
    scheme.set_automatic_fusion(false);
    record(&mut scheme)?;
    let nodes = scheme.ir_node_count();
    let mut frame = scheme.submit()?;
    let mut bytes = Vec::new();
    for p in parcels {
        bytes.push((&mut frame >> p).take::<u8>()?.to_vec());
    }
    Ok((nodes, bytes))
}

type Stages<K> = for<'a> fn(&'a K, &'a [Buffer]) -> Vec<Invocation<'a>>;

/// Run `stages` unfused, conservatively fused and fused with forwarding on copies of
/// `init`; require identical bytes in every parcel. Returns the forwarding fused
/// kernel and the final parcel contents.
fn fuse_and_compare<K>(
    device: &Runtime,
    kernels: &K,
    init: &[Vec<f32>],
    stages: Stages<K>,
) -> anyhow::Result<(FusedKernel, Vec<Vec<f32>>)> {
    let unfused_parcels = buffers(device, init)?;
    let unfused = stages(kernels, &unfused_parcels);
    let (unfused_nodes, want) = submit(device, &unfused_parcels, |scheme| {
        for (k, stage) in unfused.iter().enumerate() {
            stage.record(scheme, format!("stage{k}"))?;
        }
        Ok(())
    })?;
    assert_eq!(unfused_nodes, unfused.len());

    let run_fused = |prepare: fn(&Runtime, &[Invocation<'_>]) -> Result<FusedKernel, FusionError>,
                     what: &str|
     -> anyhow::Result<(FusedKernel, Vec<Vec<u8>>)> {
        let parcels = buffers(device, init)?;
        let fused_stages = stages(kernels, &parcels);
        let fused = prepare(device, &fused_stages)?;
        let (fused_nodes, got) = submit(device, &parcels, |scheme| {
            fused.record(scheme, "fused", &fused_stages)?;
            Ok(())
        })?;
        assert_eq!(fused_nodes, 1, "{what} stages must record one dispatch");
        assert_eq!(fused.def().source.canonical_slang.matches("[goldy_compute]").count(), 1);
        assert_same_parcels(&want, &got, what);
        Ok((fused, got))
    };
    let (conservative, _) = run_fused(FusedKernel::prepare_conservative, "conservative")?;
    assert!(conservative.definition().forwarded.is_empty());
    let (fused, got) = run_fused(FusedKernel::prepare, "forwarded")?;
    assert_eq!(
        fused.id() == conservative.id(),
        fused.definition().forwarded.is_empty(),
        "forwarding is part of the fused identity"
    );

    let values = got.iter().map(|b| bytemuck::cast_slice(b).to_vec()).collect();
    Ok((fused, values))
}

fn assert_same_parcels(want: &[Vec<u8>], got: &[Vec<u8>], what: &str) {
    for (p, (w, g)) in want.iter().zip(got).enumerate() {
        assert_eq!(w.len(), g.len(), "{what}: parcel {p} size");
        if let Some(i) = w.chunks(4).zip(g.chunks(4)).position(|(a, b)| a != b) {
            panic!(
                "{what}: parcel {p} element {i}: unfused {:?} vs fused {:?}",
                f32::from_le_bytes(w[i * 4..i * 4 + 4].try_into().unwrap()),
                f32::from_le_bytes(g[i * 4..i * 4 + 4].try_into().unwrap()),
            );
        }
    }
}

fn rejection(result: Result<FusedKernel, FusionError>) -> FusionRejection {
    match result {
        Err(FusionError::Rejected(r)) => r,
        Err(FusionError::Compile(e)) => panic!("expected a rejection, got a compile error: {e:#}"),
        Ok(_) => panic!("expected a rejection, got a fused kernel"),
    }
}

struct Chain {
    scale: scale::Kernel,
    bias: bias::Kernel,
}

/// `scale` covers only the first half, so a `return` that left the fused entry would
/// skip `bias` for the second half.
fn chain<'a>(k: &'a Chain, p: &'a [Buffer]) -> Vec<Invocation<'a>> {
    vec![
        k.scale.invoke(&p[0], &p[1], N / 2, 2.0).over_1d(N),
        k.bias.invoke(&p[1], &p[2], N, 1.0).over_1d(N),
    ]
}

struct ForkJoin {
    square: square::Kernel,
    cube: cube::Kernel,
    add: add::Kernel,
}

fn fork_join<'a>(k: &'a ForkJoin, p: &'a [Buffer]) -> Vec<Invocation<'a>> {
    vec![
        k.square.invoke(&p[0], &p[1], N).over_1d(N),
        k.cube.invoke(&p[0], &p[2], N).over_1d(N),
        k.add.invoke(&p[1], &p[2], &p[3], N).over_1d(N),
    ]
}

struct InPlace {
    scale: scale_in_place::Kernel,
    clamp: clamp_in_place::Kernel,
}

fn in_place<'a>(k: &'a InPlace, p: &'a [Buffer]) -> Vec<Invocation<'a>> {
    vec![
        k.scale.invoke(&p[0], N, 3.0).over_1d(N),
        k.clamp.invoke(&p[0], N, -10.0, 12.0).over_1d(N),
    ]
}

struct WithCollective {
    scale: scale::Kernel,
    block_sum: block_sum::Kernel,
}

fn with_collective<'a>(k: &'a WithCollective, p: &'a [Buffer]) -> Vec<Invocation<'a>> {
    vec![
        k.scale.invoke(&p[0], &p[1], N, 4.0).over_1d(N),
        k.block_sum.invoke(&p[0], &p[2], N).over_1d(N),
    ]
}

struct Gated {
    scale: scale::Kernel,
    damp: damp::Kernel,
}

const ROUNDS: u32 = 6;

fn gated<'a>(k: &'a Gated, p: &'a [Buffer]) -> Vec<Invocation<'a>> {
    vec![
        k.scale.invoke(&p[0], &p[1], N, 2.0).over_1d(N),
        k.damp.invoke(&p[1], &p[2], N, 1, ROUNDS).over_1d(N),
    ]
}

/// Every parcel of [`gated`] after one dispatch, computed on the host.
fn gated_reference(enabled: bool) -> Vec<Vec<f32>> {
    let input = input_data();
    let n = N as usize;
    let temporary: Vec<f32> = (0..LEN)
        .map(|i| if i < n { input[i] * 2.0 } else { SENTINEL })
        .collect();
    let output = (0..LEN)
        .map(|i| {
            if i >= n {
                return SENTINEL;
            }
            let mut v = temporary[i];
            for _ in 0..if enabled { ROUNDS } else { 0 } {
                v = v * 0.5 + 1.0;
            }
            v
        })
        .collect();
    vec![input, temporary, output]
}

/// Submit `scheme`, require every parcel to match `want` byte for byte, and let any
/// specialization compile the submit started land before the next frame.
fn submit_and_check(scheme: &mut Scheme, parcels: &[Buffer], want: &[Vec<f32>], what: &str) -> anyhow::Result<()> {
    {
        let mut frame = scheme.submit()?;
        for (j, (parcel, want)) in parcels.iter().zip(want).enumerate() {
            let got = (&mut frame >> parcel).take::<u8>()?.to_vec();
            assert!(
                got == bytemuck::cast_slice::<f32, u8>(want),
                "{what}: parcel {j} differs"
            );
        }
    }
    goldy::test_support::wait_for_specialization_compiles(scheme);
    goldy::test_support::wait_for_fusion_compiles(scheme);
    Ok(())
}

/// Every parcel of [`chain`] with `bias` for the second stage, then [`shift`] of its
/// output when `shifted`.
fn chain_reference(bias: f32, shifted: bool) -> Vec<Vec<f32>> {
    let input = input_data();
    let n = N as usize;
    let temporary: Vec<f32> = (0..LEN)
        .map(|i| if i < n / 2 { input[i] * 2.0 } else { SENTINEL })
        .collect();
    let output: Vec<f32> = (0..LEN)
        .map(|i| if i < n { temporary[i] + bias } else { SENTINEL })
        .collect();
    let mut parcels = vec![input, temporary, output.clone()];
    if shifted {
        parcels.push(
            (0..LEN)
                .map(|i| if i + 1 < n { output[i + 1] } else { SENTINEL })
                .collect(),
        );
    }
    parcels
}

/// Pins for a retained scheme whose record counts the automatic fusion trials assert.
fn auto_fusion_pins() -> (
    goldy::test_support::SpecializationOverride,
    goldy::test_support::CbReuseOverride,
) {
    (
        goldy::test_support::SpecializationOverride::force_disabled(),
        goldy::test_support::CbReuseOverride::force_enabled(),
    )
}

/// A scheme that fuses automatically whatever `GOLDY_FUSION` says.
fn fusing(ctx: &goldy::Context) -> Scheme {
    let mut scheme = Scheme::new(ctx);
    scheme.set_automatic_fusion(true);
    scheme
}

/// `((input * 2 + 1) * 3 + 1)` for the first `stages` of that sequence, past `N` the sentinel.
fn temporary_chain_reference(stages: usize) -> Vec<f32> {
    let ops: [fn(f32) -> f32; 4] = [|x| x * 2.0, |x| x + 1.0, |x| x * 3.0, |x| x + 1.0];
    input_data()
        .into_iter()
        .enumerate()
        .map(|(i, x)| {
            if i < N as usize {
                ops[..stages].iter().fold(x, |x, op| op(x))
            } else {
                SENTINEL
            }
        })
        .collect()
}

/// Record `input → scale → t0 → bias → t1 → scale → t2 → bias → output` over temporaries.
fn record_temporary_chain(scheme: &mut Scheme, k: &Chain, p: &[Buffer]) -> anyhow::Result<()> {
    let t: Vec<_> = (0..3)
        .map(|_| scheme.temporary_buffer::<f32>(LEN))
        .collect::<Result<_, _>>()?;
    k.scale.invoke(&p[0], &t[0], N, 2.0).over_1d(N).record(scheme, "a")?;
    k.bias.invoke(&t[0], &t[1], N, 1.0).over_1d(N).record(scheme, "b")?;
    k.scale.invoke(&t[1], &t[2], N, 3.0).over_1d(N).record(scheme, "c")?;
    k.bias.invoke(&t[2], &p[1], N, 1.0).over_1d(N).record(scheme, "d")?;
    Ok(())
}

/// Record [`chain`] as two recorded dispatches; returns the second node.
fn record_chain(scheme: &mut Scheme, k: &Chain, p: &[Buffer]) -> anyhow::Result<goldy::NodeId> {
    let [scale, bias] = <[_; 2]>::try_from(chain(k, p)).map_err(|_| anyhow::anyhow!("two stages"))?;
    scale.record(scheme, "scale")?;
    Ok(bias.record(scheme, "bias")?.node())
}

fn main() {
    let mut args = libtest_mimic::Arguments::from_args();
    let instance = Instance::new().expect("instance");
    let device = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_runtime(&RuntimeDescriptor::default())
        .expect("device");
    submission::clamp_test_threads(&mut args, &device);
    let device = Arc::new(device);

    let tests = vec![
        libtest_mimic::Trial::test("fusion_pointwise_chain_gpu", {
            let device = Arc::clone(&device);
            move || {
                let k = Chain {
                    scale: scale::Kernel::prepare(&device)?,
                    bias: bias::Kernel::prepare(&device)?,
                };
                let init = [input_data(), sentinel(), sentinel()];
                let (fused, got) = fuse_and_compare(&device, &k, &init, chain)?;
                assert_eq!(
                    fused.definition().forwarded,
                    [1],
                    "the intermediate stays in a register"
                );

                let params = &fused.definition().params;
                let access: Vec<_> = params.iter().map(|p| (p.name.as_str(), p.access)).collect();
                assert_eq!(
                    access,
                    [
                        ("k0_input", Some(AccessKind::Read)),
                        ("k0_output", Some(AccessKind::ReadWrite)),
                        ("k0_count", None),
                        ("k0_factor", None),
                        ("k1_output", Some(AccessKind::Write)),
                        ("k1_count", None),
                        ("k1_bias", None),
                    ]
                );
                let input = input_data();
                for i in 0..LEN {
                    let temporary = if i < (N / 2) as usize { input[i] * 2.0 } else { SENTINEL };
                    let output = if i < N as usize { temporary + 1.0 } else { SENTINEL };
                    assert_eq!(got[1][i], temporary, "temporary[{i}]");
                    assert_eq!(got[2][i], output, "output[{i}]");
                }
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("fusion_fork_join_gpu", {
            let device = Arc::clone(&device);
            move || {
                let k = ForkJoin {
                    square: square::Kernel::prepare(&device)?,
                    cube: cube::Kernel::prepare(&device)?,
                    add: add::Kernel::prepare(&device)?,
                };
                let init = [input_data(), sentinel(), sentinel(), sentinel()];
                let (fused, got) = fuse_and_compare(&device, &k, &init, fork_join)?;
                // k0_input, k0_output, k0_count, k1_output, ...: both branch outputs reach the join.
                assert_eq!(fused.definition().forwarded, [1, 3]);

                let resources: Vec<_> = fused
                    .definition()
                    .params
                    .iter()
                    .filter(|p| p.category.is_resource())
                    .map(|p| (p.name.as_str(), p.access))
                    .collect();
                assert_eq!(
                    resources,
                    [
                        ("k0_input", Some(AccessKind::Read)),
                        ("k0_output", Some(AccessKind::ReadWrite)),
                        ("k1_output", Some(AccessKind::ReadWrite)),
                        ("k2_output", Some(AccessKind::Write)),
                    ],
                    "the shared input binds once"
                );
                let input = input_data();
                for i in 0..N as usize {
                    let v = input[i];
                    assert_eq!(got[1][i], v * v, "a[{i}]");
                    assert_eq!(got[2][i], v * v * v, "b[{i}]");
                    assert_eq!(got[3][i], v * v + v * v * v, "output[{i}]");
                }
                assert!(got[1..].iter().all(|p| p[N as usize..].iter().all(|&x| x == SENTINEL)));
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("fusion_in_place_sequence_gpu", {
            let device = Arc::clone(&device);
            move || {
                let k = InPlace {
                    scale: scale_in_place::Kernel::prepare(&device)?,
                    clamp: clamp_in_place::Kernel::prepare(&device)?,
                };
                let init = [input_data()];
                let (fused, got) = fuse_and_compare(&device, &k, &init, in_place)?;
                assert_eq!(
                    fused.definition().forwarded,
                    [0],
                    "one register carries the in-place value"
                );

                let params = &fused.definition().params;
                assert_eq!(params[0].name, "k0_data");
                assert_eq!(params[0].access, Some(AccessKind::ReadWrite));
                assert_eq!(params.iter().filter(|p| p.category.is_resource()).count(), 1);
                let input = input_data();
                for i in 0..LEN {
                    let want = if i < N as usize {
                        (input[i] * 3.0).clamp(-10.0, 12.0)
                    } else {
                        input[i]
                    };
                    assert_eq!(got[0][i], want, "data[{i}]");
                }
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("fusion_with_workgroup_collective_gpu", {
            let device = Arc::clone(&device);
            move || {
                let k = WithCollective {
                    scale: scale::Kernel::prepare(&device)?,
                    block_sum: block_sum::Kernel::prepare(&device)?,
                };
                let init = [input_data(), sentinel(), sentinel()];
                let (fused, got) = fuse_and_compare(&device, &k, &init, with_collective)?;

                assert_eq!(fused.definition().dependence_rank, None);
                assert!(
                    fused.definition().forwarded.is_empty(),
                    "the stages only share an input"
                );
                let slang = &fused.def().source.canonical_slang;
                assert!(slang.contains("groupshared float _goldy_k1_scratch[64];"), "{slang}");
                let input = input_data();
                for g in 0..N.div_ceil(64) as usize {
                    let end = ((g + 1) * 64).min(N as usize);
                    let want: f32 = input[g * 64..end].iter().sum();
                    assert_eq!(got[2][g], want, "sums[{g}]");
                }
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("fusion_specializes_and_demotes_gpu", {
            let device = Arc::clone(&device);
            move || {
                let _spec = goldy::test_support::SpecializationOverride::force_enabled();
                // WebGPU layouts follow shader usage and CPU has no bake macros; neither predicts.
                let predicts = !matches!(device.backend_type(), BackendType::WebGpu | BackendType::Cpu);
                let k = Gated {
                    scale: scale::Kernel::prepare(&device)?,
                    damp: damp::Kernel::prepare(&device)?,
                };
                let init = [input_data(), sentinel(), sentinel()];
                let enabled = gated_reference(true);
                let disabled = gated_reference(false);
                assert_ne!(enabled[2], disabled[2], "the gate must change the output");

                // The universal fused program matches the unfused sequence.
                let (fused, universal) = fuse_and_compare(&device, &k, &init, gated)?;
                assert_eq!(universal, enabled);
                let gate = fused.scalar_slot(1, "enabled").expect("stage 1 binds `enabled`");
                assert_eq!(gate, 3, "after k0_count, k0_factor, k1_count");
                let origins = fused.definition().scalar_origins();
                assert_eq!(origins[gate].to_string(), "1:damp.enabled");
                assert_eq!(origins[gate].slot, 1);

                let p = buffers(&device, &init)?;
                let ctx = device.create_context()?;
                let mut scheme = Scheme::new(&ctx);
                scheme.set_automatic_fusion(false);
                let node = fused.record(&mut scheme, "scale+damp", &gated(&k, &p))?.node();

                // Frame 1 records; stable fused slots warm on frame 3 and promote on frame 11.
                for f in 1..=11 {
                    submit_and_check(&mut scheme, &p, &enabled, &format!("frame {f}"))?;
                }
                let stats = scheme.replay_stats();
                if predicts {
                    assert_eq!(stats.specialization_warms, 1, "one variant for the fused site");
                    assert_eq!(stats.specialization_promotions, 1);
                    assert!(scheme.node_is_specialized(node));
                } else {
                    assert_eq!(stats.specialization_warms, 0, "this backend declines to predict");
                }
                submit_and_check(&mut scheme, &p, &enabled, "specialized fused program")?;

                // Changing a baked scalar demotes to the universal fused program at once.
                scheme.set_node_param(node, gate, 0)?;
                assert!(!scheme.node_is_specialized(node), "demoted inside set_node_param");
                if predicts {
                    assert_eq!(scheme.replay_stats().specialization_demotions, 1);
                }
                submit_and_check(&mut scheme, &p, &disabled, "first frame after the gate changed")?;
                assert_eq!(scheme.ir_node_count(), 1, "demotion keeps the dispatch fused");

                // Left alone, the site re-earns a variant for the new facts.
                for f in 1..=25 {
                    submit_and_check(&mut scheme, &p, &disabled, &format!("post-demotion frame {f}"))?;
                }
                if predicts {
                    assert!(scheme.node_is_specialized(node), "re-promoted with the gate off");
                    assert_eq!(scheme.replay_stats().specialization_demotions, 1);
                }
                assert_eq!(scheme.ir_node_count(), 1);
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("fusion_rejects_incompatible_invocations_gpu", {
            let device = Arc::clone(&device);
            move || {
                let chain_k = Chain {
                    scale: scale::Kernel::prepare(&device)?,
                    bias: bias::Kernel::prepare(&device)?,
                };
                let shift_k = shift::Kernel::prepare(&device)?;
                let wide = bias_wide::Kernel::prepare(&device)?;
                let p = buffers(&device, &[input_data(), sentinel(), sentinel(), sentinel()])?;
                let produce = chain_k.scale.invoke(&p[0], &p[1], N, 2.0).over_1d(N);

                let r = rejection(FusedKernel::prepare(
                    &device,
                    &[produce.clone(), wide.invoke(&p[1], &p[2], N, 1.0).over_1d(N)],
                ));
                assert!(matches!(r, FusionRejection::WorkgroupSize { stage: 1, .. }), "{r}");

                let r = rejection(FusedKernel::prepare(
                    &device,
                    &[
                        produce.clone(),
                        chain_k.bias.invoke(&p[1], &p[2], N, 1.0).over_1d(N / 2),
                    ],
                ));
                assert!(matches!(r, FusionRejection::Grid { stage: 1, .. }), "{r}");

                let r = rejection(FusedKernel::prepare(
                    &device,
                    &[produce.clone(), shift_k.invoke(&p[1], &p[2], N).over_1d(N)],
                ));
                assert!(
                    matches!(&r, FusionRejection::NonLocalDependence { stage: 1, formal, .. } if formal == "input"),
                    "{r}"
                );

                // `gid.x` names one thread per x only while the grid is one row deep.
                let r = rejection(FusedKernel::prepare(
                    &device,
                    &[
                        chain_k.scale.invoke(&p[0], &p[1], N, 2.0).groups([16, 2, 1]),
                        chain_k.bias.invoke(&p[1], &p[2], N, 1.0).groups([16, 2, 1]),
                    ],
                ));
                assert!(matches!(r, FusionRejection::GridRank { rank: 1, .. }), "{r}");

                let opaque_def =
                    goldy::slang::try_kernel_def_from_source(scale::CANONICAL_SOURCE).expect("parse canonical source");
                let opaque = goldy::prepare_kernel(&device, opaque_def)?;
                let r = rejection(FusedKernel::prepare(
                    &device,
                    &[
                        produce.clone(),
                        opaque
                            .invoke()
                            .resource(&p[1])
                            .resource(&p[2])
                            .bind_u32(N)
                            .bind_f32(1.0)
                            .over_1d(N),
                    ],
                ));
                assert!(matches!(r, FusionRejection::OpaqueDefinition { stage: 1, .. }), "{r}");

                // A fused kernel only records invocations that share parcels as it was prepared.
                let prepared_for = chain(&chain_k, &p);
                let fused = FusedKernel::prepare(&device, &prepared_for)?;
                let unshared = [
                    chain_k.scale.invoke(&p[0], &p[1], N / 2, 2.0).over_1d(N),
                    chain_k.bias.invoke(&p[3], &p[2], N, 1.0).over_1d(N),
                ];
                let ctx = device.create_context()?;
                let mut scheme = Scheme::new(&ctx);
                assert!(fused.record(&mut scheme, "mismatch", &unshared).is_err());
                assert_eq!(scheme.ir_node_count(), 0);
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("auto_fusion_promotes_and_replays_gpu", {
            let device = Arc::clone(&device);
            move || {
                let _pins = auto_fusion_pins();
                let k = Chain {
                    scale: scale::Kernel::prepare(&device)?,
                    bias: bias::Kernel::prepare(&device)?,
                };
                let p = buffers(&device, &[input_data(), sentinel(), sentinel()])?;
                let want = chain_reference(1.0, false);
                let ctx = device.create_context()?;
                let mut scheme = fusing(&ctx);
                record_chain(&mut scheme, &k, &p)?;

                // Frame 1 settles the structure, frame 2 compiles, frame 3 promotes.
                for f in 1..=3 {
                    assert_eq!(scheme.executed_node_count(), 2, "frame {f} runs as recorded");
                    submit_and_check(&mut scheme, &p, &want, &format!("frame {f}"))?;
                }
                assert_eq!(scheme.executed_node_count(), 1);
                let stats = scheme.replay_stats();
                assert_eq!(stats.fusion_promotions, 1);
                assert_eq!(stats.records, 2, "recorded once, re-recorded once to promote");
                let report = scheme.fusion_report();
                assert_eq!(report.regions.len(), 1);
                assert_eq!(report.regions[0].status, FusionRegionStatus::Promoted);
                assert_eq!(report.regions[0].forwarded, 1);

                for f in 4..=8 {
                    submit_and_check(&mut scheme, &p, &want, &format!("fused frame {f}"))?;
                }
                let stats = scheme.replay_stats();
                assert_eq!(stats.records, 2, "the promoted plan replays");
                assert_eq!(stats.clean_submits, 6, "frame 2 and frames 4..=8");
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("auto_fusion_compile_failure_runs_unfused_gpu", {
            let device = Arc::clone(&device);
            move || {
                let _pins = auto_fusion_pins();
                let _fault = goldy::test_support::FusionCompileFault::install();
                let k = Chain {
                    scale: scale::Kernel::prepare(&device)?,
                    bias: bias::Kernel::prepare(&device)?,
                };
                let p = buffers(&device, &[input_data(), sentinel(), sentinel()])?;
                let want = chain_reference(1.0, false);
                let ctx = device.create_context()?;
                let mut scheme = fusing(&ctx);
                record_chain(&mut scheme, &k, &p)?;
                for f in 1..=5 {
                    submit_and_check(&mut scheme, &p, &want, &format!("frame {f}"))?;
                }
                assert_eq!(scheme.executed_node_count(), 2);
                let stats = scheme.replay_stats();
                assert_eq!(stats.fusion_compile_failures, 1);
                assert_eq!(stats.fusion_promotions, 0);
                assert_eq!(stats.records, 1);
                let report = scheme.fusion_report();
                assert!(
                    matches!(&report.regions[0].status, FusionRegionStatus::Failed(e) if e.contains("injected")),
                    "{:?}",
                    report.regions[0].status
                );
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("auto_fusion_follows_mutations_gpu", {
            let device = Arc::clone(&device);
            move || {
                let _pins = auto_fusion_pins();
                let k = Chain {
                    scale: scale::Kernel::prepare(&device)?,
                    bias: bias::Kernel::prepare(&device)?,
                };
                let shift_k = shift::Kernel::prepare(&device)?;
                let p = buffers(&device, &[input_data(), sentinel(), sentinel(), sentinel()])?;
                let ctx = device.create_context()?;
                let mut scheme = fusing(&ctx);
                let bias_node = record_chain(&mut scheme, &k, &p)?;
                let mut want = chain_reference(1.0, false);
                want.push(sentinel());
                for f in 1..=3 {
                    submit_and_check(&mut scheme, &p, &want, &format!("frame {f}"))?;
                }
                assert_eq!(scheme.executed_node_count(), 1);

                // A constituent's scalar reaches the fused dispatch without dropping the plan.
                scheme.set_node_param(bias_node, 1, 4.0f32.to_bits())?;
                let mut want = chain_reference(4.0, false);
                want.push(sentinel());
                submit_and_check(&mut scheme, &p, &want, "after set_node_param")?;
                assert_eq!(scheme.executed_node_count(), 1);

                // A neighbour read of the fused output stays unfused; recording drops the
                // plan, and the cached fused pipeline promotes again once it settles.
                shift_k
                    .invoke(&p[2], &p[3], N)
                    .over_1d(N)
                    .record(&mut scheme, "shift")?;
                assert_eq!(scheme.executed_node_count(), 3);
                let want = chain_reference(4.0, true);
                submit_and_check(&mut scheme, &p, &want, "settling after shift")?;
                submit_and_check(&mut scheme, &p, &want, "re-promoted")?;
                assert_eq!(scheme.executed_node_count(), 2);
                let stats = scheme.replay_stats();
                assert_eq!(stats.fusion_fallbacks, 1);
                assert_eq!(stats.fusion_promotions, 2);
                let report = scheme.fusion_report();
                assert!(
                    matches!(
                        &report.rejected[..],
                        [r] if matches!(&r.reason, FusionRejection::NonLocalDependence { stage: 2, .. })
                    ),
                    "{:?}",
                    report.rejected
                );
                for f in 1..=3 {
                    submit_and_check(&mut scheme, &p, &want, &format!("fused with shift, frame {f}"))?;
                }
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("temporaries_share_storage_gpu", {
            let device = Arc::clone(&device);
            move || {
                let _pins = auto_fusion_pins();
                let k = Chain {
                    scale: scale::Kernel::prepare(&device)?,
                    bias: bias::Kernel::prepare(&device)?,
                };
                let p = buffers(&device, &[input_data(), sentinel()])?;
                let want = [input_data(), temporary_chain_reference(4)];
                let ctx = device.create_context()?;
                let mut scheme = Scheme::new(&ctx);
                scheme.set_automatic_fusion(false);
                record_temporary_chain(&mut scheme, &k, &p)?;
                for f in 1..=4 {
                    submit_and_check(&mut scheme, &p, &want, &format!("frame {f}"))?;
                }
                assert_eq!(scheme.temporary_storage_count(), 2, "t0 and t2 share one buffer");
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("temporary_elision_keeps_every_parcel_gpu", {
            let device = Arc::clone(&device);
            move || {
                let _pins = auto_fusion_pins();
                let k = Chain {
                    scale: scale::Kernel::prepare(&device)?,
                    bias: bias::Kernel::prepare(&device)?,
                };
                let p = buffers(&device, &[input_data(), sentinel()])?;
                let want = [input_data(), temporary_chain_reference(4)];
                let ctx = device.create_context()?;
                let mut scheme = fusing(&ctx);
                record_temporary_chain(&mut scheme, &k, &p)?;

                submit_and_check(&mut scheme, &p, &want, "frame 1")?;
                assert_eq!(
                    scheme.temporary_storage_count(),
                    2,
                    "unfused, the temporaries need storage"
                );
                for f in 2..=3 {
                    submit_and_check(&mut scheme, &p, &want, &format!("frame {f}"))?;
                }
                assert_eq!(scheme.executed_node_count(), 1);
                let report = scheme.fusion_report();
                assert_eq!(report.regions[0].status, FusionRegionStatus::Promoted);
                assert_eq!((report.regions[0].forwarded, report.regions[0].elided), (3, 3));
                assert_eq!(scheme.temporary_storage_count(), 0, "every temporary is elided");
                for f in 4..=6 {
                    submit_and_check(&mut scheme, &p, &want, &format!("elided frame {f}"))?;
                }
                assert_eq!(scheme.replay_stats().records, 2, "the elided plan replays");
                Ok(())
            }
        }),
        libtest_mimic::Trial::test("temporary_read_outside_the_region_is_stored_gpu", {
            let device = Arc::clone(&device);
            move || {
                let _pins = auto_fusion_pins();
                let k = Chain {
                    scale: scale::Kernel::prepare(&device)?,
                    bias: bias::Kernel::prepare(&device)?,
                };
                let shift_k = shift::Kernel::prepare(&device)?;
                let p = buffers(&device, &[input_data(), sentinel(), sentinel()])?;
                let doubled = temporary_chain_reference(1);
                let shifted: Vec<f32> = (0..LEN)
                    .map(|i| if i + 1 < N as usize { doubled[i + 1] } else { SENTINEL })
                    .collect();
                let want = [input_data(), temporary_chain_reference(2), shifted];
                let ctx = device.create_context()?;
                let mut scheme = fusing(&ctx);
                let t = scheme.temporary_buffer::<f32>(LEN)?;
                k.scale
                    .invoke(&p[0], &t, N, 2.0)
                    .over_1d(N)
                    .record(&mut scheme, "scale")?;
                k.bias
                    .invoke(&t, &p[1], N, 1.0)
                    .over_1d(N)
                    .record(&mut scheme, "bias")?;
                shift_k.invoke(&t, &p[2], N).over_1d(N).record(&mut scheme, "shift")?;
                for f in 1..=5 {
                    submit_and_check(&mut scheme, &p, &want, &format!("frame {f}"))?;
                }
                assert_eq!(scheme.executed_node_count(), 2);
                let region = &scheme.fusion_report().regions[0];
                assert_eq!((region.forwarded, region.elided), (1, 0));
                assert_eq!(scheme.temporary_storage_count(), 1);
                Ok(())
            }
        }),
    ];

    let conclusion = libtest_mimic::run(&args, tests);
    drop(device);
    drop(instance);
    conclusion.exit_if_failed();
}
