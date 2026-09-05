//! Yielding-script scenarios shared by the CPU and GPU backend test crates.

use goldy::{
    Backpressure, BufferKind, ComputePipeline, Context, Device, DeviceDescriptor, GoldyError, Instance, MemoryExchange,
    NodeAccess, Parcel, Petition, Promised, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule, YieldPoint,
};
use std::sync::{Arc, Mutex};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Fetch {
    pub key: u32,
}

impl Petition for Fetch {
    const SLANG_NAME: &'static str = "Fetch";
    type Result = u32;
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Step {
    pub lane: u32,
    pub remaining: u32,
}

impl Petition for Step {
    const SLANG_NAME: &'static str = "Step";
    type Result = u32;
}

pub fn make_device() -> Device {
    let instance = Instance::new().expect("instance");
    instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_device(&DeviceDescriptor::default())
        .expect("device")
}

fn read_u32(ctx: &Context, parcel: &Parcel) -> Vec<u32> {
    let mut scheme = Scheme::new(ctx);
    let w = MemoryExchange::new(ctx)
        .bind_withdraw(&mut scheme, parcel)
        .expect("withdraw");
    let mut sub = scheme.submit().expect("submit");
    let bytes = w.claim(&mut sub).expect("claim").consume().expect("consume");
    bytemuck::cast_slice(&bytes).to_vec()
}

/// Odd values petition the host for `key * 10`; even values are doubled in place.
const FETCH_SRC: &str = r#"
import goldy_exp;

[goldy_petition(Result = BufRO<uint>)]
struct Fetch { uint key; };

struct St { uint lane; uint acc; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, uint scale, ThreadId tid) {
    if (tid.x >= goldy_buf_len(data)) return;
    uint v = data[tid.x];
    if (v % 2u == 1u) {
        $yield(cs_resume, Fetch { v }, St { tid.x, v * scale });
        return;
    }
    data[tid.x] = v * 2u;
}

[goldy_resume]
[numthreads(32, 1, 1)]
void cs_resume(Scattered<uint> data, Resolved<uint> r, St s, ThreadId tid) {
    data[s.lane] = r.is_null() ? 0xFFFFFFFFu : r[0] + s.acc;
}
"#;

fn fetch_setup(device: &Device, n: u32) -> (Context, RetainedPool, goldy::Buffer, ComputePipeline) {
    let ctx = device.create_context().expect("ctx");
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let input: Vec<u32> = (0..n).collect();
    let data = pool
        .acquire_buffer_with_data(&input, BufferKind::Scattered)
        .expect("buffer");
    let shader = ShaderModule::from_slang(device, FETCH_SRC).expect("compile yielding script");
    let pipeline = ComputePipeline::new(device, &shader).expect("pipeline");
    assert!(pipeline.is_yielding());
    (ctx, pool, data, pipeline)
}

/// Basic yield → CPU handler → resume, with rejections, resubmitted twice.
pub fn fetch_and_resume(device: &Device) {
    let n = 200u32;
    let (ctx, _pool, data, pipeline) = fetch_setup(device, n);
    let calls = Arc::new(Mutex::new(Vec::<u32>::new()));
    let seen = Arc::clone(&calls);
    let mut scheme = Scheme::new(&ctx);
    let node = scheme
        .node("fetch", &pipeline)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .with_param(3)
        .yield_point(
            "cs_resume",
            YieldPoint::cpu(256, 1024, move |p: &Fetch, promised: Promised<'_, u32>| {
                seen.lock().unwrap().push(p.key);
                if p.key % 5 == 0 {
                    promised.reject();
                } else {
                    promised.fulfil(&[p.key * 10]);
                }
            }),
        )
        .dispatch(n.div_ceil(64), 1, 1);

    scheme.submit().expect("submit").wait_until_settled().expect("settle");
    let out = read_u32(&ctx, &data);
    for i in 0..n {
        let expect = if i % 2 == 1 {
            if i % 5 == 0 {
                u32::MAX
            } else {
                i * 10 + i * 3
            }
        } else {
            i * 2
        };
        assert_eq!(out[i as usize], expect, "lane {i}");
    }
    let stats = scheme.yield_stats(node).expect("stats");
    assert_eq!(stats.chunks, 1);
    assert_eq!(stats.rounds, 1);
    assert_eq!(stats.petitions, u64::from(n / 2));
    assert_eq!(stats.resumed, u64::from(n / 2));
    assert_eq!(
        stats.rejected,
        (1..n).filter(|i| i % 2 == 1 && i % 5 == 0).count() as u64
    );
    assert_eq!(stats.dropped, 0);
    let mut keys = calls.lock().unwrap().clone();
    keys.sort_unstable();
    assert_eq!(keys, (0..n).filter(|i| i % 2 == 1).collect::<Vec<_>>());

    // Second submission (retained scheme): 13*i and MAX are odd, so every odd lane
    // petitions again with its new value.
    calls.lock().unwrap().clear();
    scheme.submit().expect("resubmit").wait_until_settled().expect("settle");
    let stats = scheme.yield_stats(node).expect("stats");
    assert_eq!(stats.petitions, u64::from(n / 2));
    assert_eq!(stats.rounds, 1);
    let mut keys = calls.lock().unwrap().clone();
    keys.sort_unstable();
    let mut expect: Vec<u32> = out.iter().copied().filter(|v| v % 2 == 1).collect();
    expect.sort_unstable();
    assert_eq!(keys, expect);
}

/// `Backpressure::Stall` splits the prologue into chunks no larger than the capacity.
pub fn stall_chunks_the_prologue(device: &Device) {
    let n = 1024u32;
    let (ctx, _pool, data, pipeline) = fetch_setup(device, n);
    let mut scheme = Scheme::new(&ctx);
    let node = scheme
        .node("fetch", &pipeline)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .with_param(1)
        .yield_point(
            "cs_resume",
            YieldPoint::cpu(128, 128, |p: &Fetch, promised: Promised<'_, u32>| {
                promised.fulfil(&[p.key]);
            }),
        )
        .dispatch(n.div_ceil(64), 1, 1);
    scheme.submit().expect("submit").wait_until_settled().expect("settle");
    let out = read_u32(&ctx, &data);
    for i in 0..n {
        assert_eq!(out[i as usize], if i % 2 == 1 { 2 * i } else { 2 * i }, "lane {i}");
    }
    let stats = scheme.yield_stats(node).expect("stats");
    assert_eq!(stats.chunks, 8);
    assert_eq!(stats.rounds, 8);
    assert_eq!(stats.petitions, u64::from(n / 2));
    assert_eq!(stats.dropped, 0);
}

/// `Backpressure::Drop` launches once and loses lanes beyond the capacity.
pub fn drop_loses_excess_lanes(device: &Device) {
    let n = 512u32;
    let (ctx, _pool, data, pipeline) = fetch_setup(device, n);
    let mut scheme = Scheme::new(&ctx);
    let node = scheme
        .node("fetch", &pipeline)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .with_param(0)
        .yield_point(
            "cs_resume",
            YieldPoint::cpu(100, 100, |p: &Fetch, promised: Promised<'_, u32>| {
                promised.fulfil(&[p.key + 1000]);
            })
            .backpressure(Backpressure::Drop),
        )
        .dispatch(n.div_ceil(64), 1, 1);
    scheme.submit().expect("submit").wait_until_settled().expect("settle");
    let out = read_u32(&ctx, &data);
    let mut serviced = 0;
    let mut dropped = 0;
    for i in 0..n {
        let v = out[i as usize];
        if i % 2 == 0 {
            assert_eq!(v, 2 * i, "lane {i}");
        } else if v == i + 1000 {
            serviced += 1;
        } else {
            assert_eq!(v, i, "dropped lane {i} must be untouched");
            dropped += 1;
        }
    }
    assert_eq!(serviced, 100);
    assert_eq!(dropped, n / 2 - 100);
    let stats = scheme.yield_stats(node).expect("stats");
    assert_eq!(stats.chunks, 1);
    assert_eq!(stats.petitions, 100);
    assert_eq!(stats.dropped, u64::from(n / 2 - 100));
}

/// A continuation that yields to itself: each lane walks `remaining` steps, summing the
/// handler's answers, then writes the total.
const COUNTDOWN_SRC: &str = r#"
import goldy_exp;

[goldy_petition(Result = BufRO<uint>)]
struct Step { uint lane; uint remaining; };

struct Walk { uint lane; uint acc; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> steps, Scattered<uint> out, ThreadId tid) {
    if (tid.x >= goldy_buf_len(steps)) return;
    $yield(step, Step { tid.x, steps[tid.x] }, Walk { tid.x, 0u });
}

[goldy_resume(Step)]
[numthreads(64, 1, 1)]
void step(Scattered<uint> out, Resolved<uint> r, Walk w) {
    uint acc = w.acc + (r.is_null() ? 0u : r[0]);
    // The handler answers with the remaining count; keep walking until it says zero.
    if (!r.is_null() && r[0] > 0u) {
        $yield(step, Step { w.lane, r[0] - 1u }, Walk { w.lane, acc });
        return;
    }
    out[w.lane] = acc;
}
"#;

pub fn continuation_yields_to_itself(device: &Device) {
    let n = 96u32;
    let ctx = device.create_context().expect("ctx");
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let steps_in: Vec<u32> = (0..n).map(|i| i % 5).collect();
    let steps = pool
        .acquire_buffer_with_data(&steps_in, BufferKind::Scattered)
        .expect("buffer");
    let out = pool
        .acquire_buffer_with_data(&vec![u32::MAX; n as usize], BufferKind::Scattered)
        .expect("buffer");
    let shader = ShaderModule::from_slang(device, COUNTDOWN_SRC).expect("compile");
    let pipeline = ComputePipeline::new(device, &shader).expect("pipeline");

    let mut scheme = Scheme::new(&ctx);
    let node = scheme
        .node("walk", &pipeline)
        .with_parcel(&steps, NodeAccess::Read)
        .with_parcel(&out, NodeAccess::Write)
        .yield_point(
            "step",
            YieldPoint::cpu(128, 128, |p: &Step, promised: Promised<'_, u32>| {
                promised.fulfil(&[p.remaining]);
            }),
        )
        .dispatch(n.div_ceil(64), 1, 1);
    scheme.submit().expect("submit").wait_until_settled().expect("settle");
    let got = read_u32(&ctx, &out);
    for i in 0..n {
        // Answers are remaining, remaining-1, .., 0.
        let k = i % 5;
        assert_eq!(got[i as usize], k * (k + 1) / 2, "lane {i}");
    }
    let stats = scheme.yield_stats(node).expect("stats");
    assert_eq!(stats.rounds, 5);
    let expected_petitions: u64 = (0..n).map(|i| u64::from(i % 5) + 1).sum();
    assert_eq!(stats.petitions, expected_petitions);
    assert_eq!(stats.resumed, expected_petitions);
}

/// Two continuations, one petition type each, the second reached only through the first.
const CHAIN_SRC: &str = r#"
import goldy_exp;

[goldy_petition(Result = BufRO<uint>)]
struct Fetch { uint key; };

[goldy_petition(Result = BufRO<float>)]
struct Scale { uint key; uint factor; };

struct A { uint lane; };
struct B { uint lane; uint fetched; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> out, Scattered<float> fout, ThreadId tid) {
    if (tid.x >= goldy_buf_len(out)) return;
    $yield(got_fetch, Fetch { tid.x }, A { tid.x });
}

[goldy_resume]
void got_fetch(Scattered<uint> out, Resolved<uint> r, A a) {
    if (r.is_null()) { out[a.lane] = 0u; return; }
    uint sum = 0u;
    for (uint i = 0u; i < r.len(); ++i) sum += r[i];
    out[a.lane] = sum;
    if (sum % 2u == 0u) {
        $yield(got_scale, Scale { a.lane, 2u }, B { a.lane, sum });
    }
}

[goldy_resume]
void got_scale(Scattered<float> fout, Resolved<float> r, B b) {
    fout[b.lane] = r.is_null() ? -1.0 : r[0] * float(b.fetched);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Scale {
    key: u32,
    factor: u32,
}

impl Petition for Scale {
    const SLANG_NAME: &'static str = "Scale";
    type Result = f32;
}

pub fn chained_continuations_with_multi_element_results(device: &Device) {
    let n = 130u32;
    let ctx = device.create_context().expect("ctx");
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let out = pool
        .acquire_buffer_with_data(&vec![u32::MAX; n as usize], BufferKind::Scattered)
        .expect("buffer");
    let fout = pool
        .acquire_buffer_with_data(&vec![0.0f32; n as usize], BufferKind::Scattered)
        .expect("buffer");
    let shader = ShaderModule::from_slang(device, CHAIN_SRC).expect("compile");
    let pipeline = ComputePipeline::new(device, &shader).expect("pipeline");

    let mut scheme = Scheme::new(&ctx);
    let node = scheme
        .node("chain", &pipeline)
        .with_parcel(&out, NodeAccess::Write)
        .with_parcel(&fout, NodeAccess::Write)
        .yield_point(
            "got_fetch",
            YieldPoint::cpu(256, 16384, |p: &Fetch, promised: Promised<'_, u32>| {
                // key elements: 1..=key (sum = key(key+1)/2); key 0 → empty (non-null) view.
                let items: Vec<u32> = (1..=p.key).collect();
                promised.fulfil(&items);
            }),
        )
        .yield_point(
            "got_scale",
            YieldPoint::cpu(256, 256, |p: &Scale, promised: Promised<'_, f32>| {
                promised.fulfil(&[p.factor as f32 * 0.5]);
            }),
        )
        .dispatch(n.div_ceil(64), 1, 1);
    scheme.submit().expect("submit").wait_until_settled().expect("settle");
    let got = read_u32(&ctx, &out);
    let gotf: Vec<f32> = bytemuck::cast_slice(&read_u32(&ctx, &fout)).to_vec();
    let mut scaled = 0u64;
    for i in 0..n {
        let sum = i * (i + 1) / 2;
        assert_eq!(got[i as usize], sum, "lane {i}");
        if sum % 2 == 0 {
            assert_eq!(gotf[i as usize], sum as f32, "lane {i} scaled");
            scaled += 1;
        } else {
            assert_eq!(gotf[i as usize], 0.0, "lane {i} untouched");
        }
    }
    let stats = scheme.yield_stats(node).expect("stats");
    assert_eq!(stats.rounds, 2);
    assert_eq!(stats.petitions, u64::from(n) + scaled);
    assert_eq!(
        stats.arena_overflow, 0,
        "sum_{{k<130}} k = 8385 elements fit the 16384-element arena"
    );
}

/// Fulfilments past the arena capacity become rejections the continuation can see.
pub fn arena_overflow_rejects(device: &Device) {
    let n = 200u32;
    let (ctx, _pool, data, pipeline) = fetch_setup(device, n);
    let mut scheme = Scheme::new(&ctx);
    let node = scheme
        .node("fetch", &pipeline)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .with_param(0)
        .yield_point(
            "cs_resume",
            YieldPoint::cpu(256, 10, |p: &Fetch, promised: Promised<'_, u32>| {
                promised.fulfil(&[p.key]);
            }),
        )
        .dispatch(n.div_ceil(64), 1, 1);
    scheme.submit().expect("submit").wait_until_settled().expect("settle");
    let out = read_u32(&ctx, &data);
    let fulfilled = (0..n).filter(|&i| i % 2 == 1 && out[i as usize] == i).count();
    let nulled = (0..n).filter(|&i| i % 2 == 1 && out[i as usize] == u32::MAX).count();
    assert_eq!(fulfilled, 10);
    assert_eq!(nulled, (n / 2) as usize - 10);
    let stats = scheme.yield_stats(node).expect("stats");
    assert_eq!(stats.arena_overflow, u64::from(n / 2) - 10);
    assert_eq!(stats.rejected, u64::from(n / 2) - 10);
}

/// GPU handler: resolves every petition on the device with `goldy_resolve`.
const GPU_HANDLER_SRC: &str = r#"
import goldy_exp;

struct Fetch { uint key; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(BufRO<Fetch> petitions, Scattered<Resolution> resolutions, Scattered<uint> arena,
             BufRO<uint> table, uint count, ThreadId tid) {
    if (tid.x >= count) return;
    uint key = petitions[tid.x].key;
    if (key % 7u == 0u) {
        goldy_reject(resolutions, tid.x);
        return;
    }
    arena[tid.x] = table[key % goldy_buf_len(table)];
    goldy_resolve(resolutions, tid.x, tid.x, 1u);
}
"#;

pub fn node_handler_resolves_on_gpu(device: &Device) {
    let n = 256u32;
    let (ctx, mut pool, data, pipeline) = fetch_setup(device, n);
    let table_in: Vec<u32> = (0..16u32).map(|i| 1000 + i).collect();
    let table = pool
        .acquire_buffer_with_data(&table_in, BufferKind::Scattered)
        .expect("buffer");
    let handler_shader = ShaderModule::from_slang(device, GPU_HANDLER_SRC).expect("compile handler");
    let handler = ComputePipeline::new(device, &handler_shader).expect("handler pipeline");

    let mut scheme = Scheme::new(&ctx);
    let node = scheme
        .node("fetch", &pipeline)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .with_param(1)
        .yield_point(
            "cs_resume",
            YieldPoint::node(256, 256, &handler).with_parcel(&table, NodeAccess::Read),
        )
        .dispatch(n.div_ceil(64), 1, 1);
    scheme.submit().expect("submit").wait_until_settled().expect("settle");
    let out = read_u32(&ctx, &data);
    for i in 0..n {
        let expect = if i % 2 == 1 {
            if i % 7 == 0 {
                u32::MAX
            } else {
                table_in[(i % 16) as usize] + i
            }
        } else {
            i * 2
        };
        assert_eq!(out[i as usize], expect, "lane {i}");
    }
    let stats = scheme.yield_stats(node).expect("stats");
    assert_eq!(stats.petitions, u64::from(n / 2));
    assert_eq!(stats.rejected, 0, "GPU handlers are not counted host-side");
}

/// Struct result elements (`Resolved<Pair>`) through both a CPU and a GPU handler.
const PAIR_SRC: &str = r#"
import goldy_exp;

struct Pair { uint a; uint b; };

[goldy_petition(Result = BufRO<Pair>)]
struct Ask { uint key; };

struct S { uint lane; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> out, ThreadId tid) {
    if (tid.x >= goldy_buf_len(out)) return;
    $yield(got, Ask { tid.x }, S { tid.x });
}

[goldy_resume]
void got(Scattered<uint> out, Resolved<Pair> r, S s) {
    out[s.lane] = (r.is_null() || r.len() != 2u) ? 0u : r[0].a * 1000u + r[1].b;
}
"#;

const PAIR_HANDLER_SRC: &str = r#"
import goldy_exp;

struct Pair { uint a; uint b; };
struct Ask { uint key; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(BufRO<Ask> petitions, Scattered<Resolution> resolutions, Scattered<Pair> arena,
             uint count, ThreadId tid) {
    if (tid.x >= count) return;
    uint key = petitions[tid.x].key;
    arena[2u * tid.x] = { key, 0u };
    arena[2u * tid.x + 1u] = { 0u, key + 1u };
    goldy_resolve(resolutions, tid.x, 2u * tid.x, 2u);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Ask {
    key: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Pair {
    a: u32,
    b: u32,
}

impl Petition for Ask {
    const SLANG_NAME: &'static str = "Ask";
    type Result = Pair;
}

pub fn struct_result_elements(device: &Device) {
    let n = 100u32;
    let ctx = device.create_context().expect("ctx");
    let mut pool = RetainedPool::new(Arc::new(device.clone()));
    let shader = ShaderModule::from_slang(device, PAIR_SRC).expect("compile");
    let pipeline = ComputePipeline::new(device, &shader).expect("pipeline");
    let handler_shader = ShaderModule::from_slang(device, PAIR_HANDLER_SRC).expect("compile handler");
    let handler = ComputePipeline::new(device, &handler_shader).expect("handler pipeline");

    for gpu in [false, true] {
        let out = pool
            .acquire_buffer_with_data(&vec![u32::MAX; n as usize], BufferKind::Scattered)
            .expect("buffer");
        let point = if gpu {
            YieldPoint::node(128, 256, &handler)
        } else {
            YieldPoint::cpu(128, 256, |p: &Ask, promised: Promised<'_, Pair>| {
                promised.fulfil(&[Pair { a: p.key, b: 0 }, Pair { a: 0, b: p.key + 1 }]);
            })
        };
        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("pairs", &pipeline)
            .with_parcel(&out, NodeAccess::Write)
            .yield_point("got", point)
            .dispatch(n.div_ceil(64), 1, 1);
        scheme.submit().expect("submit").wait_until_settled().expect("settle");
        let got = read_u32(&ctx, &out);
        for i in 0..n {
            assert_eq!(got[i as usize], i * 1000 + i + 1, "lane {i} (gpu handler: {gpu})");
        }
    }
}

/// Record-time mistakes are reported on submit.
pub fn validation_errors(device: &Device) {
    let n = 64u32;
    let (ctx, _pool, data, pipeline) = fetch_setup(device, n);

    // Missing yield point.
    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("fetch", &pipeline)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .with_param(1)
        .dispatch(1, 1, 1);
    let err = scheme.submit().err().expect("missing yield point must fail");
    assert!(matches!(err, GoldyError::Validation(_)), "{err}");
    assert!(err.to_string().contains("cs_resume"), "{err}");

    // Wrong petition type.
    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("fetch", &pipeline)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .with_param(1)
        .yield_point(
            "cs_resume",
            YieldPoint::cpu(64, 64, |_: &Step, p: Promised<'_, u32>| p.reject()),
        )
        .dispatch(1, 1, 1);
    let err = scheme.submit().err().expect("petition mismatch must fail");
    assert!(err.to_string().contains("SLANG_NAME"), "{err}");

    // Stall with capacity below the workgroup size.
    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("fetch", &pipeline)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .with_param(1)
        .yield_point(
            "cs_resume",
            YieldPoint::cpu(16, 64, |_: &Fetch, p: Promised<'_, u32>| p.reject()),
        )
        .dispatch(1, 1, 1);
    let err = scheme.submit().err().expect("small stall capacity must fail");
    assert!(err.to_string().contains("numthreads.x"), "{err}");

    // Scalar count mismatch.
    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("fetch", &pipeline)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .yield_point(
            "cs_resume",
            YieldPoint::cpu(64, 64, |_: &Fetch, p: Promised<'_, u32>| p.reject()),
        )
        .dispatch(1, 1, 1);
    let err = scheme.submit().err().expect("scalar mismatch must fail");
    assert!(err.to_string().contains("scalar"), "{err}");

    // Yield point on a plain pipeline.
    let plain = ShaderModule::from_slang(
        device,
        "import goldy_exp;\n[goldy_compute] void cs_main(Scattered<uint> d, ThreadId t) { d[0] = 1u; }",
    )
    .expect("plain");
    let plain = ComputePipeline::new(device, &plain).expect("plain pipeline");
    assert!(!plain.is_yielding());
    let mut scheme = Scheme::new(&ctx);
    scheme
        .node("plain", &plain)
        .with_parcel(&data, NodeAccess::Write)
        .yield_point(
            "nope",
            YieldPoint::cpu(64, 64, |_: &Fetch, p: Promised<'_, u32>| p.reject()),
        )
        .dispatch(1, 1, 1);
    let err = scheme.submit().err().expect("yield point on plain pipeline must fail");
    assert!(err.to_string().contains("not a yielding script"), "{err}");
}

/// A yielding node ordered between ordinary GPU nodes in the same scheme.
pub fn ordered_with_neighbouring_nodes(device: &Device) {
    let n = 128u32;
    let (ctx, _pool, data, pipeline) = fetch_setup(device, n);
    let plus_one = ShaderModule::from_slang(
        device,
        r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> d, ThreadId t) { if (t.x < goldy_buf_len(d)) d[t.x] = d[t.x] + 1u; }
"#,
    )
    .expect("plus_one");
    let plus_one = ComputePipeline::new(device, &plus_one).expect("pipeline");

    let mut scheme = Scheme::new(&ctx);
    // +1 flips parity: lanes with even i become odd and petition.
    scheme
        .node("pre", &plus_one)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .dispatch(n.div_ceil(64), 1, 1);
    scheme
        .node("fetch", &pipeline)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .with_param(0)
        .yield_point(
            "cs_resume",
            YieldPoint::cpu(256, 256, |p: &Fetch, promised: Promised<'_, u32>| {
                promised.fulfil(&[p.key * 100]);
            }),
        )
        .dispatch(n.div_ceil(64), 1, 1);
    scheme
        .node("post", &plus_one)
        .with_parcel(&data, NodeAccess::ReadWrite)
        .dispatch(n.div_ceil(64), 1, 1);
    let w = MemoryExchange::new(&ctx)
        .bind_withdraw(&mut scheme, &data)
        .expect("withdraw");
    let mut sub = scheme.submit().expect("submit");
    let bytes = w.claim(&mut sub).expect("claim").consume().expect("consume");
    let out: Vec<u32> = bytemuck::cast_slice(&bytes).to_vec();
    for i in 0..n {
        let v = i + 1;
        let expect = if v % 2 == 1 { v * 100 + 1 } else { v * 2 + 1 };
        assert_eq!(out[i as usize], expect, "lane {i}");
    }
}
