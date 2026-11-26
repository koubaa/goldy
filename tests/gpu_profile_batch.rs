//! `GOLDY_GPU_PROFILE` with batched dispatches.
//!
//! Independent same-pipeline nodes in one wave lower to a single `DispatchBatch`.
//! The query pool reserves one pair per dispatch in that batch. Recording must
//! advance the timestamp index once per dispatch, or a later command writes past
//! the pool.
//!
//! Its own test binary: the profiler reads `GOLDY_GPU_PROFILE` once per process.

#[cfg(any(feature = "vulkan", feature = "dx12"))]
#[test]
fn batched_dispatches_profile_without_overrunning_queries() {
    use goldy::{BufferKind, ComputePipeline, Instance, NodeAccess, Parcel, RequestAdapterOptions};
    use goldy::{RuntimeDescriptor, Scheme, ShaderModule, Submission};

    std::env::set_var("GOLDY_GPU_PROFILE", "1");

    const OVERWRITE_SHADER: &str = r#"
import goldy_exp;
[goldy_compute][numthreads(1,1,1)]
void cs_main(Scattered<uint> buf, ThreadId id) {
    buf[id.x] = 42u;
}
"#;
    const COPY_SHADER: &str = r#"
import goldy_exp;
[goldy_compute][numthreads(1,1,1)]
void cs_main(BufRO<uint> src, Scattered<uint> dst, ThreadId id) {
    dst[id.x] = src[id.x] + 1u;
}
"#;

    let device = Instance::new()
        .expect("instance")
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_runtime(&RuntimeDescriptor::default())
        .expect("runtime");
    let ctx = device.create_context().expect("context");
    let overwrite =
        ComputePipeline::new(&device, &ShaderModule::from_slang(&device, OVERWRITE_SHADER).unwrap()).unwrap();
    let copy = ComputePipeline::new(&device, &ShaderModule::from_slang(&device, COPY_SHADER).unwrap()).unwrap();
    let acquire = || {
        device
            .acquire_buffer_with_data(&[0u32; 1], BufferKind::Scattered)
            .expect("buffer")
    };
    let sources: Vec<_> = (0..4).map(|_| acquire()).collect();
    let sinks: Vec<_> = (0..4).map(|_| acquire()).collect();

    let mut scheme = Scheme::new(&ctx);
    for source in &sources {
        scheme
            .node("overwrite", &overwrite)
            .with_parcel(source, NodeAccess::Write)
            .dispatch(1, 1, 1);
    }
    for (source, sink) in sources.iter().zip(&sinks) {
        scheme
            .node("copy", &copy)
            .with_parcel(source, NodeAccess::Read)
            .with_parcel(sink, NodeAccess::Write)
            .dispatch(1, 1, 1);
    }

    let read = |submission: &mut Submission, parcel: &Parcel| (submission >> parcel).take::<u32>().unwrap()[0];
    for _ in 0..3 {
        let mut submission = scheme.submit().expect("submit");
        for sink in &sinks {
            assert_eq!(read(&mut submission, sink), 43);
        }
    }
}
