#![allow(deprecated)]

#[path = "common/submission.rs"]
mod submission;

#[cfg(feature = "gpu")]
mod imp {
    use crate::submission::submission_context;
    use goldy::{
        types::{BackendType, BufferFlags},
        AccelInstance, AccelerationStructure, BufferKind, ComputePipeline, Device, DeviceDescriptor, Instance,
        MemoryExchange, NodeAccess, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule,
    };
    use std::sync::{Arc, Mutex};

    static GPU: Mutex<()> = Mutex::new(());

    fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
        GPU.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn make_device() -> Device {
        Instance::new()
            .expect("instance")
            .request_adapter(&RequestAdapterOptions::default())
            .expect("adapter")
            .request_device(&DeviceDescriptor::default())
            .expect("device")
    }

    const TRACE_SLANG: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Accel scene, Scattered<uint> hits, ThreadId id)
{
    RayDesc ray;
    ray.Origin = float3(0.0, 0.0, -2.0);
    ray.TMin = 0.001;
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMax = 100.0;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    q.Proceed();
    uint hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 1 : 0;
    hits[id.x] = hit;
}
"#;

    #[test]
    fn triangle_blas_tlas_ray_query() {
        let _gpu = gpu_lock();
        let device = make_device();
        if !device.capabilities().ray_query {
            eprintln!("skip: DeviceCapabilities::ray_query is false on this adapter");
            return;
        }
        if device.backend_type() == BackendType::WebGpu {
            eprintln!(
                "skip: Slang WGSL has no inline ray tracing (TraceRayInline); wgpu AS create/build is wired, shader RQ is not"
            );
            return;
        }
        let ctx = submission_context(&device);

        let positions: [[f32; 3]; 3] = [[0.0, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]];
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let verts = pool
            .acquire_buffer_with_data_and_flags(&positions, BufferKind::Scattered, BufferFlags::ACCEL_INPUT)
            .expect("vertex buffer");
        let hits = pool
            .acquire_buffer_sized::<u32>(1, BufferKind::Scattered, BufferFlags::empty())
            .expect("hits");

        let blas = AccelerationStructure::blas_triangles(&device, 1, 3, 12).expect("BLAS");
        let tlas = AccelerationStructure::tlas(&device, 1).expect("TLAS");
        let shader = ShaderModule::from_slang(&device, TRACE_SLANG).expect("compile ray query shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .build_blas(&blas, verts.whole(), 3, 12, None)
            .expect("build_blas");
        let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        scheme
            .build_tlas(
                &tlas,
                &[AccelInstance {
                    blas: &blas,
                    transform: identity,
                    mask: 0xFF,
                    custom_index: 0,
                }],
            )
            .expect("build_tlas");
        drop(blas);
        scheme
            .node("trace", &pipeline)
            .with_parcel(&tlas, NodeAccess::Read)
            .with_parcel(&hits, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, hits.whole())
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let bytes = grant.claim(&mut frame).expect("claim").consume().expect("consume");
        let value: u32 = bytemuck::pod_read_unaligned(&bytes);
        assert_eq!(value, 1, "expected a closest-hit on the unit triangle");
    }

    #[test]
    fn indexed_blas_more_triangles_than_vertex_triples() {
        let _gpu = gpu_lock();
        let device = make_device();
        if !device.capabilities().ray_query {
            eprintln!("skip: DeviceCapabilities::ray_query is false on this adapter");
            return;
        }
        if device.backend_type() == BackendType::WebGpu {
            eprintln!("skip: WebGPU does not advertise ray_query");
            return;
        }
        let ctx = submission_context(&device);

        let positions: [[f32; 3]; 3] = [[0.0, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, -0.5, 0.0]];
        let indices: [u32; 3] = [0, 1, 2];
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let verts = pool
            .acquire_buffer_with_data_and_flags(&positions, BufferKind::Scattered, BufferFlags::ACCEL_INPUT)
            .expect("vertex buffer");
        let idx = pool
            .acquire_buffer_with_data_and_flags(&indices, BufferKind::Scattered, BufferFlags::ACCEL_INPUT)
            .expect("index buffer");
        let hits = pool
            .acquire_buffer_sized::<u32>(1, BufferKind::Scattered, BufferFlags::empty())
            .expect("hits");

        // Create-time max_triangles (2) is larger than max_vertices/3 so DX12
        // prebuild must size from IndexCount, not VertexCount alone.
        let blas = AccelerationStructure::blas_triangles(&device, 2, 3, 12).expect("BLAS");
        let tlas = AccelerationStructure::tlas(&device, 1).expect("TLAS");
        let shader = ShaderModule::from_slang(&device, TRACE_SLANG).expect("compile ray query shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .build_blas(&blas, verts.whole(), 3, 12, Some((idx.whole(), 3)))
            .expect("build_blas");
        let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        scheme
            .build_tlas(
                &tlas,
                &[AccelInstance {
                    blas: &blas,
                    transform: identity,
                    mask: 0xFF,
                    custom_index: 0,
                }],
            )
            .expect("build_tlas");
        drop(blas);
        scheme
            .node("trace", &pipeline)
            .with_parcel(&tlas, NodeAccess::Read)
            .with_parcel(&hits, NodeAccess::Write)
            .dispatch(1, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, hits.whole())
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let bytes = grant.claim(&mut frame).expect("claim").consume().expect("consume");
        let value: u32 = bytemuck::pod_read_unaligned(&bytes);
        assert_eq!(value, 1, "expected a closest-hit on the indexed mesh");
    }

    /// Indexed BLAS: `CommittedPrimitiveIndex` is a triangle index (use `prim * 3` into the IB),
    /// and barycentric interpolation of the fetched positions must match the ray hit.
    #[test]
    fn indexed_primitive_index_matches_hit_position() {
        let _gpu = gpu_lock();
        let device = make_device();
        if !device.capabilities().ray_query {
            eprintln!("skip: DeviceCapabilities::ray_query is false on this adapter");
            return;
        }
        if device.backend_type() == BackendType::WebGpu {
            eprintln!("skip: WebGPU does not advertise ray_query");
            return;
        }
        eprintln!(
            "indexed_primitive_index_matches_hit_position backend={:?}",
            device.backend_type()
        );
        let ctx = submission_context(&device);

        let positions: [[f32; 3]; 6] = [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, -1.0, 10.0],
            [1.0, -1.0, 10.0],
            [0.0, 1.0, 10.0],
        ];
        let indices: [u32; 6] = [0, 1, 2, 3, 4, 5];
        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let verts = pool
            .acquire_buffer_with_data_and_flags(&positions, BufferKind::Scattered, BufferFlags::ACCEL_INPUT)
            .expect("vertex buffer");
        let idx = pool
            .acquire_buffer_with_data_and_flags(&indices, BufferKind::Scattered, BufferFlags::ACCEL_INPUT)
            .expect("index buffer");
        let flat: [f32; 18] = bytemuck::cast(positions);
        let attr = pool
            .acquire_buffer_with_data(&flat, BufferKind::Scattered)
            .expect("attr verts");
        let out = pool
            .acquire_buffer_sized::<u32>(4, BufferKind::Scattered, BufferFlags::empty())
            .expect("out");

        let blas = AccelerationStructure::blas_triangles(&device, 2, 6, 12).expect("BLAS");
        let tlas = AccelerationStructure::tlas(&device, 1).expect("TLAS");
        const SLANG: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(2, 1, 1)]
void cs_main(Accel scene, BufRO<float> verts, BufRO<uint> indices, Scattered<uint> outp, ThreadId id)
{
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = 100.0;
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.Origin = id.x == 0 ? float3(0.0, 0.0, -2.0) : float3(0.0, 0.0, 8.0);

    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    q.Proceed();
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        outp[id.x * 2 + 0] = 0xFFFFFFFFu;
        outp[id.x * 2 + 1] = 0;
        return;
    }
    uint prim = q.CommittedPrimitiveIndex();
    uint i0 = indices[prim * 3 + 0];
    uint i1 = indices[prim * 3 + 1];
    uint i2 = indices[prim * 3 + 2];
    float2 bary = q.CommittedTriangleBarycentrics();
    float w0 = 1.0 - bary.x - bary.y;
    float w1 = bary.x;
    float w2 = bary.y;
    float3 p0 = float3(verts[i0 * 3 + 0], verts[i0 * 3 + 1], verts[i0 * 3 + 2]);
    float3 p1 = float3(verts[i1 * 3 + 0], verts[i1 * 3 + 1], verts[i1 * 3 + 2]);
    float3 p2 = float3(verts[i2 * 3 + 0], verts[i2 * 3 + 1], verts[i2 * 3 + 2]);
    float3 p = p0 * w0 + p1 * w1 + p2 * w2;
    float3 hitp = ray.Origin + ray.Direction * q.CommittedRayT();
    float err = length(p - hitp);
    outp[id.x * 2 + 0] = prim;
    outp[id.x * 2 + 1] = err < 0.05 ? 1 : 0;
}
"#;
        let shader = ShaderModule::from_slang(&device, SLANG).expect("compile");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

        let mut scheme = Scheme::new(&ctx);
        scheme
            .build_blas(&blas, verts.whole(), 6, 12, Some((idx.whole(), 6)))
            .expect("build_blas");
        let identity = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        scheme
            .build_tlas(
                &tlas,
                &[AccelInstance {
                    blas: &blas,
                    transform: identity,
                    mask: 0xFF,
                    custom_index: 0,
                }],
            )
            .expect("build_tlas");
        scheme
            .node("trace", &pipeline)
            .with_parcel(&tlas, NodeAccess::Read)
            .with_parcel(&attr, NodeAccess::Read)
            .with_parcel(&idx, NodeAccess::Read)
            .with_parcel(&out, NodeAccess::Write)
            .dispatch(2, 1, 1);

        let grant = MemoryExchange::new(scheme.context())
            .bind_withdraw(&mut scheme, out.whole())
            .expect("withdraw");
        let mut frame = scheme.submit().expect("submit");
        let bytes = grant.claim(&mut frame).expect("claim").consume().expect("consume");
        let vals: [u32; 4] = bytemuck::pod_read_unaligned(&bytes);
        assert_eq!(vals[0], 0, "ray 0 should hit triangle 0, got prim={}", vals[0]);
        assert_eq!(vals[1], 1, "ray 0 interpolated pos must match hit");
        assert_eq!(vals[2], 1, "ray 1 should hit triangle 1, got prim={}", vals[2]);
        assert_eq!(vals[3], 1, "ray 1 interpolated pos must match hit");
    }
}
