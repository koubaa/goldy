#![allow(deprecated)]

#[path = "common/submission.rs"]
mod submission;

#[cfg(feature = "gpu")]
mod imp {
    use crate::submission::submission_context;
    use goldy::{
        types::{BackendType, BufferFlags}, AccelInstance, AccelerationStructure, BufferKind, ComputePipeline, Device, DeviceDescriptor,
        Instance, MemoryExchange, NodeAccess, RequestAdapterOptions, RetainedPool, Scheme, ShaderModule,
    };
    use std::sync::Arc;

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
        scheme.build_blas(&blas, verts.whole(), 3, 12, None).expect("build_blas");
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
}
