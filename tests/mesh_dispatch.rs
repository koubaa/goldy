#![allow(deprecated)]

#[path = "common/submission.rs"]
mod submission;

#[path = "common/scheme_render.rs"]
mod scheme_render;

#[cfg(all(feature = "gpu", feature = "graphics"))]
mod imp {
    use crate::scheme_render::{acquire_readback_texture, scheme_render_and_readback};
    use crate::submission::submission_context;
    use goldy::{
        types::BackendType, Color, Device, DeviceDescriptor, Instance, MeshPipeline, MeshPipelineDesc,
        RequestAdapterOptions, RetainedPool, ShaderModule, TargetLoad, TextureFormat,
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

    const MESH_SLANG: &str = r#"
import goldy_exp;

struct MeshOutput {
    float4 pos : SV_Position;
    float4 color : COLOR;
};

[goldy_mesh]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void mesh_main(out vertices MeshOutput verts[3], out indices uint3 tris[1]) {
    SetMeshOutputCounts(3, 1);
    verts[0].pos = float4(-1.0, -1.0, 0.0, 1.0);
    verts[0].color = float4(1.0, 0.0, 0.0, 1.0);
    verts[1].pos = float4(3.0, -1.0, 0.0, 1.0);
    verts[1].color = float4(1.0, 0.0, 0.0, 1.0);
    verts[2].pos = float4(-1.0, 3.0, 0.0, 1.0);
    verts[2].color = float4(1.0, 0.0, 0.0, 1.0);
    tris[0] = uint3(0, 1, 2);
}

struct FsIn {
    float4 pos : SV_Position;
    float4 color : COLOR;
};

[goldy_fragment]
float4 fs_main(FsIn input) : SV_Target {
    return input.color;
}
"#;

    #[test]
    fn mesh_fullscreen_triangle_is_red() {
        let device = make_device();
        if !device.capabilities().mesh_shaders {
            eprintln!("skip: DeviceCapabilities::mesh_shaders is false on this adapter");
            return;
        }
        match device.backend_type() {
            BackendType::Vulkan | BackendType::Dx12 => {}
            other => {
                eprintln!("skip: mesh pipelines are Vulkan/DX12 only (backend={other:?})");
                return;
            }
        }

        let ctx = submission_context(&device);
        let format = device.capabilities().preferred_render_target_format;
        let shader = ShaderModule::from_slang(&device, MESH_SLANG).expect("compile mesh shader");
        let pipeline = MeshPipeline::new(
            &device,
            &MeshPipelineDesc {
                mesh: &shader,
                fragment: &shader,
                amplification: None,
                target_format: format,
                depth_stencil: None,
            },
        )
        .expect("mesh pipeline");

        let mut pool = RetainedPool::new(Arc::new(device.clone()));
        let width = 16u32;
        let height = 16u32;
        let readback = acquire_readback_texture(&mut pool, width, height, format);
        let raw = scheme_render_and_readback(
            &ctx,
            width,
            height,
            format,
            None,
            &readback,
            "mesh",
            TargetLoad::Clear(Color::BLACK),
            |pass| {
                pass.set_mesh_pipeline(&pipeline);
                pass.dispatch_mesh(1, 1, 1);
            },
        );

        let center = ((height / 2) * width + (width / 2)) as usize;
        match format {
            TextureFormat::Rgba8Unorm => {
                let i = center * 4;
                assert!(raw[i] > 200, "expected red, got {:?}", &raw[i..i + 4]);
                assert!(raw[i + 1] < 40, "expected red, got {:?}", &raw[i..i + 4]);
            }
            TextureFormat::Rgba32Float => {
                let floats: &[f32] = bytemuck::cast_slice(&raw);
                let i = center * 4;
                assert!(floats[i] > 0.9, "expected red, got {:?}", &floats[i..i + 4]);
                assert!(floats[i + 1] < 0.1, "expected red, got {:?}", &floats[i..i + 4]);
            }
            other => panic!("unexpected render-target format {other:?}"),
        }
    }
}
