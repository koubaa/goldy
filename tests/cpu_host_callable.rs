//! `#[goldy::compute]` kernels on the Slang host-callable CPU path (issue #292).

use goldy::cpu_shaders::{compile_kernel, CpuBinding};
use goldy::slang::{try_kernel_def_from_source, SlangCompiler};

#[goldy::compute(workgroup_size = [64, 1, 1])]
fn rust_double(data: &mut [u32]) {
    let i = goldy::gpu::global_id().x;
    if i < data.len() {
        data[i] = data[i] * 2u32;
    }
}

fn shader_search_path() -> String {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .to_string_lossy()
        .into_owned()
}

#[test]
fn goldy_compute_macro_runs_on_cpu() {
    let compiler = SlangCompiler::new().expect("Slang");
    let path = shader_search_path();
    let def = try_kernel_def_from_source(rust_double::CANONICAL_SOURCE).expect("kernel def");
    let kernel = compile_kernel(&compiler, &def, &[&path]).expect("compile rust_double");
    let mut data: Vec<u32> = (0..64).collect();
    kernel
        .dispatch_1d(64, &mut [CpuBinding::u32s(&mut data)])
        .expect("dispatch");
    for i in 0..64u32 {
        assert_eq!(data[i as usize], i * 2, "index {i}");
    }
}
