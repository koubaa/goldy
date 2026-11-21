//! `tanh` / `sigmoid` against an f64 reference.
//!
//! Included by the default-backend test and the CPU-backend test.

use goldy::{
    RequestAdapterOptions, Runtime, RuntimeDescriptor, Scheme, Tensor, TensorDType, TensorKernels, TensorShape,
};

#[goldy::compute(workgroup_size = [64, 1, 1])]
fn activation_kernel(
    src: goldy::gpu::Tensor<f32>,
    tanh_out: goldy::gpu::TensorWrite<f32>,
    sigmoid_out: goldy::gpu::TensorWrite<f32>,
) {
    let i = goldy::gpu::global_id().x;
    if i < src.len() {
        let x = src[i];
        tanh_out[i] = goldy::gpu::tanh(x);
        sigmoid_out[i] = goldy::gpu::sigmoid(x);
    }
}

fn reference_inputs() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        0.25,
        -0.25,
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.0,
        -2.0,
        4.0,
        -4.0,
        8.0,
        -8.0,
        9.0,
        -9.0,
        10.0,
        -10.0,
        16.0,
        -16.0,
        25.0,
        -25.0,
        30.0,
        -30.0,
        40.0,
        -40.0,
        80.0,
        -80.0,
        88.0,
        -88.0,
        100.0,
        -100.0,
        1.0e10,
        -1.0e10,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ]
}

fn sigmoid_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn close(got: f32, expected: f64) -> bool {
    if expected.is_nan() {
        return got.is_nan();
    }
    let diff = (f64::from(got) - expected).abs();
    diff <= 1e-5 + 1e-4 * expected.abs()
}

fn assert_close(what: &str, got: &[f32], inputs: &[f32], expected: impl Fn(f64) -> f64) {
    assert_eq!(got.len(), inputs.len(), "{what}: length");
    for (i, (&g, &x)) in got.iter().zip(inputs).enumerate() {
        let e = expected(f64::from(x));
        assert!(
            close(g, e),
            "{what}[{i}] input={x:?} got={g:?} expected={e:?} abs_err={}",
            (f64::from(g) - e).abs()
        );
    }
}

fn read_f32(scheme: &mut Scheme, buf: &goldy::Buffer) -> Vec<f32> {
    let mut sub = scheme.submit().expect("submit");
    let bytes = (&mut sub >> buf).take::<u8>().expect("host take");
    bytemuck::cast_slice(&bytes).to_vec()
}

pub fn assert_activations(device: &Runtime) {
    let inputs = reference_inputs();
    let n = inputs.len() as u32;
    let shape = TensorShape::vector(n);
    let ctx = device.create_context().expect("context");
    let src = Tensor::from_f32(device, shape, &inputs).unwrap();

    let kernel = activation_kernel::Kernel::prepare(device).expect("prepare dialect kernel");
    let tanh_out = Tensor::zeros(device, shape, TensorDType::F32).unwrap();
    let sigmoid_out = Tensor::zeros(device, shape, TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    kernel
        .record(
            &mut scheme,
            "activations",
            src.view(),
            tanh_out.view(),
            sigmoid_out.view(),
        )
        .unwrap()
        .over_1d(n);
    let where_ran = format!("{:?} / {}", device.backend_type(), device.adapter().name());
    assert_close(
        &format!("gpu::tanh on {where_ran}"),
        &read_f32(&mut scheme, tanh_out.buffer()),
        &inputs,
        f64::tanh,
    );
    assert_close(
        &format!("gpu::sigmoid on {where_ran}"),
        &read_f32(&mut scheme, sigmoid_out.buffer()),
        &inputs,
        sigmoid_f64,
    );

    let kernels = TensorKernels::new(device).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let mut rec = kernels.recorder(&mut scheme);
    let t = rec.tanh("tanh", src.view()).unwrap();
    let s = rec.sigmoid("sigmoid", src.view()).unwrap();
    drop(rec);
    assert_close(
        &format!("tensor tanh on {where_ran}"),
        &read_f32(&mut scheme, t.buffer()),
        &inputs,
        f64::tanh,
    );
    assert_close(
        &format!("tensor sigmoid on {where_ran}"),
        &read_f32(&mut scheme, s.buffer()),
        &inputs,
        sigmoid_f64,
    );

    let inplace = Tensor::from_f32(device, shape, &inputs).unwrap();
    let into = Tensor::zeros(device, shape, TensorDType::F32).unwrap();
    let mut scheme = Scheme::new(&ctx);
    let mut rec = kernels.recorder(&mut scheme);
    rec.tanh_into("tanh_into", inplace.view(), inplace.view()).unwrap();
    rec.sigmoid_into("sigmoid_into", src.view(), into.view()).unwrap();
    drop(rec);
    assert_close(
        &format!("tanh_into on {where_ran}"),
        &read_f32(&mut scheme, inplace.buffer()),
        &inputs,
        f64::tanh,
    );
    assert_close(
        &format!("sigmoid_into on {where_ran}"),
        &read_f32(&mut scheme, into.buffer()),
        &inputs,
        sigmoid_f64,
    );
}

#[allow(dead_code)]
pub fn runtime() -> Runtime {
    goldy::Instance::new()
        .expect("instance")
        .request_adapter(&RequestAdapterOptions::default())
        .expect("adapter")
        .request_runtime(&RuntimeDescriptor::default())
        .expect("runtime")
}
