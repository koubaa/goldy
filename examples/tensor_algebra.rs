//! Headless dense tensor algebra: fill, add with broadcast, and GEMV.

use goldy::{
    Instance, MemoryExchange, RequestAdapterOptions, RuntimeDescriptor, Scheme, Tensor, TensorContext, TensorDType,
    TensorScalar, TensorShape,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runtime = Instance::new()?
        .request_adapter(&RequestAdapterOptions::default())?
        .request_runtime(&RuntimeDescriptor::default())?;
    let ctx = runtime.create_context()?;
    let mut tensors = TensorContext::new(&runtime)?;

    let x = Tensor::from_f32(&runtime, TensorShape::vector(4), &[1.0, 2.0, 3.0, 4.0])?;
    let w = Tensor::from_f32(
        &runtime,
        TensorShape::matrix(2, 4),
        &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )?;
    let mut scheme = Scheme::new(&ctx);
    let mut rec = tensors.recorder(&mut scheme);
    rec.fill("fill_bias", x.view(), TensorScalar::F32(1.0))?;
    let y = rec.matmul("gemv", w.view(), x.view())?;
    drop(rec);

    let grant = MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, y.buffer())?;
    let mut sub = scheme.submit()?;
    let bytes = grant.claim(&mut sub)?.consume()?;
    let out: &[f32] = bytemuck::cast_slice(&bytes);
    println!("gemv(W, ones) = {out:?}");
    assert_eq!(out, &[1.0, 1.0]);
    let _ = TensorDType::F32;
    Ok(())
}
