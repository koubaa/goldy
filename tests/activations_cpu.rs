//! `tanh` and `sigmoid` on the compute-only CPU backend.
//!
//! Isolated crate so `GOLDY_BACKEND=cpu` cannot race other GPU tests.

#![cfg(all(feature = "gpu", feature = "tensor"))]

#[path = "common/activations.rs"]
mod activations;

#[test]
fn tanh_and_sigmoid_match_f64_on_cpu() {
    // SAFETY: this integration test is its own process.
    unsafe { std::env::set_var("GOLDY_BACKEND", "cpu") };
    let device = activations::runtime();
    assert_eq!(device.backend_type(), goldy::BackendType::Cpu);
    activations::assert_activations(&device);
}
