//! `tanh` and `sigmoid` on the default GPU backend (Vulkan lavapipe in CI).

#![cfg(all(feature = "gpu", feature = "tensor"))]

#[path = "common/activations.rs"]
mod activations;

#[test]
fn tanh_and_sigmoid_match_f64() {
    let device = activations::runtime();
    activations::assert_activations(&device);
}
