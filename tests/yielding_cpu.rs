//! Yielding scripts on the CPU backend (`GOLDY_BACKEND=cpu`).
//!
//! Isolated crate so the env override cannot race other GPU tests.

#[path = "common/yielding.rs"]
mod yielding;

fn cpu_device() -> goldy::Device {
    static SELECT_CPU: std::sync::Once = std::sync::Once::new();
    // SAFETY: this integration test is its own process, and the override is written once
    // before any thread reads it (every test goes through this `Once`).
    SELECT_CPU.call_once(|| unsafe { std::env::set_var("GOLDY_BACKEND", "cpu") });
    let device = yielding::make_device();
    assert_eq!(device.backend_type(), goldy::BackendType::Cpu);
    device
}

#[test]
fn fetch_and_resume() {
    yielding::fetch_and_resume(&cpu_device());
}

#[test]
fn stall_chunks_the_prologue() {
    yielding::stall_chunks_the_prologue(&cpu_device());
}

#[test]
fn drop_loses_excess_lanes() {
    yielding::drop_loses_excess_lanes(&cpu_device());
}

#[test]
fn continuation_yields_to_itself() {
    yielding::continuation_yields_to_itself(&cpu_device());
}

#[test]
fn chained_continuations_with_multi_element_results() {
    yielding::chained_continuations_with_multi_element_results(&cpu_device());
}

#[test]
fn arena_overflow_rejects() {
    yielding::arena_overflow_rejects(&cpu_device());
}

#[test]
fn node_handler_resolves_on_gpu() {
    yielding::node_handler_resolves_on_gpu(&cpu_device());
}

#[test]
fn validation_errors() {
    yielding::validation_errors(&cpu_device());
}

#[test]
fn ordered_with_neighbouring_nodes() {
    yielding::ordered_with_neighbouring_nodes(&cpu_device());
}
