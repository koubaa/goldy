//! Yielding scripts on the CPU backend (`GOLDY_BACKEND=cpu`).
//!
//! Isolated crate so the env override cannot race other GPU tests.

#[path = "common/yielding.rs"]
mod yielding;

fn cpu_device() -> (goldy::Device, std::sync::MutexGuard<'static, ()>) {
    static SELECT_CPU: std::sync::Once = std::sync::Once::new();
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // Host-callable JIT is not safe to compile from several tests at once on
    // every CI image (WebGPU Linux SIGILL'd under parallel slang CPU compiles).
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: this integration test is its own process, and the override is written once
    // before any thread reads it (every test goes through this `Once`).
    SELECT_CPU.call_once(|| unsafe { std::env::set_var("GOLDY_BACKEND", "cpu") });
    let device = yielding::make_device();
    assert_eq!(device.backend_type(), goldy::BackendType::Cpu);
    (device, guard)
}

#[test]
fn fetch_and_resume() {
    let (device, _lock) = cpu_device();
    yielding::fetch_and_resume(&device);
}

#[test]
fn stall_chunks_the_prologue() {
    let (device, _lock) = cpu_device();
    yielding::stall_chunks_the_prologue(&device);
}

#[test]
fn drop_loses_excess_lanes() {
    let (device, _lock) = cpu_device();
    yielding::drop_loses_excess_lanes(&device);
}

#[test]
fn continuation_yields_to_itself() {
    let (device, _lock) = cpu_device();
    yielding::continuation_yields_to_itself(&device);
}

#[test]
fn chained_continuations_with_multi_element_results() {
    let (device, _lock) = cpu_device();
    yielding::chained_continuations_with_multi_element_results(&device);
}

#[test]
fn arena_overflow_rejects() {
    let (device, _lock) = cpu_device();
    yielding::arena_overflow_rejects(&device);
}

#[test]
fn node_handler_resolves_on_gpu() {
    let (device, _lock) = cpu_device();
    yielding::node_handler_resolves_on_gpu(&device);
}

#[test]
fn struct_result_elements() {
    let (device, _lock) = cpu_device();
    yielding::struct_result_elements(&device);
}

#[test]
fn validation_errors() {
    let (device, _lock) = cpu_device();
    yielding::validation_errors(&device);
}

#[test]
fn ordered_with_neighbouring_nodes() {
    let (device, _lock) = cpu_device();
    yielding::ordered_with_neighbouring_nodes(&device);
}
