//! Yielding scripts on the default GPU backend.

#![cfg(feature = "gpu")]

#[path = "common/yielding.rs"]
mod yielding;

#[test]
fn fetch_and_resume() {
    yielding::fetch_and_resume(&yielding::make_device());
}

#[test]
fn stall_chunks_the_prologue() {
    yielding::stall_chunks_the_prologue(&yielding::make_device());
}

#[test]
fn drop_loses_excess_lanes() {
    yielding::drop_loses_excess_lanes(&yielding::make_device());
}

#[test]
fn continuation_yields_to_itself() {
    yielding::continuation_yields_to_itself(&yielding::make_device());
}

#[test]
fn chained_continuations_with_multi_element_results() {
    yielding::chained_continuations_with_multi_element_results(&yielding::make_device());
}

#[test]
fn arena_overflow_rejects() {
    yielding::arena_overflow_rejects(&yielding::make_device());
}

#[test]
fn node_handler_resolves_on_gpu() {
    yielding::node_handler_resolves_on_gpu(&yielding::make_device());
}

#[test]
fn validation_errors() {
    yielding::validation_errors(&yielding::make_device());
}

#[test]
fn ordered_with_neighbouring_nodes() {
    yielding::ordered_with_neighbouring_nodes(&yielding::make_device());
}
