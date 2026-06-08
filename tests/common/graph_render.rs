//! Submit a single render-pass node via [`TaskGraph`] and block until complete.

use goldy::{Device, NodeAccess, RenderTarget, TaskGraph};

/// Record one render pass on `target`, submit on `device`'s default context, and wait.
pub fn submit_render_pass(
    device: &Device,
    target: &RenderTarget,
    label: &'static str,
    record: impl FnOnce(&mut goldy::RenderPassBuilder<'_>),
) {
    let ctx = device.create_context().expect("context");
    let mut graph = TaskGraph::new();
    let mut pass = graph.render_pass(label, target);
    record(&mut pass);
    pass.finish_recorded();
    graph.dispatch(&ctx).expect("graph dispatch");
}
