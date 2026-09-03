//! Graph-level validation with human-readable errors (issue #112 item 7).
//!
//! Always-on checks catch cycles, mesh/raster command mix-ups, and BLAS/TLAS
//! misuse. [`crate::validation_env::scheme_validation_enabled`] adds stricter
//! lifetime/access hints (unused Accel builds, missing `ACCEL_INPUT`).

use super::analysis::build_edges;
use super::ir::{GraphIR, NodeAccess, NodeKind};
use super::ResourceId;
#[cfg(feature = "graphics")]
use crate::backend::RenderCommand;
use crate::error::GoldyError;
use std::collections::{HashMap, HashSet};

/// Fail submit when the IR cannot be executed as recorded.
pub(crate) fn validate_graph(ir: &GraphIR) -> Result<(), GoldyError> {
    let edges = build_edges(ir);
    validate_acyclic(ir, &edges)?;
    validate_render_pass_commands(ir)?;
    validate_accel_kind_uses(ir)?;
    if crate::validation_env::scheme_validation_enabled() {
        validate_scheme_strict(ir)?;
    }
    Ok(())
}

fn validation(msg: String) -> GoldyError {
    GoldyError::Validation(msg)
}

fn validate_acyclic(ir: &GraphIR, edges: &[(usize, usize)]) -> Result<(), GoldyError> {
    let n = ir.nodes.len();
    if n == 0 {
        return Ok(());
    }
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree = vec![0usize; n];
    for &(from, to) in edges {
        successors[from].push(to);
        in_degree[to] += 1;
    }
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut processed = 0;
    while processed < queue.len() {
        let node = queue[processed];
        processed += 1;
        for &succ in &successors[node] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push(succ);
            }
        }
    }
    if processed == n {
        return Ok(());
    }
    let leftover: Vec<&str> = (0..n)
        .filter(|&i| in_degree[i] > 0)
        .map(|i| ir.nodes[i].label)
        .collect();
    Err(validation(format!(
        "scheme graph contains a dependency cycle involving nodes {leftover:?}. \
         hint: each resource should flow forward in record order (build then read, \
         write then copy). Reverse or mutual writes on the same parcel create a cycle; \
         split the work into two schemes or record the producer first."
    )))
}

#[cfg(feature = "graphics")]
fn validate_render_pass_commands(ir: &GraphIR) -> Result<(), GoldyError> {
    for node in &ir.nodes {
        let NodeKind::RenderPass { commands, .. } = &node.kind else {
            continue;
        };
        let mut pipeline: Option<bool> = None;
        for cmd in commands {
            match cmd {
                RenderCommand::SetPipeline(_) => pipeline = Some(false),
                RenderCommand::SetMeshPipeline(_) => pipeline = Some(true),
                RenderCommand::Draw { .. } | RenderCommand::DrawIndexed { .. } => match pipeline {
                    None => {
                        return Err(validation(format!(
                            "render pass \"{}\" recorded draw/draw_indexed without a pipeline. \
                                 hint: call set_pipeline(&render_pipeline) before draw, or \
                                 set_mesh_pipeline + dispatch_mesh for mesh shaders.",
                            node.label
                        )));
                    }
                    Some(true) => {
                        return Err(validation(format!(
                            "render pass \"{}\" recorded draw/draw_indexed after set_mesh_pipeline. \
                                 hint: mesh pipelines use dispatch_mesh(x, y, z), not draw. \
                                 Call set_pipeline for a vertex/fragment pipeline before draw.",
                            node.label
                        )));
                    }
                    Some(false) => {}
                },
                RenderCommand::DispatchMesh { x, y, z } => {
                    if *x == 0 || *y == 0 || *z == 0 {
                        return Err(validation(format!(
                            "render pass \"{}\" recorded dispatch_mesh({x}, {y}, {z}) with a zero dimension. \
                             hint: mesh workgroup counts must be at least (1, 1, 1). Use a vertex pipeline \
                             and draw() if you do not want a mesh dispatch.",
                            node.label
                        )));
                    }
                    match pipeline {
                        None => {
                            return Err(validation(format!(
                                "render pass \"{}\" recorded dispatch_mesh without set_mesh_pipeline. \
                                 hint: bind a MeshPipeline first (requires DeviceCapabilities::mesh_shaders). \
                                 Vertex pipelines use set_pipeline + draw, not dispatch_mesh.",
                                node.label
                            )));
                        }
                        Some(false) => {
                            return Err(validation(format!(
                                "render pass \"{}\" recorded dispatch_mesh after set_pipeline (vertex/fragment). \
                                 hint: call set_mesh_pipeline(&mesh_pipeline) before dispatch_mesh.",
                                node.label
                            )));
                        }
                        Some(true) => {}
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "graphics"))]
fn validate_render_pass_commands(_ir: &GraphIR) -> Result<(), GoldyError> {
    Ok(())
}

/// Shader `Accel` parameters and TLAS builds must not use a BLAS handle as the dest/scene.
fn validate_accel_kind_uses(ir: &GraphIR) -> Result<(), GoldyError> {
    let mut kinds: HashMap<u64, bool> = HashMap::new();
    for node in &ir.nodes {
        if let NodeKind::BuildAccelerationStructure(cmd) = &node.kind {
            match cmd {
                crate::backend::AccelBuildCommand::BlasTriangles { dest, .. } => {
                    kinds.insert(*dest, false);
                }
                crate::backend::AccelBuildCommand::Tlas { dest, instances } => {
                    kinds.insert(*dest, true);
                    for inst in instances.iter() {
                        if kinds.get(&inst.blas) == Some(&true) {
                            return Err(validation(
                                "build_tlas instance references a TLAS, not a BLAS. \
                                 hint: AccelInstance.blas must be AccelerationStructure::blas_triangles. \
                                 Put instances of triangle BLASes into AccelerationStructure::tlas."
                                    .into(),
                            ));
                        }
                    }
                }
            }
        }
    }
    for node in &ir.nodes {
        let reads_accel = matches!(&node.kind, NodeKind::Dispatch { .. } | NodeKind::TraceRays { .. });
        if !reads_accel {
            continue;
        }
        for b in &node.bindings {
            let ResourceId::Accel(h) = b.resource else {
                continue;
            };
            if b.access == NodeAccess::Read && kinds.get(&h) == Some(&false) {
                return Err(validation(format!(
                    "node \"{}\" binds a BLAS as a shader Accel parameter. \
                     hint: RayQuery / TraceRay take a TLAS. Build the BLAS, then \
                     Scheme::build_tlas, and with_parcel(&tlas, NodeAccess::Read).",
                    node.label
                )));
            }
        }
    }
    Ok(())
}

fn validate_scheme_strict(ir: &GraphIR) -> Result<(), GoldyError> {
    let mut built: HashSet<u64> = HashSet::new();
    for node in &ir.nodes {
        if let NodeKind::BuildAccelerationStructure(cmd) = &node.kind {
            let dest = match cmd {
                crate::backend::AccelBuildCommand::BlasTriangles { dest, .. }
                | crate::backend::AccelBuildCommand::Tlas { dest, .. } => *dest,
            };
            built.insert(dest);
        }
        let traces = matches!(&node.kind, NodeKind::Dispatch { .. } | NodeKind::TraceRays { .. });
        if !traces {
            continue;
        }
        for b in &node.bindings {
            let ResourceId::Accel(h) = b.resource else {
                continue;
            };
            if b.access == NodeAccess::Read && !built.contains(&h) {
                return Err(validation(format!(
                    "node \"{}\" reads an acceleration structure that is not built in this scheme. \
                     hint: record Scheme::build_blas / build_tlas before the dispatch or trace_rays \
                     that binds Accel, or disable GOLDY_VALIDATION=scheme if the AS was built in an \
                     earlier submit on the same GPU object.",
                    node.label
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::AccelBuildCommand;
    use crate::task_graph::ir::{ResourceBinding, TaskNode};

    fn dispatch_reading_accel(label: &'static str, accel: u64) -> TaskNode {
        TaskNode {
            label,
            bindings: vec![ResourceBinding {
                resource: ResourceId::Accel(accel),
                access: NodeAccess::Read,
            }],
            kind: NodeKind::Dispatch {
                pipeline: 1,
                resource_slots: vec![0],
                user_slots: vec![],
                dispatch: crate::task_graph::DispatchDim::Direct { x: 1, y: 1, z: 1 },
            },
        }
    }

    #[test]
    fn empty_graph_ok() {
        validate_graph(&GraphIR::default()).expect("empty");
    }

    #[test]
    fn blas_as_shader_accel_is_rejected() {
        let ir = GraphIR {
            nodes: vec![
                TaskNode {
                    label: "build_blas",
                    bindings: vec![ResourceBinding {
                        resource: ResourceId::Accel(7),
                        access: NodeAccess::Overwrite,
                    }],
                    kind: NodeKind::BuildAccelerationStructure(AccelBuildCommand::BlasTriangles {
                        dest: 7,
                        vertex_buffer: 1,
                        vertex_offset: 0,
                        vertex_count: 3,
                        vertex_stride: 12,
                        index_buffer: None,
                        index_offset: 0,
                        index_count: 0,
                    }),
                },
                dispatch_reading_accel("trace", 7),
            ],
        };
        let err = validate_graph(&ir).expect_err("BLAS as Accel");
        let s = err.to_string();
        assert!(s.contains("BLAS"), "{s}");
        assert!(s.contains("TLAS"), "{s}");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn dispatch_mesh_without_pipeline_is_rejected() {
        let ir = GraphIR {
            nodes: vec![TaskNode {
                label: "mesh",
                bindings: vec![],
                kind: NodeKind::RenderPass {
                    target: 1,
                    color_load: crate::types::TargetLoad::Discard,
                    commands: vec![RenderCommand::DispatchMesh { x: 1, y: 1, z: 1 }],
                },
            }],
        };
        let err = validate_graph(&ir).expect_err("mesh");
        let s = err.to_string();
        assert!(s.contains("set_mesh_pipeline"), "{s}");
    }

    #[cfg(feature = "graphics")]
    #[test]
    fn draw_after_mesh_pipeline_is_rejected() {
        let ir = GraphIR {
            nodes: vec![TaskNode {
                label: "mesh",
                bindings: vec![],
                kind: NodeKind::RenderPass {
                    target: 1,
                    color_load: crate::types::TargetLoad::Discard,
                    commands: vec![
                        RenderCommand::SetMeshPipeline(1),
                        RenderCommand::Draw {
                            vertex_count: 3,
                            instance_count: 1,
                            first_vertex: 0,
                            first_instance: 0,
                        },
                    ],
                },
            }],
        };
        let err = validate_graph(&ir).expect_err("draw after mesh");
        let s = err.to_string();
        assert!(s.contains("dispatch_mesh"), "{s}");
    }
}
