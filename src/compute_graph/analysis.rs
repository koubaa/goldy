//! Dependency analysis, wave scheduling, and command emission.
//!
//! This module implements the core scheduling algorithm:
//!
//! 1. **Edge construction**: for each pair of nodes (i, j) where i < j,
//!    a dependency edge exists if they share a resource and at least one
//!    writes (RAW, WAR, or WAW). Multiple reads create no edge (SWMR).
//!
//! 2. **Wave scheduling**: nodes are assigned to waves via BFS-based
//!    topological sort with longest-path depth tracking. Independent nodes
//!    share a wave and can execute concurrently on the GPU.
//!
//! 3. **Barrier computation**: for each wave boundary, only the specific
//!    resources involved in cross-wave dependency edges are listed in the
//!    barrier set.
//!
//! 4. **Command emission**: waves are serialized into a flat
//!    `Vec<ComputeCommand>` with `ResourceBarrier` commands between waves.

use std::collections::HashSet;

use super::ir::{BarrierSet, CompiledSchedule, DispatchKind, GraphIR, NodeAccess, Wave};
use super::ResourceId;
use crate::backend::ComputeCommand;

/// Returns true if accesses `a` and `b` on the same resource form a dependency
/// (RAW, WAR, or WAW). Two reads do not conflict.
fn accesses_conflict(a: NodeAccess, b: NodeAccess) -> bool {
    a.writes() || b.writes()
}

/// Build directed dependency edges between graph nodes.
///
/// An edge (i -> j) means node j depends on node i and must execute after it.
/// Edges are created when two nodes access the same resource and at least one writes.
pub fn build_edges(ir: &GraphIR) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    let n = ir.nodes.len();

    for j in 0..n {
        for i in 0..j {
            let conflict = ir.nodes[i].bindings.iter().any(|bi| {
                ir.nodes[j]
                    .bindings
                    .iter()
                    .any(|bj| bi.resource == bj.resource && accesses_conflict(bi.access, bj.access))
            });
            if conflict {
                edges.push((i, j));
            }
        }
    }

    edges
}

/// Schedule nodes into waves using a longest-path (depth) assignment.
///
/// Each node's wave index equals one plus the maximum wave index of its
/// predecessors. Nodes with no predecessors land in wave 0. Independent
/// nodes naturally share a wave.
pub fn schedule_waves(ir: &GraphIR, edges: &[(usize, usize)]) -> CompiledSchedule {
    let n = ir.nodes.len();
    if n == 0 {
        return CompiledSchedule { waves: Vec::new() };
    }

    // Adjacency list: for each node, which nodes depend on it.
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for &(from, to) in edges {
        successors[from].push(to);
        in_degree[to] += 1;
    }

    // BFS-based topological sort with depth tracking.
    let mut depth: Vec<usize> = vec![0; n];
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut processed = 0;

    while processed < queue.len() {
        let node = queue[processed];
        processed += 1;
        for &succ in &successors[node] {
            depth[succ] = depth[succ].max(depth[node] + 1);
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                queue.push(succ);
            }
        }
    }

    let num_waves = depth.iter().copied().max().unwrap_or(0) + 1;

    // Group nodes into waves.
    let mut wave_nodes: Vec<Vec<usize>> = vec![Vec::new(); num_waves];
    for (i, &d) in depth.iter().enumerate() {
        wave_nodes[d].push(i);
    }

    // For each wave (beyond wave 0), compute which resources need barriers.
    // A barrier is needed for resource R before wave W if:
    //   - some node in a prior wave writes R, and some node in wave W accesses R
    //   - OR some node in a prior wave reads R, and some node in wave W writes R
    let waves = wave_nodes
        .into_iter()
        .enumerate()
        .map(|(wave_idx, node_indices)| {
            let barriers_before = if wave_idx == 0 {
                BarrierSet::default()
            } else {
                compute_barriers(ir, edges, &depth, wave_idx, &node_indices)
            };
            Wave {
                node_indices,
                barriers_before,
            }
        })
        .collect();

    CompiledSchedule { waves }
}

/// Determine which resources need barriers before `wave_idx` executes.
fn compute_barriers(
    ir: &GraphIR,
    edges: &[(usize, usize)],
    depth: &[usize],
    wave_idx: usize,
    wave_nodes: &[usize],
) -> BarrierSet {
    let wave_set: HashSet<usize> = wave_nodes.iter().copied().collect();
    let mut barrier_resources: HashSet<ResourceId> = HashSet::new();

    // Any edge crossing into this wave means the shared resource needs a barrier.
    for &(from, to) in edges {
        if depth[from] < wave_idx && wave_set.contains(&to) {
            // Find conflicting resources between `from` and `to`.
            for bi in &ir.nodes[from].bindings {
                for bj in &ir.nodes[to].bindings {
                    if bi.resource == bj.resource && accesses_conflict(bi.access, bj.access) {
                        barrier_resources.insert(bi.resource);
                    }
                }
            }
        }
    }

    let mut buffers = Vec::new();
    let mut textures = Vec::new();
    for r in barrier_resources {
        match r {
            ResourceId::Buffer(h) => buffers.push(h),
            ResourceId::Texture(h) => textures.push(h),
        }
    }
    buffers.sort();
    textures.sort();

    BarrierSet { buffers, textures }
}

/// Emit a flat `Vec<ComputeCommand>` from a graph IR and its compiled schedule.
pub fn emit_commands(ir: &GraphIR, schedule: &CompiledSchedule) -> Vec<ComputeCommand> {
    let mut commands = Vec::new();

    for wave in &schedule.waves {
        if !wave.barriers_before.is_empty() {
            commands.push(ComputeCommand::ResourceBarrier {
                buffers: wave.barriers_before.buffers.clone(),
                textures: wave.barriers_before.textures.clone(),
            });
        }

        for &idx in &wave.node_indices {
            let node = &ir.nodes[idx];
            commands.push(ComputeCommand::SetPipeline(node.pipeline));
            if !node.push_constants.is_empty() {
                commands.push(ComputeCommand::SetPushConstantsRaw {
                    indices: node.push_constants.clone(),
                });
            }
            match &node.dispatch {
                DispatchKind::Direct { x, y, z } => {
                    commands.push(ComputeCommand::Dispatch {
                        workgroups_x: *x,
                        workgroups_y: *y,
                        workgroups_z: *z,
                    });
                }
                DispatchKind::Indirect { buffer, offset } => {
                    commands.push(ComputeCommand::DispatchIndirect {
                        buffer: *buffer,
                        offset: *offset,
                    });
                }
            }
        }
    }

    commands
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_graph::ir::{DispatchKind, GraphNode, ResourceBinding};

    fn buf(id: u64) -> ResourceId {
        ResourceId::Buffer(id)
    }

    fn node(
        label: &str,
        pipeline: u64,
        bindings: Vec<(ResourceId, NodeAccess)>,
        wg: u32,
    ) -> GraphNode {
        GraphNode {
            label: label.to_string(),
            pipeline,
            bindings: bindings
                .into_iter()
                .map(|(resource, access)| ResourceBinding { resource, access })
                .collect(),
            push_constants: Vec::new(),
            dispatch: DispatchKind::Direct { x: wg, y: 1, z: 1 },
        }
    }

    #[test]
    fn linear_chain_raw() {
        // A writes X, B reads X -> A before B
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
        assert_eq!(schedule.waves[0].node_indices, vec![0]);
        assert_eq!(schedule.waves[1].node_indices, vec![1]);
        assert!(!schedule.waves[1].barriers_before.is_empty());
        assert_eq!(schedule.waves[1].barriers_before.buffers, vec![0]);
    }

    #[test]
    fn independent_nodes_same_wave() {
        // A writes X, B writes Y -> no dependency, same wave
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(1), NodeAccess::Write)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
        assert_eq!(schedule.waves[0].node_indices, vec![0, 1]);
    }

    #[test]
    fn swmr_multiple_reads() {
        // A reads X, B reads X -> no conflict, same wave
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Read)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Read)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
        assert_eq!(schedule.waves[0].node_indices, vec![0, 1]);
    }

    #[test]
    fn war_edge() {
        // A reads X, B writes X -> WAR dependency
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Read)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Write)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
    }

    #[test]
    fn waw_edge() {
        // A writes X, B writes X -> WAW dependency
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(0), NodeAccess::Write)], 1),
            ],
        };
        let edges = build_edges(&ir);
        assert_eq!(edges, vec![(0, 1)]);

        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 2);
    }

    #[test]
    fn diamond_dependency() {
        //   A (writes X)
        //  / \
        // B   C  (both read X, write Y/Z respectively)
        //  \ /
        //   D  (reads Y and Z)
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node(
                    "B",
                    2,
                    vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Write)],
                    1,
                ),
                node(
                    "C",
                    3,
                    vec![(buf(0), NodeAccess::Read), (buf(2), NodeAccess::Write)],
                    1,
                ),
                node(
                    "D",
                    4,
                    vec![(buf(1), NodeAccess::Read), (buf(2), NodeAccess::Read)],
                    1,
                ),
            ],
        };
        let edges = build_edges(&ir);

        let schedule = schedule_waves(&ir, &edges);
        // Wave 0: A, Wave 1: B+C (both read X), Wave 2: D (reads Y,Z)
        assert_eq!(schedule.waves.len(), 3);
        assert_eq!(schedule.waves[0].node_indices, vec![0]);
        let mut w1 = schedule.waves[1].node_indices.clone();
        w1.sort();
        assert_eq!(w1, vec![1, 2]);
        assert_eq!(schedule.waves[2].node_indices, vec![3]);
    }

    #[test]
    fn empty_graph() {
        let ir = GraphIR { nodes: Vec::new() };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());
        let schedule = schedule_waves(&ir, &edges);
        assert!(schedule.waves.is_empty());
    }

    #[test]
    fn single_node() {
        let ir = GraphIR {
            nodes: vec![node("A", 1, vec![(buf(0), NodeAccess::ReadWrite)], 4)],
        };
        let edges = build_edges(&ir);
        assert!(edges.is_empty());
        let schedule = schedule_waves(&ir, &edges);
        assert_eq!(schedule.waves.len(), 1);
        assert!(schedule.waves[0].barriers_before.is_empty());
    }

    #[test]
    fn barrier_targets_correct_resources() {
        // A writes buf0, B writes buf1, C reads buf0 and buf1
        // A->C (buf0 RAW), B->C (buf1 RAW), A and B independent
        let ir = GraphIR {
            nodes: vec![
                node("A", 1, vec![(buf(0), NodeAccess::Write)], 1),
                node("B", 2, vec![(buf(1), NodeAccess::Write)], 1),
                node(
                    "C",
                    3,
                    vec![(buf(0), NodeAccess::Read), (buf(1), NodeAccess::Read)],
                    1,
                ),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);

        assert_eq!(schedule.waves.len(), 2);
        let barrier = &schedule.waves[1].barriers_before;
        assert_eq!(barrier.buffers, vec![0, 1]);
    }

    #[test]
    fn command_emission_linear_chain() {
        let ir = GraphIR {
            nodes: vec![
                node("A", 10, vec![(buf(0), NodeAccess::Write)], 8),
                node("B", 20, vec![(buf(0), NodeAccess::Read)], 4),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let cmds = emit_commands(&ir, &schedule);

        // Wave 0: SetPipeline(10), Dispatch(8,1,1)
        // ResourceBarrier([0])
        // Wave 1: SetPipeline(20), Dispatch(4,1,1)
        assert_eq!(cmds.len(), 5);
        assert!(matches!(cmds[0], ComputeCommand::SetPipeline(10)));
        assert!(matches!(
            cmds[1],
            ComputeCommand::Dispatch {
                workgroups_x: 8,
                ..
            }
        ));
        assert!(matches!(cmds[2], ComputeCommand::ResourceBarrier { .. }));
        assert!(matches!(cmds[3], ComputeCommand::SetPipeline(20)));
        assert!(matches!(
            cmds[4],
            ComputeCommand::Dispatch {
                workgroups_x: 4,
                ..
            }
        ));
    }

    #[test]
    fn command_emission_independent() {
        let ir = GraphIR {
            nodes: vec![
                node("A", 10, vec![(buf(0), NodeAccess::Write)], 8),
                node("B", 20, vec![(buf(1), NodeAccess::Write)], 4),
            ],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let cmds = emit_commands(&ir, &schedule);

        // Single wave, no barriers
        assert_eq!(cmds.len(), 4);
        assert!(!cmds
            .iter()
            .any(|c| matches!(c, ComputeCommand::ResourceBarrier { .. })));
    }

    #[test]
    fn command_emission_with_push_constants() {
        let ir = GraphIR {
            nodes: vec![GraphNode {
                label: "A".to_string(),
                pipeline: 10,
                bindings: vec![ResourceBinding {
                    resource: buf(0),
                    access: NodeAccess::Write,
                }],
                push_constants: vec![42, 7],
                dispatch: DispatchKind::Direct { x: 1, y: 1, z: 1 },
            }],
        };
        let edges = build_edges(&ir);
        let schedule = schedule_waves(&ir, &edges);
        let cmds = emit_commands(&ir, &schedule);

        assert_eq!(cmds.len(), 3);
        assert!(matches!(cmds[0], ComputeCommand::SetPipeline(10)));
        assert!(
            matches!(cmds[1], ComputeCommand::SetPushConstantsRaw { ref indices } if indices == &[42, 7])
        );
        assert!(matches!(
            cmds[2],
            ComputeCommand::Dispatch {
                workgroups_x: 1,
                ..
            }
        ));
    }
}
