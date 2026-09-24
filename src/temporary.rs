//! Scheme-local temporaries: buffers whose contents exist only for dataflow inside one
//! submission of the scheme that declared them.
//!
//! A [`Temporary`] is recorded as a logical [`ResourceId::TransientBuffer`] with
//! placeholder shader slots, and every binding of it is noted as a [`TemporaryUse`].
//! When the executed IR changes structure, the scheme lowers each use in place onto a
//! pool buffer it owns, so barriers, cross-submit hazards and command-list retention
//! treat the storage like any other parcel.
//!
//! Lowering packs temporaries whose lifetimes do not overlap onto one buffer. Two
//! temporaries share storage only when they have the same size and element stride and
//! every node and wave that binds one precedes every node and wave that binds the other,
//! so the ordering the shared buffer adds between them was already implied by the
//! schedule. A temporary that no executed node binds, such as one a fused dispatch
//! elides, gets no storage at all.

use crate::context::Context;
use crate::error::GoldyError;
use crate::parcel::ParcelStamp;
use crate::task_graph::analysis::{build_edges, schedule_waves};
use crate::task_graph::{GraphIR, NodeKind, ResourceId, TransientId};
use crate::types::{BufferFlags, BufferKind, ResourceAccess};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Resource slot a temporary's binding holds until its scheme lowers it.
pub(crate) const TEMPORARY_SLOT_PLACEHOLDER: u32 = u32::MAX - 3;

/// A buffer declared by [`crate::Scheme::temporary_buffer`].
///
/// Its contents are undefined when each submission of the scheme starts and
/// unobservable after it ends: storage may be shared with other temporaries of the same
/// scheme whose lifetimes do not overlap, and a fused dispatch that is the temporary's
/// only user may keep its elements in registers and never store them. The first access
/// in a submission should therefore write every element it later reads.
///
/// Bind it like a buffer, with [`crate::SchemeNodeBuilder::with_temporary`] or as a
/// buffer argument of a generated kernel's `record` or `invoke`. It belongs to the
/// scheme that declared it; binding it on another scheme is a record error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Temporary {
    pub(crate) scheme_id: u64,
    pub(crate) id: u32,
    len: u64,
    stride: u32,
}

impl Temporary {
    pub(crate) fn new(scheme_id: u64, id: u32, len: u64, stride: u32) -> Self {
        Self {
            scheme_id,
            id,
            len,
            stride,
        }
    }

    /// Number of elements.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Always `false`: a temporary has at least one element.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes per element.
    pub fn element_stride(&self) -> u32 {
        self.stride
    }

    /// Size in bytes: `len() * element_stride()`.
    pub fn byte_size(&self) -> u64 {
        self.len * u64::from(self.stride)
    }

    pub(crate) fn resource_id(&self) -> ResourceId {
        ResourceId::TransientBuffer(TransientId(self.id))
    }

    pub(crate) fn decl(&self) -> TemporaryDecl {
        TemporaryDecl {
            byte_size: self.byte_size(),
            stride: self.stride,
        }
    }
}

/// Storage shape of a declared temporary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TemporaryDecl {
    pub(crate) byte_size: u64,
    pub(crate) stride: u32,
}

/// One binding of a temporary: which node, binding and shader resource slot hold it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TemporaryUse {
    pub(crate) node: u32,
    pub(crate) binding: u32,
    pub(crate) slot: u32,
    /// Descriptor the slot needs (SRV or UAV).
    pub(crate) descriptor: ResourceAccess,
    pub(crate) temporary: u32,
}

/// Inclusive node and wave ranges over which the executed IR binds one temporary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Interval {
    pub(crate) temporary: u32,
    pub(crate) nodes: (u32, u32),
    pub(crate) waves: (u32, u32),
}

/// Group temporaries that may share one buffer.
///
/// Members of a group have the same shape, and each member's first node and first wave
/// come after the previous member's last node and last wave.
pub(crate) fn pack(decls: &[TemporaryDecl], mut intervals: Vec<Interval>) -> Vec<Vec<u32>> {
    struct Slot {
        decl: TemporaryDecl,
        last_node: u32,
        last_wave: u32,
        members: Vec<u32>,
    }
    intervals.sort_by_key(|i| (i.nodes.0, i.temporary));
    let mut slots: Vec<Slot> = Vec::new();
    for interval in intervals {
        let decl = decls[interval.temporary as usize];
        let free = slots
            .iter_mut()
            .find(|s| s.decl == decl && s.last_node < interval.nodes.0 && s.last_wave < interval.waves.0);
        match free {
            Some(slot) => {
                slot.last_node = interval.nodes.1;
                slot.last_wave = interval.waves.1;
                slot.members.push(interval.temporary);
            }
            None => slots.push(Slot {
                decl,
                last_node: interval.nodes.1,
                last_wave: interval.waves.1,
                members: vec![interval.temporary],
            }),
        }
    }
    slots.into_iter().map(|s| s.members).collect()
}

/// Liveness of every temporary `uses` binds in `ir`, whose temporary bindings must be logical.
fn intervals(ir: &GraphIR, uses: &[TemporaryUse]) -> Vec<Interval> {
    let schedule = schedule_waves(ir, &build_edges(ir));
    let mut wave_of = vec![0u32; ir.nodes.len()];
    for (w, wave) in schedule.waves.iter().enumerate() {
        for &node in &wave.node_indices {
            wave_of[node] = w as u32;
        }
    }
    let mut spans: BTreeMap<u32, Interval> = BTreeMap::new();
    for u in uses {
        let wave = wave_of[u.node as usize];
        spans
            .entry(u.temporary)
            .and_modify(|i| {
                i.nodes = (i.nodes.0.min(u.node), i.nodes.1.max(u.node));
                i.waves = (i.waves.0.min(wave), i.waves.1.max(wave));
            })
            .or_insert(Interval {
                temporary: u.temporary,
                nodes: (u.node, u.node),
                waves: (wave, wave),
            });
    }
    spans.into_values().collect()
}

/// Stamp registry changes a lowering asks of its scheme.
#[derive(Default)]
pub(crate) struct Lowered {
    /// Storage the executed IR now binds.
    pub(crate) bound: Vec<(ResourceId, Arc<ParcelStamp>)>,
    /// Storage returned to the pool, which the scheme must stop tracking.
    pub(crate) released: Vec<ResourceId>,
}

struct Backing {
    buffer: crate::Buffer,
    decl: TemporaryDecl,
    /// Temporaries the last lowering placed here.
    members: Vec<u32>,
}

/// Pool buffers a scheme's temporaries are lowered onto.
#[derive(Default)]
pub(crate) struct TemporaryStorage {
    backings: Vec<Backing>,
}

impl TemporaryStorage {
    /// Number of pool buffers the executed IR binds.
    pub(crate) fn backing_count(&self) -> usize {
        self.backings.len()
    }

    /// Lower every use in `uses` onto pool storage, patching `ir` in place.
    ///
    /// Storage no use needs any more goes back to the context pool; reuse there waits
    /// for the submissions that referenced it.
    pub(crate) fn lower(
        &mut self,
        ctx: &Context,
        ir: &mut GraphIR,
        uses: &[TemporaryUse],
        decls: &[TemporaryDecl],
    ) -> Result<Lowered, GoldyError> {
        for u in uses {
            ir.nodes[u.node as usize].bindings[u.binding as usize].resource =
                ResourceId::TransientBuffer(TransientId(u.temporary));
        }
        let groups = if uses.is_empty() {
            Vec::new()
        } else {
            pack(decls, intervals(ir, uses))
        };

        // Keep each group on a buffer that held one of its members, when one did.
        let mut old = std::mem::take(&mut self.backings);
        let mut lowered = Lowered::default();
        let mut backing_of: BTreeMap<u32, usize> = BTreeMap::new();
        for members in groups {
            let decl = decls[members[0] as usize];
            let reuse = old
                .iter()
                .position(|b| b.decl == decl && b.members.iter().any(|m| members.contains(m)))
                .or_else(|| old.iter().position(|b| b.decl == decl));
            let buffer = match reuse {
                Some(i) => old.swap_remove(i).buffer,
                None => ctx
                    .acquire_transient_buffer(
                        decl.byte_size,
                        BufferKind::Scattered,
                        BufferFlags::empty(),
                        Some(decl.stride),
                    )
                    .map_err(|e| ctx.classify(e))?,
            };
            for &m in &members {
                backing_of.insert(m, self.backings.len());
            }
            self.backings.push(Backing { buffer, decl, members });
        }
        for backing in old {
            lowered.released.push(backing.buffer.whole().resource_id());
            ctx.return_transient_buffer(backing.buffer);
        }

        for u in uses {
            let parcel = self.backings[backing_of[&u.temporary]].buffer.whole();
            let node = &mut ir.nodes[u.node as usize];
            node.bindings[u.binding as usize].resource = parcel.resource_id();
            let slots = match &mut node.kind {
                NodeKind::Dispatch { resource_slots, .. } | NodeKind::TraceRays { resource_slots, .. } => {
                    resource_slots
                }
                _ => unreachable!("temporaries bind only to dispatch nodes"),
            };
            slots[u.slot as usize] = parcel.resource_index(u.descriptor).ok_or_else(|| {
                GoldyError::Validation(format!(
                    "temporary {} has no {:?} descriptor for node `{}`",
                    u.temporary, u.descriptor, node.label
                ))
            })?;
        }
        for backing in &self.backings {
            let parcel = backing.buffer.whole();
            lowered.bound.push((parcel.resource_id(), parcel.stamp_handle()));
        }
        tracing::debug!(
            temporaries = backing_of.len(),
            backings = self.backings.len(),
            released = lowered.released.len(),
            "scheme temporaries lowered"
        );
        Ok(lowered)
    }

    /// Return every buffer to the pool. The caller has waited for the scheme's submissions.
    pub(crate) fn release_all(&mut self, ctx: &Context) {
        for backing in self.backings.drain(..) {
            ctx.return_transient_buffer(backing.buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F32: TemporaryDecl = TemporaryDecl {
        byte_size: 256,
        stride: 4,
    };

    fn iv(temporary: u32, nodes: (u32, u32), waves: (u32, u32)) -> Interval {
        Interval {
            temporary,
            nodes,
            waves,
        }
    }

    #[test]
    fn disjoint_lifetimes_share_and_overlapping_ones_do_not() {
        // A chain a → t0 → b → t1 → c → t2 → d, one node per wave.
        let decls = [F32; 3];
        let chain = vec![iv(0, (0, 1), (0, 1)), iv(1, (1, 2), (1, 2)), iv(2, (2, 3), (2, 3))];
        assert_eq!(pack(&decls, chain), [vec![0, 2], vec![1]]);
    }

    #[test]
    fn sharing_needs_the_same_shape() {
        let decls = [
            F32,
            TemporaryDecl {
                byte_size: 512,
                stride: 4,
            },
            TemporaryDecl {
                byte_size: 256,
                stride: 8,
            },
        ];
        let later = vec![iv(0, (0, 1), (0, 1)), iv(1, (2, 3), (2, 3)), iv(2, (4, 5), (4, 5))];
        assert_eq!(pack(&decls, later), [vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn sharing_needs_both_record_order_and_wave_order() {
        let decls = [F32; 2];
        // Recorded apart but scheduled in one wave: independent work stays parallel.
        assert_eq!(
            pack(&decls, vec![iv(0, (0, 0), (0, 0)), iv(1, (1, 1), (0, 0))]).len(),
            2
        );
        // Scheduled apart but interleaved in record order.
        assert_eq!(
            pack(&decls, vec![iv(0, (0, 2), (0, 0)), iv(1, (1, 3), (1, 1))]).len(),
            2
        );
        assert_eq!(
            pack(&decls, vec![iv(0, (0, 0), (0, 0)), iv(1, (1, 1), (1, 1))]).len(),
            1
        );
    }
}
