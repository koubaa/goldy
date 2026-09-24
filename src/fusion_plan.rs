//! Automatic fusion of retained scheme dispatches.
//!
//! A scheme's `GraphIR` is what the caller recorded. When automatic fusion is on
//! ([`crate::Scheme::set_automatic_fusion`], else `GOLDY_FUSION`; see
//! `docs/src/programming-model/rust-kernels.md`), the scheme
//! derives an execution plan from it: every maximal run of adjacent generated-kernel
//! dispatches that the explicit [`crate::kernel::FusedKernel`] admission accepts runs
//! as one fused dispatch, and every other node runs as recorded. The recorded IR is
//! never rewritten.
//!
//! Planning waits until the recorded structure has survived one submit, so one-shot
//! schemes compile nothing. Fused pipelines compile on worker threads while the scheme
//! keeps submitting the recorded IR; the plan is promoted in one step once every
//! compile it needs has finished, leaving regions whose compile failed unfused. Any
//! structural change drops the plan; compiled fused pipelines stay cached by
//! [`KernelId`], so the same regions re-promote without compiling again.
//!
//! A region whose fused dispatch forwards a scheme-local temporary ([`crate::Temporary`])
//! that no node outside the region binds elides it: the temporary lives only in
//! registers and the fused dispatch does not bind it.

use crate::backend::ComputePipelineHandle;
use crate::kernel::{access_kind_to_node, admit, prepare_fused, ArgShape, PreparedKernel, StageView};
use crate::runtime::Runtime;
use crate::scheme::NodeId;
use crate::shader::ShaderProvenance;
use crate::task_graph::{
    DispatchDim, GraphIR, GroupInfo, NodeKind, ResourceBinding, ResourceId, TaskNode, TransientId,
};
use crate::temporary::{TemporaryUse, TEMPORARY_SLOT_PLACEHOLDER};
use crate::types::ResourceAccess;
use crate::SchemeLabel;
use goldy_shader_ir::{FusedDefinition, FusionRejection, KernelDef, KernelId};
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::{Arc, Mutex};

/// Whether records of `def` carry a kernel site: a retained definition, no tensor formals.
pub(crate) fn fusable_kernel(def: &KernelDef) -> bool {
    def.definition.is_some() && !def.params.iter().any(|p| p.is_tensor)
}

/// One actual argument of a recorded kernel dispatch, in declaration order.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SiteArg {
    /// Resource slot `slot` of the node, bound through a `descriptor` view.
    ///
    /// `resource` is `None` for parcels fusion does not track (present leases,
    /// deposits, samplers, acceleration structures).
    Resource {
        resource: Option<ResourceId>,
        slot: usize,
        descriptor: ResourceAccess,
    },
    /// User slot `slot` of the node.
    Scalar { slot: usize },
}

/// A kernel site being collected by a node builder.
pub(crate) struct SiteDraft {
    kernel: Arc<KernelDef>,
    args: Vec<SiteArg>,
}

impl SiteDraft {
    pub(crate) fn new(kernel: Arc<KernelDef>) -> Self {
        Self {
            args: Vec::with_capacity(kernel.params.len()),
            kernel,
        }
    }

    pub(crate) fn resource(&mut self, resource: Option<ResourceId>, slot: usize, descriptor: ResourceAccess) {
        let resource = resource.filter(|r| {
            matches!(
                r,
                ResourceId::Buffer(_)
                    | ResourceId::BufferRange { .. }
                    | ResourceId::Texture(_)
                    | ResourceId::TransientBuffer(_)
            )
        });
        self.args.push(SiteArg::Resource {
            resource,
            slot,
            descriptor,
        });
    }

    pub(crate) fn scalar(&mut self, slot: usize) {
        self.args.push(SiteArg::Scalar { slot });
    }

    /// The site of a node with `bindings` bindings, if its arguments follow the signature
    /// and it declares no bindings besides its tracked resource arguments.
    pub(crate) fn finish(
        self,
        universal: ComputePipelineHandle,
        provenance: &Arc<ShaderProvenance>,
        bindings: usize,
    ) -> Option<KernelSite> {
        let params = &self.kernel.params;
        let shaped = params.len() == self.args.len()
            && params
                .iter()
                .zip(&self.args)
                .all(|(p, a)| p.category.is_resource() == matches!(a, SiteArg::Resource { .. }));
        let tracked = self
            .args
            .iter()
            .filter(|a| matches!(a, SiteArg::Resource { resource: Some(_), .. }))
            .count();
        (shaped && tracked == bindings).then(|| KernelSite {
            kernel: self.kernel,
            universal,
            provenance: Arc::clone(provenance),
            args: self.args,
        })
    }
}

/// A recorded dispatch of a generated kernel: what admission and plan building need.
#[derive(Clone)]
pub(crate) struct KernelSite {
    pub(crate) kernel: Arc<KernelDef>,
    /// The pipeline the caller recorded, before any specialization.
    pub(crate) universal: ComputePipelineHandle,
    pub(crate) provenance: Arc<ShaderProvenance>,
    pub(crate) args: Vec<SiteArg>,
}

impl KernelSite {
    fn shapes(&self) -> Vec<ArgShape> {
        self.args
            .iter()
            .map(|a| match *a {
                SiteArg::Resource { resource, .. } => ArgShape::Resource(resource),
                SiteArg::Scalar { .. } => ArgShape::Scalar,
            })
            .collect()
    }
}

/// Automatic fusion state of a scheme; see [`crate::Scheme::fusion_report`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FusionReport {
    /// Runs of recorded dispatches the planner fuses or tried to fuse.
    pub regions: Vec<FusionRegion>,
    /// Runs of adjacent generated-kernel dispatches that admission rejected.
    pub rejected: Vec<RejectedFusion>,
}

/// A run of recorded dispatches planned as one fused dispatch.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionRegion {
    /// Recorded constituents, in execution order.
    pub nodes: Vec<NodeId>,
    pub labels: Vec<SchemeLabel>,
    /// Identity of the fused program.
    pub kernel: KernelId,
    /// Parcels whose values the fused dispatch forwards between constituents in registers.
    pub forwarded: usize,
    /// Forwarded scheme-local temporaries that exist only in registers: the fused
    /// dispatch never stores them and binds no storage for them.
    pub elided: usize,
    pub status: FusionRegionStatus,
}

/// Whether a [`FusionRegion`] runs fused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusionRegionStatus {
    /// The fused pipeline is compiling; the constituents run as recorded meanwhile.
    Compiling,
    /// The scheme runs the region as one dispatch.
    Promoted,
    /// The fused pipeline could not be compiled or bound; the constituents run as recorded.
    Failed(String),
}

/// Adjacent generated-kernel dispatches that stay separate.
#[derive(Debug, Clone, PartialEq)]
pub struct RejectedFusion {
    /// The run admission was asked to fuse; the last node is the one that could not join.
    pub nodes: Vec<NodeId>,
    pub labels: Vec<SchemeLabel>,
    pub reason: FusionRejection,
}

/// Counters the planner bumps; the scheme folds them into [`crate::ReplayStats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FusionEvents {
    pub(crate) promotions: u64,
    pub(crate) fallbacks: u64,
    pub(crate) compile_failures: u64,
}

/// The IR a scheme submits while a plan is promoted, and how it maps to the recorded IR.
pub(crate) struct Plan {
    pub(crate) ir: GraphIR,
    /// Executed node of every recorded node.
    exec_of: Vec<u32>,
    /// Recorded node of every executed node that is not fused.
    recorded_of: Vec<Option<u32>>,
    pub(crate) fused: Vec<FusedNode>,
    /// Every binding of a temporary in `ir`.
    pub(crate) temporary_uses: Vec<TemporaryUse>,
}

/// One executed dispatch standing for a run of recorded ones.
pub(crate) struct FusedNode {
    pub(crate) exec: u32,
    pub(crate) nodes: Range<usize>,
    /// Per constituent: the fused user slot each of its user slots binds.
    scalar_slots: Vec<Vec<usize>>,
    pub(crate) kernel: Arc<PreparedKernel>,
}

impl Plan {
    pub(crate) fn exec_of(&self, recorded: usize) -> Option<u32> {
        self.exec_of.get(recorded).copied()
    }

    pub(crate) fn recorded_of(&self, exec: u32) -> Option<u32> {
        self.recorded_of.get(exec as usize).copied().flatten()
    }

    /// The fused node recorded `node` runs in, if any.
    pub(crate) fn fused_of(&self, node: usize) -> Option<&FusedNode> {
        self.fused.iter().find(|f| f.nodes.contains(&node))
    }

    /// Executed node and user slot that user slot `slot` of recorded `node` runs as.
    pub(crate) fn executed_param(&self, node: usize, slot: usize) -> Option<(usize, usize)> {
        match self.fused_of(node) {
            Some(f) => f.scalar_slots[node - f.nodes.start]
                .get(slot)
                .map(|&s| (f.exec as usize, s)),
            None => self.exec_of(node).map(|e| (e as usize, slot)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// The recorded structure changed since the last submit.
    Settling,
    /// The structure survived a submit; the next submit plans.
    Unplanned,
    /// Waiting for fused compiles.
    Compiling,
    /// Nothing more to do until the structure changes.
    Settled,
}

struct Region {
    nodes: Range<usize>,
    definition: FusedDefinition,
    id: KernelId,
    status: FusionRegionStatus,
}

struct Rejection {
    nodes: Range<usize>,
    reason: FusionRejection,
}

type Compiled = Result<Arc<PreparedKernel>, String>;

/// Fused pipelines by program identity, shared with compile workers.
#[derive(Default)]
struct Compiles {
    done: HashMap<KernelId, Compiled>,
    running: HashSet<KernelId>,
}

/// The planner a [`crate::Scheme`] owns.
pub(crate) struct FusionPlanner {
    phase: Phase,
    regions: Vec<Region>,
    rejected: Vec<Rejection>,
    compiles: Arc<Mutex<Compiles>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    /// Failed compiles already counted in `events`.
    failures_seen: HashSet<KernelId>,
    plan: Option<Plan>,
    events: FusionEvents,
}

impl FusionPlanner {
    pub(crate) fn new() -> Self {
        Self {
            phase: Phase::Settling,
            regions: Vec::new(),
            rejected: Vec::new(),
            compiles: Arc::new(Mutex::new(Compiles::default())),
            workers: Vec::new(),
            failures_seen: HashSet::new(),
            plan: None,
            events: FusionEvents::default(),
        }
    }

    pub(crate) fn events(&self) -> FusionEvents {
        self.events
    }

    pub(crate) fn plan(&self) -> Option<&Plan> {
        self.plan.as_ref()
    }

    pub(crate) fn plan_mut(&mut self) -> Option<&mut Plan> {
        self.plan.as_mut()
    }

    pub(crate) fn install(&mut self, plan: Plan) {
        self.plan = Some(plan);
    }

    /// Forget the planned regions because the recorded structure changed.
    ///
    /// Returns the promoted plan, which the scheme must revert.
    pub(crate) fn reset(&mut self) -> Option<Plan> {
        self.phase = Phase::Settling;
        self.regions.clear();
        self.rejected.clear();
        let plan = self.plan.take();
        if plan.is_some() {
            self.events.fallbacks += 1;
        }
        plan
    }

    /// Whether recorded `node` belongs to a planned or promoted region.
    pub(crate) fn involves(&self, node: usize) -> bool {
        self.regions.iter().any(|r| r.nodes.contains(&node))
    }

    pub(crate) fn end_submit(&mut self) {
        if self.phase == Phase::Settling {
            self.phase = Phase::Unplanned;
        }
    }

    /// Join every in-flight fused compile (tests).
    pub(crate) fn wait_for_compiles(&mut self) {
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }

    /// Advance planning at the top of a submit. Returns a plan for the scheme to promote.
    ///
    /// `temporaries` lists every recorded binding of a scheme-local temporary.
    pub(crate) fn step(
        &mut self,
        device: &Runtime,
        ir: &GraphIR,
        sites: &HashMap<u32, KernelSite>,
        temporaries: &[TemporaryUse],
    ) -> Option<Plan> {
        if self.phase == Phase::Unplanned {
            self.match_regions(ir, sites, temporaries);
            if self.regions.is_empty() {
                self.phase = Phase::Settled;
                return None;
            }
            self.request_compiles(device);
            self.phase = Phase::Compiling;
        }
        if self.phase != Phase::Compiling {
            return None;
        }
        let outcomes: Vec<Compiled> = {
            let compiles = self.compiles.lock().unwrap();
            let mut outcomes = Vec::with_capacity(self.regions.len());
            for region in &self.regions {
                outcomes.push(compiles.done.get(&region.id)?.clone());
            }
            outcomes
        };
        self.phase = Phase::Settled;
        self.workers.retain(|w| !w.is_finished());

        let mut fused = Vec::new();
        for (region, outcome) in self.regions.iter_mut().zip(outcomes) {
            let built = outcome.and_then(|kernel| fused_node(ir, sites, region, &kernel).map(|node| (node, kernel)));
            match built {
                Ok((node, kernel)) => {
                    region.status = FusionRegionStatus::Promoted;
                    fused.push(Built {
                        nodes: region.nodes.clone(),
                        node,
                        kernel,
                    });
                }
                Err(err) => {
                    if self.failures_seen.insert(region.id) {
                        self.events.compile_failures += 1;
                    }
                    tracing::warn!(
                        kernel = %region.definition.name,
                        id = %region.id,
                        %err,
                        "kernel fusion: region stays unfused"
                    );
                    region.status = FusionRegionStatus::Failed(err);
                }
            }
        }
        if fused.is_empty() {
            return None;
        }
        self.events.promotions += 1;
        tracing::debug!(regions = fused.len(), "kernel fusion: plan promoted");
        Some(assemble(ir, fused, temporaries))
    }

    /// Greedily grow maximal admissible runs of adjacent kernel sites in one group.
    fn match_regions(&mut self, ir: &GraphIR, sites: &HashMap<u32, KernelSite>, temporaries: &[TemporaryUse]) {
        self.regions.clear();
        self.rejected.clear();
        let view = |i: usize| -> Option<StageView<'_>> {
            let site = sites.get(&(i as u32))?;
            let NodeKind::Dispatch {
                dispatch: DispatchDim::Direct { x, y, z },
                ..
            } = &ir.nodes.get(i)?.kind
            else {
                return None;
            };
            Some(StageView {
                kernel: &site.kernel.entry,
                definition: site.kernel.definition.as_ref(),
                workgroup_size: site.kernel.workgroup_size,
                groups: [*x, *y, *z],
                args: site.shapes(),
            })
        };
        let n = ir.nodes.len();
        let mut start = 0;
        while start < n {
            let Some(first) = view(start) else {
                start += 1;
                continue;
            };
            let mut stages = vec![first];
            let mut best = None;
            let mut end = start + 1;
            while end < n && ir.nodes[end].group == ir.nodes[start].group {
                let Some(next) = view(end) else { break };
                stages.push(next);
                match admit(&stages) {
                    Ok(definition) => {
                        best = Some(definition);
                        end += 1;
                    }
                    Err(reason) => {
                        tracing::debug!(%reason, first = start, next = end, "kernel fusion: run ends");
                        self.rejected.push(Rejection {
                            nodes: start..end + 1,
                            reason,
                        });
                        break;
                    }
                }
            }
            match best {
                Some(definition) => {
                    let definition = elide_local_temporaries(definition, start..end, sites, temporaries);
                    self.regions.push(Region {
                        nodes: start..end,
                        id: definition.id(),
                        definition,
                        status: FusionRegionStatus::Compiling,
                    });
                    start = end;
                }
                None => start += 1,
            }
        }
    }

    fn request_compiles(&mut self, device: &Runtime) {
        let fault = crate::validation_env::fusion_compile_fault();
        for region in &self.regions {
            {
                let mut compiles = self.compiles.lock().unwrap();
                if compiles.done.contains_key(&region.id) || !compiles.running.insert(region.id) {
                    continue;
                }
            }
            tracing::debug!(
                kernel = %region.definition.name,
                id = %region.id,
                stages = region.definition.stages.len(),
                forwarded = region.definition.forwarded.len(),
                elided = region.definition.elided.len(),
                "kernel fusion: compiling"
            );
            let device = device.clone();
            let definition = region.definition.clone();
            let compiles = Arc::clone(&self.compiles);
            let id = region.id;
            let spawned = std::thread::Builder::new().name("goldy-fuse".into()).spawn(move || {
                let outcome = if fault {
                    Err("injected fused compile failure".to_string())
                } else {
                    prepare_fused(&device, &definition)
                        .map(Arc::new)
                        .map_err(|e| format!("{e:#}"))
                };
                let mut compiles = compiles.lock().unwrap();
                compiles.running.remove(&id);
                compiles.done.insert(id, outcome);
            });
            match spawned {
                Ok(worker) => self.workers.push(worker),
                Err(err) => {
                    let mut compiles = self.compiles.lock().unwrap();
                    compiles.running.remove(&id);
                    compiles.done.insert(id, Err(format!("spawn fusion worker: {err}")));
                }
            }
        }
    }

    pub(crate) fn report(&self, ir: &GraphIR, node_id: impl Fn(usize) -> NodeId) -> FusionReport {
        let nodes = |r: &Range<usize>| r.clone().map(&node_id).collect();
        let labels = |r: &Range<usize>| r.clone().map(|i| ir.nodes[i].label.clone()).collect();
        FusionReport {
            regions: self
                .regions
                .iter()
                .map(|r| FusionRegion {
                    nodes: nodes(&r.nodes),
                    labels: labels(&r.nodes),
                    kernel: r.id,
                    forwarded: r.definition.forwarded.len(),
                    elided: r.definition.elided.len(),
                    status: r.status.clone(),
                })
                .collect(),
            rejected: self
                .rejected
                .iter()
                .map(|r| RejectedFusion {
                    nodes: nodes(&r.nodes),
                    labels: labels(&r.nodes),
                    reason: r.reason.clone(),
                })
                .collect(),
        }
    }
}

/// The recorded parcel fused parameter `j` of the region starting at recorded node `start` binds.
fn param_resource(
    definition: &FusedDefinition,
    start: usize,
    sites: &HashMap<u32, KernelSite>,
    j: usize,
) -> Option<ResourceId> {
    definition.stages.iter().enumerate().find_map(|(k, s)| {
        let i = s.args.iter().position(|&a| a == j)?;
        match sites.get(&((start + k) as u32))?.args.get(i)? {
            SiteArg::Resource { resource, .. } => *resource,
            SiteArg::Scalar { .. } => None,
        }
    })
}

/// Elide every forwarded temporary that no recorded node outside `nodes` binds.
fn elide_local_temporaries(
    definition: FusedDefinition,
    nodes: Range<usize>,
    sites: &HashMap<u32, KernelSite>,
    temporaries: &[TemporaryUse],
) -> FusedDefinition {
    let local: Vec<usize> = definition
        .forwarded
        .iter()
        .copied()
        .filter(|&j| match param_resource(&definition, nodes.start, sites, j) {
            Some(ResourceId::TransientBuffer(TransientId(t))) => temporaries
                .iter()
                .filter(|u| u.temporary == t)
                .all(|u| nodes.contains(&(u.node as usize))),
            _ => false,
        })
        .collect();
    if local.is_empty() {
        definition
    } else {
        definition.elide(&local)
    }
}

/// An executed fused dispatch before it has a position in the plan.
struct DraftNode {
    node: TaskNode,
    /// Per constituent: the fused user slot each of its user slots binds.
    scalar_slots: Vec<Vec<usize>>,
    /// Temporaries `node` binds; assembly fills in the node index.
    temporaries: Vec<TemporaryUse>,
}

/// The executed node of `region`.
fn fused_node(
    ir: &GraphIR,
    sites: &HashMap<u32, KernelSite>,
    region: &Region,
    kernel: &PreparedKernel,
) -> Result<DraftNode, String> {
    let definition = &region.definition;
    let pipeline = kernel.pipeline();
    let mut constituents = Vec::with_capacity(region.nodes.len());
    for i in region.nodes.clone() {
        let site = sites.get(&(i as u32)).ok_or("constituent lost its kernel site")?;
        let NodeKind::Dispatch {
            resource_slots,
            user_slots,
            dispatch,
            ..
        } = &ir.nodes[i].kind
        else {
            return Err("constituent is not a dispatch".into());
        };
        constituents.push((site, resource_slots, user_slots, dispatch));
    }

    let mut bindings = Vec::new();
    let mut resource_slots = Vec::new();
    let mut user_slots = Vec::new();
    let mut temporaries = Vec::new();
    let mut scalar_slots: Vec<Vec<usize>> = constituents.iter().map(|c| vec![0; c.2.len()]).collect();
    for (j, param) in definition.params.iter().enumerate() {
        if definition.elided.contains(&j) {
            continue;
        }
        let mut uses = definition.stages.iter().enumerate().flat_map(|(k, s)| {
            s.args
                .iter()
                .enumerate()
                .filter(move |&(_, &a)| a == j)
                .map(move |(i, _)| (k, i))
        });
        if param.category.is_resource() {
            // A slot is an SRV or a UAV of its parcel; bind the view the fused signature asks for.
            let want = pipeline.slot_access.get(resource_slots.len()).copied().flatten();
            let chosen = uses.find_map(|(k, i)| match constituents[k].0.args[i] {
                SiteArg::Resource {
                    resource: Some(resource),
                    slot,
                    descriptor,
                } if want.is_none_or(|w| is_uav(w) == is_uav(descriptor)) => Some((k, resource, slot, descriptor)),
                _ => None,
            });
            let (k, resource, slot, descriptor) =
                chosen.ok_or_else(|| format!("no recorded view of `{}` matches the fused signature", param.name))?;
            if let ResourceId::TransientBuffer(TransientId(temporary)) = resource {
                temporaries.push(TemporaryUse {
                    node: u32::MAX,
                    binding: bindings.len() as u32,
                    slot: resource_slots.len() as u32,
                    descriptor: want.unwrap_or(descriptor),
                    temporary,
                });
                resource_slots.push(TEMPORARY_SLOT_PLACEHOLDER);
            } else {
                resource_slots.push(constituents[k].1[slot]);
            }
            let access = param
                .access
                .ok_or_else(|| format!("resource `{}` declares no access", param.name))?;
            bindings.push(ResourceBinding {
                resource,
                access: access_kind_to_node(access),
            });
        } else {
            let (k, i) = uses
                .next()
                .ok_or_else(|| format!("scalar `{}` is bound by no formal", param.name))?;
            let SiteArg::Scalar { slot } = constituents[k].0.args[i] else {
                return Err(format!("scalar `{}` binds a resource argument", param.name));
            };
            scalar_slots[k][slot] = user_slots.len();
            user_slots.push(constituents[k].2[slot]);
        }
    }

    let first = &ir.nodes[region.nodes.start];
    let label = region
        .nodes
        .clone()
        .map(|i| ir.nodes[i].label.as_str())
        .collect::<Vec<_>>()
        .join("+");
    let node = TaskNode {
        label: label.into(),
        group: first.group,
        bindings,
        kind: NodeKind::Dispatch {
            pipeline: pipeline.handle,
            resource_slots,
            user_slots,
            dispatch: constituents[0].3.clone(),
        },
    };
    Ok(DraftNode {
        node,
        scalar_slots,
        temporaries,
    })
}

fn is_uav(access: ResourceAccess) -> bool {
    access != ResourceAccess::Read
}

/// A promoted region: its recorded run, executed node and pipeline.
struct Built {
    nodes: Range<usize>,
    node: DraftNode,
    kernel: Arc<PreparedKernel>,
}

/// The execution plan running each of `fused` in place of its recorded run.
///
/// `temporaries` lists the recorded IR's temporary bindings.
fn assemble(ir: &GraphIR, mut fused: Vec<Built>, temporaries: &[TemporaryUse]) -> Plan {
    fused.sort_by_key(|f| f.nodes.start);
    let n = ir.nodes.len();
    let mut plan = Plan {
        ir: GraphIR::default(),
        exec_of: vec![0; n],
        recorded_of: Vec::with_capacity(n),
        fused: Vec::with_capacity(fused.len()),
        temporary_uses: Vec::new(),
    };
    // Executed nodes created before each recorded node (and before the end).
    let mut before = vec![0usize; n + 1];
    let mut fused = fused.into_iter().peekable();
    let mut r = 0;
    while r < n {
        let exec = plan.ir.nodes.len();
        if fused.peek().is_some_and(|f| f.nodes.start == r) {
            let Built { nodes, node, kernel } = fused.next().expect("peeked");
            for c in nodes.clone() {
                plan.exec_of[c] = exec as u32;
                before[c] = if c == r { exec } else { exec + 1 };
            }
            plan.ir.nodes.push(node.node);
            plan.recorded_of.push(None);
            plan.temporary_uses.extend(
                node.temporaries
                    .into_iter()
                    .map(|u| TemporaryUse { node: exec as u32, ..u }),
            );
            r = nodes.end;
            plan.fused.push(FusedNode {
                exec: exec as u32,
                nodes,
                scalar_slots: node.scalar_slots,
                kernel,
            });
        } else {
            plan.exec_of[r] = exec as u32;
            before[r] = exec;
            plan.ir.nodes.push(ir.nodes[r].clone());
            plan.recorded_of.push(Some(r as u32));
            r += 1;
        }
    }
    let unfused = temporaries
        .iter()
        .filter(|u| plan.recorded_of[plan.exec_of[u.node as usize] as usize].is_some());
    plan.temporary_uses.extend(unfused.map(|u| TemporaryUse {
        node: plan.exec_of[u.node as usize],
        ..*u
    }));
    before[n] = plan.ir.nodes.len();
    plan.ir.groups = ir
        .groups
        .iter()
        .map(|g| GroupInfo {
            node_range: before[g.node_range.start]..before[g.node_range.end],
            ..g.clone()
        })
        .collect();
    plan.ir.extra_edges = ir
        .extra_edges
        .iter()
        .map(|&(i, j)| (plan.exec_of[i] as usize, plan.exec_of[j] as usize))
        .filter(|(i, j)| i != j)
        .collect();
    plan.ir.extra_group_edges = ir.extra_group_edges.clone();
    plan
}
