//! Generic shader specialization prediction for retained schemes.
//!
//! Design: `docs/src/design/shader-specialization.md`. Every [`crate::Scheme`] owns one
//! [`SchemePredictor`]; every compute dispatch node that carries at least one `with_param`
//! scalar gets a [`SitePredictor`]. On each submit the scheme hands the predictor its IR
//! and whether the scheme was otherwise clean. The predictor keeps a per-slot streak of
//! clean submits during which the scalar word held its value, compiles a variant with the
//! stable slots baked in once the streaks pass the warm threshold, swaps the variant onto
//! the node once they pass the promote threshold, and — through the scheme's
//! `set_node_param` — demotes the node back to its universal pipeline the moment a baked
//! word changes.
//!
//! Nothing here changes what a dispatch computes. The universal pipeline reads every scalar
//! from the push-constant word; a variant reads the baked ones as literals through
//! [`scalar_specialization_macro`]. Both have the same binding layout on every backend
//! where the predictor runs (see
//! `GpuBackend::compute_pipeline_layout_follows_signature`).

use crate::backend::ComputePipelineHandle;
use crate::compute::ComputePipeline;
use crate::runtime::Runtime;
use crate::shader::{ShaderModule, ShaderProvenance};
use crate::slang::virtual_main::scalar_specialization_macro;
use crate::task_graph::{GraphIR, NodeKind};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Thresholds the predictor runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpecializationPolicy {
    /// Clean submits a slot must hold its value before a variant baking it is compiled.
    pub warm_after: u32,
    /// Clean submits every baked slot must have held its value before the variant is swapped in.
    pub promote_after: u32,
    /// Failed variant compiles before a site is pinned to its universal pipeline for good.
    pub max_failures: u32,
    /// Compiled variants a scheme keeps alive beyond the ones currently promoted.
    pub max_cached_variants: usize,
}

impl SpecializationPolicy {
    pub const DEFAULT_WARM_AFTER: u32 = 2;
    pub const DEFAULT_PROMOTE_AFTER: u32 = 10;
    pub const DEFAULT_MAX_FAILURES: u32 = 3;
    pub const DEFAULT_MAX_CACHED_VARIANTS: usize = 16;
}

impl Default for SpecializationPolicy {
    fn default() -> Self {
        Self {
            warm_after: Self::DEFAULT_WARM_AFTER,
            promote_after: Self::DEFAULT_PROMOTE_AFTER,
            max_failures: Self::DEFAULT_MAX_FAILURES,
            max_cached_variants: Self::DEFAULT_MAX_CACHED_VARIANTS,
        }
    }
}

/// `(slot, wire word)` pairs in ascending slot order — the identity of one variant.
///
/// Slots below [`TENSOR_FACT_SLOT_BASE`] are scalar params; the rest are tensor layout facts.
pub(crate) type BakedSlots = Vec<(u32, u32)>;

/// First slot id of a tensor layout fact; see [`tensor_fact_slot`].
pub(crate) const TENSOR_FACT_SLOT_BASE: u32 = 1 << 16;
const TENSOR_FACT_STRIDE: u32 = 16;
const _: () = assert!(goldy_shader_ir::TENSOR_FACTS.len() <= TENSOR_FACT_STRIDE as usize);

/// Slot id of layout field `TENSOR_FACTS[fact]` of tensor slot `tensor`.
pub(crate) fn tensor_fact_slot(tensor: u32, fact: usize) -> u32 {
    TENSOR_FACT_SLOT_BASE + tensor * TENSOR_FACT_STRIDE + fact as u32
}

fn is_fact_slot(slot: u32) -> bool {
    slot >= TENSOR_FACT_SLOT_BASE
}

/// `(tensor slot, fact index)` of a fact slot id.
fn split_fact_slot(slot: u32) -> (u32, usize) {
    let rel = slot - TENSOR_FACT_SLOT_BASE;
    (rel / TENSOR_FACT_STRIDE, (rel % TENSOR_FACT_STRIDE) as usize)
}

/// Preprocessor macro that bakes `slot` of `entry`.
fn bake_macro(entry: &str, slot: u32) -> String {
    if is_fact_slot(slot) {
        let (tensor, fact) = split_fact_slot(slot);
        goldy_shader_ir::tensor_fact_macro(entry, tensor, fact)
    } else {
        scalar_specialization_macro(entry, slot)
    }
}

/// The program a variant specializes.
///
/// A module that declares a [`crate::shader::KernelIdentity`] shares variants with every
/// other module of that kernel program; any other module has variants of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VariantKey {
    Module(u64),
    Kernel(goldy_shader_ir::KernelId),
}

impl VariantKey {
    fn of(provenance: &ShaderProvenance) -> Self {
        match provenance.kernel() {
            Some(kernel) => Self::Kernel(kernel.id),
            None => Self::Module(provenance.id()),
        }
    }
}

/// Counters the predictor bumps; the scheme folds them into [`crate::scheme::ReplayStats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SpecializationEvents {
    pub warms: u64,
    pub promotions: u64,
    pub demotions: u64,
}

/// Variants a scheme has compiled, most recently used at the back.
///
/// Shared with compile workers so a compile that finished after its site lost interest
/// (cancelled, or the scheme moved on) still lands here instead of being thrown away.
/// Holding a variant here does not promote it; sites re-earn promotion through streaks.
struct VariantCache {
    entries: VecDeque<VariantEntry>,
    capacity: usize,
}

struct VariantEntry {
    key: VariantKey,
    baked: BakedSlots,
    pipeline: Arc<ComputePipeline>,
}

impl VariantCache {
    fn get(&mut self, key: VariantKey, baked: &[(u32, u32)]) -> Option<Arc<ComputePipeline>> {
        let pos = self.entries.iter().position(|e| e.key == key && e.baked == baked)?;
        let entry = self.entries.remove(pos).expect("position came from iter");
        let pipeline = Arc::clone(&entry.pipeline);
        self.entries.push_back(entry);
        Some(pipeline)
    }

    fn insert(&mut self, key: VariantKey, baked: BakedSlots, pipeline: Arc<ComputePipeline>) {
        self.entries.retain(|e| !(e.key == key && e.baked == baked));
        self.entries.push_back(VariantEntry { key, baked, pipeline });
        while self.entries.len() > self.capacity {
            self.entries.pop_front();
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// One variant compile, shared by every site of the scheme that wants the same variant.
struct CompileJob {
    key: VariantKey,
    baked: BakedSlots,
    /// Sites holding a [`WarmJob`] on this compile. The worker skips the compile when none
    /// are left by the time it starts.
    holders: AtomicUsize,
    /// `None` while running; `Some(Ok)` once the variant is in the cache.
    outcome: Mutex<Option<Result<(), String>>>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

/// A site's hold on a [`CompileJob`].
struct WarmJob {
    baked: BakedSlots,
    job: Arc<CompileJob>,
}

impl WarmJob {
    /// Hold `job` too, unless every holder has already let it go.
    fn attach(job: &Arc<CompileJob>) -> Option<Self> {
        job.holders
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| (n > 0).then_some(n + 1))
            .ok()?;
        Some(Self {
            baked: job.baked.clone(),
            job: Arc::clone(job),
        })
    }

    fn poll(&self) -> Option<Result<(), String>> {
        self.job.outcome.lock().unwrap().clone()
    }
}

impl Drop for WarmJob {
    fn drop(&mut self) {
        // The worker keeps its own reference and still files a compile that already started
        // in the shared cache.
        self.job.holders.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A compiled variant a site is holding but has not swapped in yet.
struct Candidate {
    baked: BakedSlots,
    pipeline: Arc<ComputePipeline>,
}

/// Per-dispatch-site predictor state.
pub(crate) struct SitePredictor {
    /// The pipeline the caller bound. Everything demotes back to this.
    universal: ComputePipelineHandle,
    provenance: Arc<ShaderProvenance>,
    /// `[goldy_compute]` function name the bake macros are scoped to.
    entry: String,
    label: crate::SchemeLabel,
    /// Facts that hold for the node's lifetime: `#[fact]` scalars and tensor layout facts.
    /// They skip the streak and join every bake target.
    certain: BakedSlots,
    /// Scalar words seen at the previous submit.
    last: Vec<u32>,
    /// Per slot: consecutive clean submits the word has held its current value.
    streak: Vec<u32>,
    /// Per slot: streak a slot needs before it is baked. Starts at `warm_after`; every
    /// time a slot invalidates a compile or a promotion it grows, so a fact that flips
    /// every few frames stops causing compiles.
    bake_threshold: Vec<u32>,
    failures: u32,
    pinned: bool,
    promoted: Option<Candidate>,
    ready: Option<Candidate>,
    job: Option<WarmJob>,
}

impl SitePredictor {
    fn new(
        universal: ComputePipelineHandle,
        provenance: Arc<ShaderProvenance>,
        entry: String,
        label: crate::SchemeLabel,
        slots: &[u32],
        certain: &[(u32, u32)],
        policy: &SpecializationPolicy,
    ) -> Self {
        debug_assert!(certain
            .iter()
            .all(|&(s, _)| is_fact_slot(s) || (s as usize) < slots.len()));
        debug_assert!(certain.windows(2).all(|w| w[0].0 < w[1].0));
        Self {
            universal,
            provenance,
            entry,
            label,
            certain: certain.to_vec(),
            last: slots.to_vec(),
            streak: vec![0; slots.len()],
            bake_threshold: vec![policy.warm_after; slots.len()],
            failures: 0,
            pinned: false,
            promoted: None,
            ready: None,
            job: None,
        }
    }

    fn is_promoted(&self) -> bool {
        self.promoted.is_some()
    }

    fn variant_key(&self) -> VariantKey {
        VariantKey::of(&self.provenance)
    }

    /// Kernel identity for diagnostics; `-` for a module without one.
    fn kernel(&self) -> String {
        self.provenance
            .kernel()
            .map_or_else(|| "-".into(), |k| k.id.to_string())
    }

    /// `baked` as `name=word` pairs, naming scalar slots by their kernel's scalar origins
    /// and facts as `t{tensor}.{field}`.
    fn describe(&self, baked: &[(u32, u32)]) -> String {
        let names = self.provenance.kernel().map(|k| k.scalars.as_slice()).unwrap_or(&[]);
        baked
            .iter()
            .map(|&(slot, word)| {
                if is_fact_slot(slot) {
                    let (tensor, fact) = split_fact_slot(slot);
                    return format!("t{tensor}.{}={word:#x}", goldy_shader_ir::TENSOR_FACTS[fact]);
                }
                match names.get(slot as usize) {
                    Some(name) => format!("{name}={word:#x}"),
                    None => format!("slot{slot}={word:#x}"),
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn is_certain(&self, slot: u32) -> bool {
        self.certain.iter().any(|&(s, _)| s == slot)
    }

    /// The certain facts, and the slots that have held their value long enough to be
    /// baked, with those values.
    fn bake_target(&self, slots: &[u32]) -> BakedSlots {
        let mut target: BakedSlots = slots
            .iter()
            .enumerate()
            .filter(|&(s, _)| self.streak[s] >= self.bake_threshold[s] && !self.is_certain(s as u32))
            .map(|(s, &word)| (s as u32, word))
            .collect();
        target.extend_from_slice(&self.certain);
        target.sort_unstable_by_key(|&(s, _)| s);
        target
    }

    fn all_baked_still_hold(&self, baked: &[(u32, u32)], slots: &[u32]) -> bool {
        baked
            .iter()
            .all(|&(s, word)| is_fact_slot(s) || slots[s as usize] == word)
    }

    fn all_baked_at_least(&self, baked: &[(u32, u32)], threshold: u32) -> bool {
        baked
            .iter()
            .all(|&(s, _)| is_fact_slot(s) || self.is_certain(s) || self.streak[s as usize] >= threshold)
    }

    fn burn_slot(&mut self, slot: usize, policy: &SpecializationPolicy) {
        let t = &mut self.bake_threshold[slot];
        *t = (*t).saturating_mul(2).max(policy.promote_after).min(1 << 16);
    }

    /// Advance streaks for one submit.
    ///
    /// Slots whose word changed reset to zero (on any submit). Slots that held their word
    /// advance only on clean submits: a scheme that re-records every frame for other reasons
    /// keeps its history but does not earn promotions from it.
    fn observe(&mut self, slots: &[u32], ir_clean: bool) {
        for ((streak, &now), &before) in self.streak.iter_mut().zip(slots).zip(&self.last) {
            if now != before {
                *streak = 0;
            } else if ir_clean {
                *streak = streak.saturating_add(1);
            }
        }
        self.last.copy_from_slice(slots);
    }

    fn reset_streaks(&mut self) {
        self.streak.iter_mut().for_each(|s| *s = 0);
    }
}

/// What the scheme must do to a node after a predictor step.
enum NodeChange {
    None,
    /// Bind this pipeline on the node (params-dirty).
    Bind(ComputePipelineHandle),
}

/// The predictor a [`crate::Scheme`] owns.
pub(crate) struct SchemePredictor {
    policy: SpecializationPolicy,
    /// Keyed by node index in the scheme IR (nodes are append-only).
    sites: HashMap<u32, SitePredictor>,
    variants: Arc<Mutex<VariantCache>>,
    /// Variants unbound from a node in recent submits. A demoted variant's last command
    /// list may still be executing; the pipeline is held here across two further submits
    /// before its `Arc` is released (the cache usually still holds it after that anyway).
    retiring: VecDeque<Vec<Arc<ComputePipeline>>>,
    /// `GpuBackend::compute_pipeline_layout_follows_signature`, queried once.
    backend_supported: Option<bool>,
    events: SpecializationEvents,
    /// Compiles started by this scheme's sites. A site warming a variant that is already
    /// compiling joins that compile instead of starting its own.
    inflight: Vec<std::sync::Weak<CompileJob>>,
}

impl SchemePredictor {
    pub(crate) fn new() -> Self {
        Self::with_policy(SpecializationPolicy::default())
    }

    pub(crate) fn with_policy(policy: SpecializationPolicy) -> Self {
        Self {
            policy,
            sites: HashMap::new(),
            variants: Arc::new(Mutex::new(VariantCache {
                entries: VecDeque::new(),
                capacity: policy.max_cached_variants,
            })),
            retiring: VecDeque::from(vec![Vec::new(), Vec::new()]),
            backend_supported: None,
            events: SpecializationEvents::default(),
            inflight: Vec::new(),
        }
    }

    /// Counters accumulated so far (the scheme copies them into its stats).
    pub(crate) fn events(&self) -> SpecializationEvents {
        self.events
    }

    /// Register (or re-register, after a caller-side pipeline swap) a dispatch site.
    ///
    /// `certain` are facts that hold for the node's lifetime, in ascending slot order:
    /// `#[fact]` scalar slots and tensor layout facts (see [`tensor_fact_slot`]). Sites
    /// with neither scalar params nor certain facts, or whose shader has no single
    /// `[goldy_compute]` entry to scope bake macros to, are not tracked.
    pub(crate) fn register_site(
        &mut self,
        node: u32,
        universal: ComputePipelineHandle,
        provenance: &Arc<ShaderProvenance>,
        label: crate::SchemeLabel,
        slots: &[u32],
        certain: &[(u32, u32)],
    ) {
        if let Some(old) = self.sites.remove(&node) {
            self.retire_site(old);
        }
        if slots.is_empty() && certain.is_empty() {
            return;
        }
        let Some(entry) = provenance.compute_entry() else {
            return;
        };
        let site = SitePredictor::new(
            universal,
            Arc::clone(provenance),
            entry.to_string(),
            label,
            slots,
            certain,
            &self.policy,
        );
        self.sites.insert(node, site);
    }

    /// The certain facts `node` was registered with.
    pub(crate) fn certain_facts(&self, node: u32) -> BakedSlots {
        self.sites.get(&node).map(|s| s.certain.clone()).unwrap_or_default()
    }

    /// Re-register child's tracked dispatch sites at `map(child node)` in the parent IR;
    /// `None` skips a site.
    pub(crate) fn copy_sites_mapped(&mut self, child: &Self, map: impl Fn(u32) -> Option<u32>) {
        let snapshot: Vec<_> = child
            .sites
            .iter()
            .filter_map(|(&idx, site)| {
                Some((
                    map(idx)?,
                    site.universal,
                    Arc::clone(&site.provenance),
                    site.label.clone(),
                    site.last.clone(),
                    site.certain.clone(),
                ))
            })
            .collect();
        for (idx, universal, provenance, label, slots, certain) in snapshot {
            self.register_site(idx, universal, &provenance, label, &slots, &certain);
        }
    }

    /// Stop tracking `node`, releasing its variants through the retire queue.
    pub(crate) fn remove_site(&mut self, node: u32) {
        if let Some(site) = self.sites.remove(&node) {
            self.retire_site(site);
        }
    }

    /// Move every site to `map(node)`, keeping its history; `None` removes the site.
    pub(crate) fn rekey(&mut self, map: impl Fn(u32) -> Option<u32>) {
        let sites = std::mem::take(&mut self.sites);
        for (node, site) in sites {
            match map(node) {
                Some(to) => {
                    self.sites.insert(to, site);
                }
                None => self.retire_site(site),
            }
        }
    }

    /// Whether `node` currently runs a predictor-chosen variant instead of the caller's pipeline.
    pub(crate) fn is_promoted(&self, node: u32) -> bool {
        self.sites.get(&node).is_some_and(SitePredictor::is_promoted)
    }

    /// The caller changed scalar `slot` on `node`. Demote if the running variant baked it.
    ///
    /// Returns the universal pipeline the scheme must rebind, if a demotion happened.
    pub(crate) fn on_param_changed(&mut self, node: u32, slot: usize) -> Option<ComputePipelineHandle> {
        let policy = self.policy;
        let site = self.sites.get_mut(&node)?;
        if slot >= site.streak.len() {
            return None;
        }
        let slot_id = slot as u32;
        // A declared fact the caller changes is an ordinary param from now on.
        site.certain.retain(|&(s, _)| s != slot_id);
        let mut burned = false;
        if site
            .job
            .as_ref()
            .is_some_and(|j| j.baked.iter().any(|&(s, _)| s == slot_id))
        {
            site.job.take();
            burned = true;
        }
        if site
            .ready
            .as_ref()
            .is_some_and(|c| c.baked.iter().any(|&(s, _)| s == slot_id))
        {
            let dropped = site.ready.take().expect("checked");
            self.retiring
                .back_mut()
                .expect("two generations")
                .push(dropped.pipeline);
            burned = true;
        }
        let mut rebind = None;
        if site
            .promoted
            .as_ref()
            .is_some_and(|c| c.baked.iter().any(|&(s, _)| s == slot_id))
        {
            let demoted = site.promoted.take().expect("checked");
            self.retiring
                .back_mut()
                .expect("two generations")
                .push(demoted.pipeline);
            self.events.demotions += 1;
            tracing::debug!(
                node,
                label = %site.label,
                kernel = %site.kernel(),
                slot,
                baked = %site.describe(&demoted.baked),
                "specialization: demoted (baked param changed)"
            );
            rebind = Some(site.universal);
            burned = true;
        }
        if burned {
            site.burn_slot(slot, &policy);
        }
        rebind
    }

    /// Run the predictor at the top of a submit, before dirtiness is read for recording.
    ///
    /// `ir_clean` is whether the scheme was clean coming into this submit. Returns `true`
    /// when a node's pipeline was rebound (the scheme must mark itself params-dirty).
    pub(crate) fn begin_submit(
        &mut self,
        device: &Runtime,
        ir: &mut GraphIR,
        ir_clean: bool,
        topo_dirty: bool,
    ) -> bool {
        if self.sites.is_empty() {
            return false;
        }
        if !self.enabled(device) {
            return self.disable_all(ir);
        }
        let policy = self.policy;
        let mut rebound = false;
        self.inflight
            .retain(|j| j.upgrade().is_some_and(|j| j.holders.load(Ordering::Acquire) > 0));
        let mut node_indices: Vec<u32> = self.sites.keys().copied().collect();
        node_indices.sort_unstable();
        for node in node_indices {
            let Some(NodeKind::Dispatch {
                pipeline, user_slots, ..
            }) = ir.nodes.get_mut(node as usize).map(|n| &mut n.kind)
            else {
                continue;
            };
            let slots: Vec<u32> = user_slots.clone();
            let site = self.sites.get_mut(&node).expect("iterating own keys");
            if slots.len() != site.last.len() {
                // Shape drift is not something the builder allows; be defensive anyway.
                continue;
            }
            if topo_dirty {
                site.reset_streaks();
            }
            site.observe(&slots, ir_clean);
            match Self::step_site(
                site,
                &slots,
                device,
                &self.variants,
                &mut self.retiring,
                &mut self.inflight,
                &mut self.events,
                &policy,
                node,
            ) {
                NodeChange::None => {}
                NodeChange::Bind(handle) => {
                    *pipeline = handle;
                    rebound = true;
                }
            }
        }
        rebound
    }

    /// Release variants retired two submits ago.
    pub(crate) fn end_submit(&mut self) {
        self.retiring.pop_front();
        self.retiring.push_back(Vec::new());
    }

    /// Join every in-flight compile (tests).
    pub(crate) fn wait_for_compiles(&mut self) {
        for site in self.sites.values_mut() {
            if let Some(job) = site.job.as_mut() {
                let thread = job.job.thread.lock().unwrap().take();
                if let Some(thread) = thread {
                    let _ = thread.join();
                }
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn cached_variants(&self) -> usize {
        self.variants.lock().unwrap().len()
    }

    #[cfg(test)]
    pub(crate) fn site_streaks(&self, node: u32) -> Option<Vec<u32>> {
        self.sites.get(&node).map(|s| s.streak.clone())
    }

    #[cfg(test)]
    pub(crate) fn site_is_pinned(&self, node: u32) -> bool {
        self.sites.get(&node).is_some_and(|s| s.pinned)
    }

    #[cfg(test)]
    pub(crate) fn site_has_job(&self, node: u32) -> bool {
        self.sites.get(&node).is_some_and(|s| s.job.is_some())
    }

    fn enabled(&mut self, device: &Runtime) -> bool {
        if !crate::validation_env::specialization_enabled() {
            return false;
        }
        *self.backend_supported.get_or_insert_with(|| {
            device
                .inner
                .backend
                .lock()
                .unwrap()
                .compute_pipeline_layout_follows_signature()
        })
    }

    /// Put every site back on its universal pipeline and forget its history.
    fn disable_all(&mut self, ir: &mut GraphIR) -> bool {
        let mut rebound = false;
        for (&node, site) in self.sites.iter_mut() {
            site.job.take();
            if let Some(c) = site.ready.take() {
                self.retiring.back_mut().expect("two generations").push(c.pipeline);
            }
            if let Some(c) = site.promoted.take() {
                self.retiring.back_mut().expect("two generations").push(c.pipeline);
                if let Some(NodeKind::Dispatch { pipeline, .. }) = ir.nodes.get_mut(node as usize).map(|n| &mut n.kind)
                {
                    *pipeline = site.universal;
                    rebound = true;
                }
                self.events.demotions += 1;
            }
            site.reset_streaks();
        }
        rebound
    }

    fn retire_site(&mut self, mut site: SitePredictor) {
        site.job.take();
        let gen = self.retiring.back_mut().expect("two generations");
        if let Some(c) = site.ready.take() {
            gen.push(c.pipeline);
        }
        if let Some(c) = site.promoted.take() {
            gen.push(c.pipeline);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step_site(
        site: &mut SitePredictor,
        slots: &[u32],
        device: &Runtime,
        variants: &Arc<Mutex<VariantCache>>,
        retiring: &mut VecDeque<Vec<Arc<ComputePipeline>>>,
        inflight: &mut Vec<std::sync::Weak<CompileJob>>,
        events: &mut SpecializationEvents,
        policy: &SpecializationPolicy,
        node: u32,
    ) -> NodeChange {
        if site.pinned {
            return NodeChange::None;
        }

        // A compile whose baked facts no longer hold is wasted work; stop it.
        if let Some(job) = site.job.as_ref() {
            if !site.all_baked_still_hold(&job.baked, slots) {
                let baked = job.baked.clone();
                site.job.take();
                // `observe` already ran: a baked slot whose word moved has streak 0.
                for (s, _) in baked.into_iter().filter(|&(s, _)| !is_fact_slot(s)) {
                    if site.streak[s as usize] == 0 {
                        site.burn_slot(s as usize, policy);
                    }
                }
            }
        }
        if let Some(c) = site.ready.as_ref() {
            if !site.all_baked_still_hold(&c.baked, slots) {
                let c = site.ready.take().expect("checked");
                retiring.back_mut().expect("two generations").push(c.pipeline);
            }
        }

        // Collect a finished compile.
        if let Some(outcome) = site.job.as_ref().and_then(WarmJob::poll) {
            let job = site.job.take().expect("checked");
            match outcome {
                Ok(()) => {
                    let pipeline = variants.lock().unwrap().get(site.variant_key(), &job.baked);
                    match pipeline {
                        Some(pipeline) => {
                            site.ready = Some(Candidate {
                                baked: job.baked.clone(),
                                pipeline,
                            })
                        }
                        // Evicted between insert and poll (cache smaller than the working set).
                        None => {
                            tracing::debug!(node, label = %site.label, "specialization: variant evicted before use")
                        }
                    }
                }
                Err(err) => {
                    site.failures += 1;
                    tracing::warn!(
                        node,
                        label = %site.label,
                        kernel = %site.kernel(),
                        baked = %site.describe(&job.baked),
                        failures = site.failures,
                        %err,
                        "specialization: variant compile failed"
                    );
                    if site.failures >= policy.max_failures {
                        site.pinned = true;
                        tracing::warn!(
                            node,
                            label = %site.label,
                            "specialization: site pinned to universal pipeline"
                        );
                        return NodeChange::None;
                    }
                }
            }
        }

        // Promote a ready variant once every baked slot has proven itself.
        if site
            .ready
            .as_ref()
            .is_some_and(|c| site.all_baked_at_least(&c.baked, policy.promote_after))
        {
            let next = site.ready.take().expect("checked");
            if let Some(prev) = site.promoted.take() {
                retiring.back_mut().expect("two generations").push(prev.pipeline);
            }
            let handle = next.pipeline.handle;
            events.promotions += 1;
            tracing::debug!(
                node,
                label = %site.label,
                kernel = %site.kernel(),
                baked = %site.describe(&next.baked),
                "specialization: promoted"
            );
            site.promoted = Some(next);
            return NodeChange::Bind(handle);
        }

        // Nothing in flight: decide whether to warm a (wider) variant.
        if site.job.is_none() && site.ready.is_none() {
            let target = site.bake_target(slots);
            let already = site.promoted.as_ref().map(|c| c.baked.as_slice()).unwrap_or(&[]);
            if !target.is_empty() && target != already {
                let cached = variants.lock().unwrap().get(site.variant_key(), &target);
                match cached {
                    Some(pipeline) => {
                        site.ready = Some(Candidate {
                            baked: target,
                            pipeline,
                        });
                    }
                    None => {
                        let key = site.variant_key();
                        let joined = inflight
                            .iter()
                            .filter_map(std::sync::Weak::upgrade)
                            .find(|j| j.key == key && j.baked == target && j.outcome.lock().unwrap().is_none())
                            .and_then(|j| WarmJob::attach(&j));
                        let (job, action) = match joined {
                            Some(job) => (job, "specialization: joining in-flight compile"),
                            None => {
                                events.warms += 1;
                                let job = spawn_compile(device, site, target, variants);
                                inflight.push(Arc::downgrade(&job.job));
                                (job, "specialization: warming")
                            }
                        };
                        tracing::debug!(
                            node,
                            label = %site.label,
                            kernel = %site.kernel(),
                            baked = %site.describe(&job.baked),
                            "{action}"
                        );
                        site.job = Some(job);
                    }
                }
            }
        }
        NodeChange::None
    }
}

/// Compile `baked` for `site` on a worker thread; the result lands in `variants`.
fn spawn_compile(
    device: &Runtime,
    site: &SitePredictor,
    baked: BakedSlots,
    variants: &Arc<Mutex<VariantCache>>,
) -> WarmJob {
    let job = Arc::new(CompileJob {
        key: site.variant_key(),
        baked: baked.clone(),
        holders: AtomicUsize::new(1),
        outcome: Mutex::new(None),
        thread: Mutex::new(None),
    });

    let device = device.clone();
    let provenance = Arc::clone(&site.provenance);
    let entry = site.entry.clone();
    let label = site.label.clone();
    let variants = Arc::clone(variants);
    let worker_job = Arc::clone(&job);

    let thread = std::thread::Builder::new()
        .name("goldy-specialize".into())
        .spawn(move || {
            let job = worker_job;
            let result = compile_variant(&device, &provenance, &entry, &label, &job.baked, &job.holders);
            let filed = match result {
                Ok(Some(pipeline)) => {
                    variants
                        .lock()
                        .unwrap()
                        .insert(job.key, job.baked.clone(), Arc::new(pipeline));
                    Ok(())
                }
                // Abandoned before it did any work: nothing to report, nothing to cache.
                Ok(None) => return,
                Err(err) => Err(err),
            };
            *job.outcome.lock().unwrap() = Some(filed);
        });

    match thread {
        Ok(handle) => *job.thread.lock().unwrap() = Some(handle),
        Err(err) => *job.outcome.lock().unwrap() = Some(Err(format!("spawn specialization worker: {err}"))),
    }

    WarmJob { baked, job }
}

/// `Ok(None)` when every holder let go before the compile started.
fn compile_variant(
    device: &Runtime,
    provenance: &ShaderProvenance,
    entry: &str,
    label: &crate::SchemeLabel,
    baked: &[(u32, u32)],
    holders: &AtomicUsize,
) -> Result<Option<ComputePipeline>, String> {
    if holders.load(Ordering::Acquire) == 0 {
        return Ok(None);
    }
    let defines: Vec<(String, String)> = baked
        .iter()
        .map(|&(slot, word)| (bake_macro(entry, slot), format!("{word}u")))
        .collect();
    let define_refs: Vec<(&str, &str)> = defines.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    // Once Slang is running, abandonment is advisory: the module compile cannot be
    // aborted, but a result that arrives after every holder let go is still worth caching.
    let module = ShaderModule::from_provenance(device, provenance, &define_refs).map_err(|e| format!("{e:#}"))?;
    let pipeline =
        ComputePipeline::new_with_label(device, &module, Some(label.as_str())).map_err(|e| format!("{e:#}"))?;
    Ok(Some(pipeline))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_is_lru_and_bounded() {
        let dev = crate::test_support::mock_runtime();
        let shader = ShaderModule::from_slang(
            &dev,
            "[goldy_compute]\n[numthreads(1,1,1)]\nvoid k(Scattered<uint> d, ThreadId id, uint a) { d[id.x] = a; }",
        )
        .unwrap();
        let mk = || Arc::new(ComputePipeline::new(&dev, &shader).unwrap());
        let mut cache = VariantCache {
            entries: VecDeque::new(),
            capacity: 2,
        };
        let (one, two) = (VariantKey::Module(1), VariantKey::Module(2));
        cache.insert(one, vec![(0, 1)], mk());
        cache.insert(one, vec![(0, 2)], mk());
        assert!(cache.get(one, &[(0, 1)]).is_some(), "touch makes (0,1) most recent");
        cache.insert(one, vec![(0, 3)], mk());
        assert_eq!(cache.len(), 2);
        assert!(cache.get(one, &[(0, 2)]).is_none(), "least recently used was evicted");
        assert!(cache.get(one, &[(0, 1)]).is_some());
        assert!(cache.get(one, &[(0, 3)]).is_some());
        assert!(cache.get(two, &[(0, 3)]).is_none(), "keyed by program too");
        let kernel = VariantKey::Kernel(goldy_shader_ir::KernelId(1));
        assert!(cache.get(kernel, &[(0, 3)]).is_none(), "a kernel id is not a module id");
    }

    #[test]
    fn bake_target_follows_per_slot_thresholds() {
        let dev = crate::test_support::mock_runtime();
        let shader = ShaderModule::from_slang(
            &dev,
            "[goldy_compute]\n[numthreads(1,1,1)]\nvoid k(Scattered<uint> d, ThreadId id, uint a, uint b) { d[id.x] = a + b; }",
        )
        .unwrap();
        let pipeline = ComputePipeline::new(&dev, &shader).unwrap();
        let policy = SpecializationPolicy::default();
        let mut site = SitePredictor::new(
            pipeline.handle,
            Arc::clone(&pipeline.provenance),
            "k".into(),
            "t".into(),
            &[7, 9],
            &[],
            &policy,
        );
        for _ in 0..2 {
            site.observe(&[7, 9], true);
        }
        assert_eq!(site.bake_target(&[7, 9]), vec![(0, 7), (1, 9)]);
        site.observe(&[7, 4], true);
        assert_eq!(site.streak, vec![3, 0]);
        assert_eq!(site.bake_target(&[7, 4]), vec![(0, 7)]);
        site.burn_slot(1, &policy);
        assert_eq!(site.bake_threshold[1], policy.promote_after);
        site.burn_slot(1, &policy);
        assert_eq!(site.bake_threshold[1], policy.promote_after * 2);
        // Not-clean submits keep but do not advance a held slot.
        site.observe(&[7, 4], false);
        assert_eq!(site.streak, vec![3, 0]);
    }

    #[test]
    fn certain_facts_skip_the_streak() {
        let dev = crate::test_support::mock_runtime();
        let shader = ShaderModule::from_slang(
            &dev,
            "[goldy_compute]\n[numthreads(1,1,1)]\nvoid k(Scattered<uint> d, ThreadId id, uint a) { d[id.x] = a; }",
        )
        .unwrap();
        let pipeline = ComputePipeline::new(&dev, &shader).unwrap();
        let policy = SpecializationPolicy::default();
        let numel = tensor_fact_slot(1, 1);
        // A `#[fact]` scalar in slot 0, and two tensor layout facts.
        let certain = vec![(0, 7), (tensor_fact_slot(0, 0), 1), (numel, 4096)];
        let site = SitePredictor::new(
            pipeline.handle,
            Arc::clone(&pipeline.provenance),
            "k".into(),
            "t".into(),
            &[7],
            &certain,
            &policy,
        );
        // Before any submit only the facts are bakeable, and they already count as proven.
        assert_eq!(site.bake_target(&[7]), certain);
        assert!(site.all_baked_at_least(&certain, policy.promote_after));
        assert!(site.all_baked_still_hold(&certain, &[7]));
        assert!(
            !site.all_baked_still_hold(&certain, &[8]),
            "a scalar fact is still checked"
        );
        assert_eq!(bake_macro("k", numel), goldy_shader_ir::tensor_fact_macro("k", 1, 1));
        assert_eq!(site.describe(&[(numel, 16)]), "t1.numel=0x10");
    }

    #[test]
    fn identical_warms_share_one_compile() {
        let dev = crate::test_support::mock_runtime();
        let shader = ShaderModule::from_slang(
            &dev,
            "[goldy_compute]\n[numthreads(1,1,1)]\nvoid k(Scattered<uint> d, ThreadId id, uint a) { d[id.x] = a; }",
        )
        .unwrap();
        let pipeline = ComputePipeline::new(&dev, &shader).unwrap();
        let policy = SpecializationPolicy::default();
        let site = SitePredictor::new(
            pipeline.handle,
            Arc::clone(&pipeline.provenance),
            "k".into(),
            "t".into(),
            &[7],
            &[],
            &policy,
        );
        let variants = Arc::new(Mutex::new(VariantCache {
            entries: VecDeque::new(),
            capacity: 4,
        }));
        let first = spawn_compile(&dev, &site, vec![(0, 7)], &variants);
        let second = WarmJob::attach(&first.job).expect("first still holds the compile");
        assert!(Arc::ptr_eq(&first.job, &second.job));
        assert_eq!(first.job.holders.load(Ordering::Acquire), 2);
        drop(first);
        assert_eq!(second.job.holders.load(Ordering::Acquire), 1);
        let thread = second.job.thread.lock().unwrap().take();
        thread.expect("worker spawned").join().unwrap();
        assert_eq!(second.poll(), Some(Ok(())));
        assert_eq!(variants.lock().unwrap().len(), 1);
        let job = Arc::clone(&second.job);
        drop(second);
        assert!(WarmJob::attach(&job).is_none(), "nobody holds it any more");
    }
}
