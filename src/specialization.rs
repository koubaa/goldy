//! Generic shader specialization prediction for retained schemes.
//!
//! Design: `docs/src/design/shader-specialization.md`. Every [`crate::Scheme`] owns one
//! [`SchemePredictor`]; every compute dispatch node that carries a `with_param` scalar or
//! a tensor layout gets a [`SitePredictor`].
//!
//! A node's words change only through the scheme's `set_node_param`, so the predictor
//! learns of every change as an event. A word the caller never changed is the one the
//! node was recorded with, and it bakes at the node's first submit, as do layout facts.
//! A changed slot is predicted: it bakes once it has held its word for its threshold,
//! however dirty the rest of the scheme is. The variant is swapped onto the node once
//! every baked slot is certain or has held for the promote threshold, and
//! `set_node_param` demotes the node back to its universal pipeline the moment a baked
//! word changes. Each submit steps only the sites with something to decide.
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
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Thresholds the predictor runs on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpecializationPolicy {
    /// Submits a changed slot must hold its word before a variant baking it is compiled.
    pub warm_after: u32,
    /// Submits every baked changed slot must have held its word before the variant is swapped in.
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
/// Holding a variant here does not promote it; a site promotes it only once its slots prove out.
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
///
/// A node's scalar words change only through `Scheme::set_node_param`, so a slot the
/// caller has never changed is not a guess: it is the word the node was recorded with,
/// and it bakes at the first submit. Only a slot that has changed is predicted, by how
/// many submits it has since held its word.
pub(crate) struct SitePredictor {
    /// The pipeline the caller bound. Everything demotes back to this.
    universal: ComputePipelineHandle,
    provenance: Arc<ShaderProvenance>,
    /// `[goldy_compute]` function name the bake macros are scoped to.
    entry: String,
    label: crate::SchemeLabel,
    /// Tensor layout facts. They hold for the node's lifetime and join every bake target.
    facts: BakedSlots,
    /// The node's scalar words as the predictor last saw them.
    last: Vec<u32>,
    /// Per slot: whether the caller has changed it since the node's first submit.
    changed: Vec<bool>,
    /// Per slot: the first submit that ran its current word.
    held_since: Vec<u64>,
    /// Per slot: submits a changed slot must hold its word before it is baked. Every time
    /// a slot invalidates a compile or a promotion it grows, so a word that flips every
    /// few frames stops causing compiles.
    bake_threshold: Vec<u32>,
    /// Whether the site has been through a submit. Changes before that are still record time.
    submitted: bool,
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
        facts: &[(u32, u32)],
        policy: &SpecializationPolicy,
    ) -> Self {
        debug_assert!(facts.iter().all(|&(s, _)| is_fact_slot(s)));
        debug_assert!(facts.windows(2).all(|w| w[0].0 < w[1].0));
        Self {
            universal,
            provenance,
            entry,
            label,
            facts: facts.to_vec(),
            last: slots.to_vec(),
            changed: vec![false; slots.len()],
            held_since: vec![0; slots.len()],
            bake_threshold: vec![policy.warm_after; slots.len()],
            submitted: false,
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

    /// Submits, as of submit `now`, that `slot` has held its word.
    fn held(&self, slot: usize, now: u64) -> u64 {
        now.saturating_sub(self.held_since[slot])
    }

    /// Whether `slot` bakes without proof: a layout fact, or a scalar never changed.
    fn is_certain(&self, slot: u32) -> bool {
        is_fact_slot(slot) || !self.changed[slot as usize]
    }

    /// The certain slots, and the changed slots that have held their word long enough to
    /// be baked, with their words.
    fn bake_target(&self, now: u64) -> BakedSlots {
        let mut target: BakedSlots = self
            .last
            .iter()
            .enumerate()
            .filter(|&(s, _)| self.is_certain(s as u32) || self.held(s, now) >= u64::from(self.bake_threshold[s]))
            .map(|(s, &word)| (s as u32, word))
            .collect();
        target.extend_from_slice(&self.facts);
        target
    }

    /// Whether every baked slot is certain or has held its word for `threshold` submits.
    fn all_baked_proven(&self, baked: &[(u32, u32)], now: u64, threshold: u32) -> bool {
        baked
            .iter()
            .all(|&(s, _)| self.is_certain(s) || self.held(s as usize, now) >= u64::from(threshold))
    }

    fn burn_slot(&mut self, slot: usize, policy: &SpecializationPolicy) {
        let t = &mut self.bake_threshold[slot];
        *t = (*t).saturating_mul(2).max(policy.promote_after).min(1 << 16);
    }

    /// Record that `slot` now holds `word`, starting at submit `from`.
    ///
    /// Returns whether it changed. Before the first submit a change is still part of
    /// recording, and the slot stays certain.
    fn set_word(&mut self, slot: usize, word: u32, from: u64) -> bool {
        if self.last[slot] == word {
            return false;
        }
        self.last[slot] = word;
        self.held_since[slot] = from;
        if self.submitted {
            self.changed[slot] = true;
        }
        true
    }

    /// Forget how long every slot has held its word, as of submit `from`.
    fn restart(&mut self, from: u64) {
        self.held_since.iter_mut().for_each(|h| *h = from);
    }

    /// The submit at which this site next has something to decide with no event in between,
    /// or `None` if it has nothing to decide until a word changes.
    fn next_wake(&self, now: u64, policy: &SpecializationPolicy) -> Option<u64> {
        if self.pinned {
            return None;
        }
        if self.job.is_some() {
            return Some(now + 1);
        }
        if let Some(c) = &self.ready {
            // Every baked slot must reach the promote threshold.
            return c
                .baked
                .iter()
                .filter(|&&(s, _)| !self.is_certain(s))
                .map(|&(s, _)| self.held_since[s as usize] + u64::from(policy.promote_after))
                .max()
                .or(Some(now + 1));
        }
        // The target widens when the first unbaked changed slot reaches its threshold.
        let baked = self.promoted.as_ref().map(|c| c.baked.as_slice()).unwrap_or(&[]);
        (0..self.last.len())
            .filter(|&s| !self.is_certain(s as u32) && !baked.iter().any(|&(b, _)| b == s as u32))
            .map(|s| self.held_since[s] + u64::from(self.bake_threshold[s]))
            .min()
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
    /// Submits begun so far.
    now: u64,
    /// Sites the next submit must step: new, changed, or due.
    awake: BTreeSet<u32>,
    /// `(submit, node)`: a site with nothing to decide before that submit. Entries can be
    /// stale; stepping a site early is harmless.
    wake: BinaryHeap<Reverse<(u64, u32)>>,
    /// Whether the previous submit ran with prediction enabled.
    was_enabled: bool,
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
            now: 0,
            awake: BTreeSet::new(),
            wake: BinaryHeap::new(),
            was_enabled: true,
        }
    }

    /// Counters accumulated so far (the scheme copies them into its stats).
    pub(crate) fn events(&self) -> SpecializationEvents {
        self.events
    }

    /// Register (or re-register, after a caller-side pipeline swap) a dispatch site.
    ///
    /// `facts` are the node's tensor layout facts in ascending slot order (see
    /// [`tensor_fact_slot`]). Sites with neither scalar params nor facts, or whose shader
    /// has no single `[goldy_compute]` entry to scope bake macros to, are not tracked.
    pub(crate) fn register_site(
        &mut self,
        node: u32,
        universal: ComputePipelineHandle,
        provenance: &Arc<ShaderProvenance>,
        label: crate::SchemeLabel,
        slots: &[u32],
        facts: &[(u32, u32)],
    ) {
        if let Some(old) = self.sites.remove(&node) {
            self.retire_site(old);
        }
        if slots.is_empty() && facts.is_empty() {
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
            facts,
            &self.policy,
        );
        self.sites.insert(node, site);
        self.awake.insert(node);
    }

    /// The tensor layout facts `node` was registered with.
    pub(crate) fn tensor_facts(&self, node: u32) -> BakedSlots {
        self.sites.get(&node).map(|s| s.facts.clone()).unwrap_or_default()
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
                    site.facts.clone(),
                ))
            })
            .collect();
        for (idx, universal, provenance, label, slots, facts) in snapshot {
            self.register_site(idx, universal, &provenance, label, &slots, &facts);
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
        // Scheduled node indices are stale; step every site once to reschedule it.
        self.wake.clear();
        self.awake = self.sites.keys().copied().collect();
    }

    /// Whether `node` currently runs a predictor-chosen variant instead of the caller's pipeline.
    pub(crate) fn is_promoted(&self, node: u32) -> bool {
        self.sites.get(&node).is_some_and(SitePredictor::is_promoted)
    }

    /// The caller set scalar `slot` on `node` to `word`. Demote if the running variant
    /// baked another word.
    ///
    /// Returns the universal pipeline the scheme must rebind, if a demotion happened.
    pub(crate) fn on_param_changed(&mut self, node: u32, slot: usize, word: u32) -> Option<ComputePipelineHandle> {
        let policy = self.policy;
        let from = self.now + 1;
        let site = self.sites.get_mut(&node)?;
        if slot >= site.last.len() || !site.set_word(slot, word, from) {
            return None;
        }
        self.awake.insert(node);
        let slot_id = slot as u32;
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
    /// Steps only the sites with something to decide: new sites, sites whose words
    /// changed, sites with a compile in flight, and sites due by their schedule. A scheme
    /// whose sites have all settled does no work here. Returns `true` when a node's
    /// pipeline was rebound (the scheme must mark itself params-dirty).
    pub(crate) fn begin_submit(&mut self, device: &Runtime, ir: &mut GraphIR) -> bool {
        self.now += 1;
        if self.sites.is_empty() {
            return false;
        }
        if !self.enabled(device) {
            let rebound = self.was_enabled && self.disable_all(ir);
            self.was_enabled = false;
            return rebound;
        }
        if !self.was_enabled {
            self.was_enabled = true;
            self.wake.clear();
            self.awake = self.sites.keys().copied().collect();
        }
        let now = self.now;
        while let Some(&Reverse((at, node))) = self.wake.peek() {
            if at > now {
                break;
            }
            self.wake.pop();
            self.awake.insert(node);
        }
        if self.awake.is_empty() {
            return false;
        }
        let _tz = crate::tracy_zone!("specialization.step");
        let policy = self.policy;
        let mut rebound = false;
        self.inflight
            .retain(|j| j.upgrade().is_some_and(|j| j.holders.load(Ordering::Acquire) > 0));
        for node in std::mem::take(&mut self.awake) {
            let Some(site) = self.sites.get_mut(&node) else {
                continue;
            };
            let Some(NodeKind::Dispatch {
                pipeline, user_slots, ..
            }) = ir.nodes.get_mut(node as usize).map(|n| &mut n.kind)
            else {
                continue;
            };
            if user_slots.len() != site.last.len() {
                // Shape drift is not something the builder allows; be defensive anyway.
                continue;
            }
            site.submitted = true;
            let change = Self::step_site(
                site,
                user_slots,
                device,
                &self.variants,
                &mut self.retiring,
                &mut self.inflight,
                &mut self.events,
                &policy,
                node,
                now,
            );
            if let NodeChange::Bind(handle) = change {
                *pipeline = handle;
                rebound = true;
            }
            if let Some(at) = site.next_wake(now, &policy) {
                self.wake.push(Reverse((at.max(now + 1), node)));
            }
        }
        rebound
    }

    /// Release variants retired two submits ago.
    pub(crate) fn end_submit(&mut self) {
        self.retiring.pop_front();
        self.retiring.push_back(Vec::new());
    }

    /// Whether a site has a compile in flight or a compiled variant awaiting promotion.
    pub(crate) fn has_pending(&self) -> bool {
        self.sites.values().any(|s| s.job.is_some() || s.ready.is_some())
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

    /// Per slot of `node`: submits it has held its word, or `None` for a slot never changed.
    #[cfg(test)]
    pub(crate) fn site_held(&self, node: u32) -> Option<Vec<Option<u64>>> {
        self.sites.get(&node).map(|s| {
            (0..s.last.len())
                .map(|slot| s.changed[slot].then(|| s.held(slot, self.now)))
                .collect()
        })
    }

    /// Whether no site is awake or scheduled: submits do no predictor work.
    #[cfg(test)]
    pub(crate) fn is_idle(&self) -> bool {
        self.awake.is_empty() && self.wake.is_empty()
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
            site.restart(self.now + 1);
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
        now: u64,
    ) -> NodeChange {
        if site.pinned {
            return NodeChange::None;
        }

        // Words normally change through `on_param_changed`; catch any that moved another way.
        let mut demote = false;
        for (s, &word) in slots.iter().enumerate() {
            if !site.set_word(s, word, now) {
                continue;
            }
            let bakes = |baked: &BakedSlots| baked.iter().any(|&(b, _)| b == s as u32);
            let mut burned = false;
            if site.job.as_ref().is_some_and(|j| bakes(&j.baked)) {
                site.job.take();
                burned = true;
            }
            if site.ready.as_ref().is_some_and(|c| bakes(&c.baked)) {
                let c = site.ready.take().expect("checked");
                retiring.back_mut().expect("two generations").push(c.pipeline);
                burned = true;
            }
            if site.promoted.as_ref().is_some_and(|c| bakes(&c.baked)) {
                let c = site.promoted.take().expect("checked");
                retiring.back_mut().expect("two generations").push(c.pipeline);
                events.demotions += 1;
                demote = true;
                burned = true;
            }
            if burned {
                site.burn_slot(s, policy);
            }
        }
        if demote {
            return NodeChange::Bind(site.universal);
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
            .is_some_and(|c| site.all_baked_proven(&c.baked, now, policy.promote_after))
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
            let target = site.bake_target(now);
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

    fn two_scalar_site(facts: &[(u32, u32)], policy: &SpecializationPolicy) -> SitePredictor {
        let dev = crate::test_support::mock_runtime();
        let shader = ShaderModule::from_slang(
            &dev,
            "[goldy_compute]\n[numthreads(1,1,1)]\nvoid k(Scattered<uint> d, ThreadId id, uint a, uint b) { d[id.x] = a + b; }",
        )
        .unwrap();
        let pipeline = ComputePipeline::new(&dev, &shader).unwrap();
        SitePredictor::new(
            pipeline.handle,
            Arc::clone(&pipeline.provenance),
            "k".into(),
            "t".into(),
            &[7, 9],
            facts,
            policy,
        )
    }

    #[test]
    fn unchanged_slots_are_certain_and_changed_ones_earn_their_threshold() {
        let policy = SpecializationPolicy::default();
        let (warm, promote) = (u64::from(policy.warm_after), u64::from(policy.promote_after));
        let numel = tensor_fact_slot(1, 1);
        let mut site = two_scalar_site(&[(numel, 4096)], &policy);

        // Recorded words and layout facts bake at the first submit, already proven.
        let all = vec![(0, 7), (1, 9), (numel, 4096)];
        assert_eq!(site.bake_target(1), all);
        assert!(site.all_baked_proven(&all, 1, policy.promote_after));

        // A change before the first submit is still recording.
        assert!(site.set_word(1, 8, 1));
        assert!(site.is_certain(1));
        site.submitted = true;

        // After it, slot 1 is predicted from the submit that first runs its new word.
        assert!(site.set_word(1, 4, 5));
        assert!(!site.set_word(1, 4, 6), "same word is not a change");
        assert!(!site.is_certain(1));
        assert_eq!(site.bake_target(5), vec![(0, 7), (numel, 4096)]);
        assert_eq!(site.next_wake(5, &policy), Some(5 + warm));
        assert_eq!(site.bake_target(5 + warm), vec![(0, 7), (1, 4), (numel, 4096)]);
        let widened = [(0, 7), (1, 4)];
        assert!(!site.all_baked_proven(&widened, 5 + warm, policy.promote_after));
        assert!(site.all_baked_proven(&widened, 5 + promote, policy.promote_after));

        site.burn_slot(1, &policy);
        assert_eq!(site.bake_threshold[1], policy.promote_after);
        site.burn_slot(1, &policy);
        assert_eq!(site.bake_threshold[1], policy.promote_after * 2);
    }

    #[test]
    fn layout_facts_name_their_macros() {
        let policy = SpecializationPolicy::default();
        let numel = tensor_fact_slot(1, 1);
        let site = two_scalar_site(&[(tensor_fact_slot(0, 0), 1), (numel, 4096)], &policy);
        assert!(site.is_certain(numel));
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
