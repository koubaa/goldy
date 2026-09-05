//! Yielding scripts: petitions, handlers, and the runtime driver.
//!
//! A *yielding script* is a `[goldy_compute]` shader whose lanes may suspend at a
//! `$yield(continuation, payload, state)` and resume later in a `[goldy_resume]`
//! function with the result of a host- or GPU-side *handler*. The Slang side is
//! lowered by [`crate::slang::yielding`]; this module is the host side:
//!
//! - [`Petition`] ties a Rust `Pod` struct to its `[goldy_petition]` Slang struct.
//! - [`YieldPoint`] binds a handler and a mailbox capacity to one continuation.
//! - [`Promised`] is what a CPU handler fills in for each petition.
//! - [`Backpressure`] says what happens when more lanes yield than the mailbox holds.
//!
//! ```rust,ignore
//! #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
//! #[repr(C)]
//! struct Fetch { key: u32 }
//! impl Petition for Fetch {
//!     const SLANG_NAME: &'static str = "Fetch";
//!     type Result = u32;
//! }
//!
//! scheme
//!     .node("walk", &pipeline)
//!     .with_parcel(&data, NodeAccess::ReadWrite)
//!     .yield_point(
//!         "cs_resume",
//!         YieldPoint::cpu(1024, 4096, |p: &Fetch, promised: Promised<'_, u32>| {
//!             promised.fulfil(&table[p.key as usize]);
//!         }),
//!     )
//!     .dispatch(groups, 1, 1);
//! ```
//!
//! # Execution model
//!
//! The dispatch is recorded as a single host-driven node. On every submission the
//! driver launches the prologue, reads back how many lanes yielded to each
//! continuation, services those petitions (CPU handlers run on the submitting thread;
//! [`YieldPoint::node`] handlers run as a GPU dispatch), and resumes every serviced
//! lane in its continuation. Continuations may `$yield` again; the driver loops until
//! no lane is suspended. Each round is one sub-scheme submission on the same
//! [`crate::Context`], so a yielding node is a full pipeline drain — like
//! [`crate::Scheme::cpu_node`], it exists to make host round-trips expressible inside
//! a scheme, not to be cheap.
//!
//! Mailboxes are double-buffered so a continuation may yield to itself. Promised
//! results live in one *arena* buffer per yield point; a `Resolved<E>` is a window
//! into it, and a handler that runs out of arena space rejects the petition.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use bytemuck::Pod;

use crate::backend::shared::{MAX_BINDLESS_SLOTS, MAX_USER_SLOTS};
use crate::parcel::{Buffer, Parcel};
use crate::retained_pool::RetainedPool;
use crate::scheme::{PipelineParts, Scheme};
use crate::shader::YieldScript;
use crate::slang::yielding::{ContinuationDecl, YieldReflection};
use crate::task_graph::NodeAccess;
use crate::types::{BufferFlags, BufferKind};
use crate::{ComputePipeline, Context, GoldyError, MemoryExchange};

/// A Rust view of a `[goldy_petition]` payload struct.
///
/// The Rust layout must match the Slang struct field for field. v0 petition structs
/// hold only `uint` / `int` / `float` scalars and arrays of them, so `#[repr(C)]` with
/// `u32` / `i32` / `f32` fields in the same order always matches.
pub trait Petition: Pod + Send + Sync {
    /// Name of the Slang struct carrying the `[goldy_petition]` attribute.
    const SLANG_NAME: &'static str;
    /// Element type of the promised buffer (`Result = BufRO<E>` on the Slang side).
    type Result: Pod + Send + Sync;
}

/// One petition's pending result, handed to a CPU handler.
///
/// Call [`fulfil`](Self::fulfil) with the elements the continuation should see through
/// its `Resolved<E>`, or [`reject`](Self::reject) (or simply drop it) to resume the lane
/// with a null view.
pub struct Promised<'a, E: Pod> {
    arena: &'a mut Vec<E>,
    arena_cap: usize,
    resolution: &'a mut [u32; 2],
    overflow: &'a mut u64,
}

impl<E: Pod> Promised<'_, E> {
    /// Resolve the petition with `data`, copied into the yield point's result arena.
    ///
    /// When fewer than `data.len()` arena elements remain the petition is rejected
    /// instead and counted in [`YieldStats::arena_overflow`].
    pub fn fulfil(self, data: &[E]) {
        if self.arena.len() + data.len() > self.arena_cap {
            *self.overflow += 1;
            return;
        }
        let off = self.arena.len();
        self.arena.extend_from_slice(data);
        *self.resolution = [off as u32, data.len() as u32];
    }

    /// Resume the lane with a null `Resolved<E>`.
    pub fn reject(self) {}

    /// Arena elements still available to this and later petitions of the batch.
    pub fn arena_remaining(&self) -> usize {
        self.arena_cap - self.arena.len()
    }
}

/// What a yield point does when more lanes yield than its mailbox capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backpressure {
    /// Never lose a lane: the prologue is launched in chunks of at most `capacity`
    /// lanes and each chunk is drained before the next starts. Requires the prologue
    /// to take a `ThreadId` and not `GroupId` / `GroupThreadId` when the dispatch is
    /// wider than one chunk, and `capacity >= numthreads.x`.
    #[default]
    Stall,
    /// Lanes that find the mailbox full are dropped: their continuation never runs.
    /// Counted in [`YieldStats::dropped`].
    Drop,
}

/// Counters from the most recent submission of a yielding node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct YieldStats {
    /// Prologue chunks launched (1 unless `Backpressure::Stall` split the dispatch).
    pub chunks: u32,
    /// Resume rounds (sub-scheme submissions after the prologue).
    pub rounds: u32,
    /// Petitions serviced by handlers.
    pub petitions: u64,
    /// Lanes resumed in a continuation.
    pub resumed: u64,
    /// Petitions a CPU handler rejected (including arena overflows).
    pub rejected: u64,
    /// Petitions that did not fit a `Backpressure::Drop` mailbox.
    pub dropped: u64,
    /// Fulfilments refused because the result arena was full.
    pub arena_overflow: u64,
}

/// Results of one CPU service batch, in the wire form the continuation reads.
struct ServiceBatch {
    resolutions: Vec<[u32; 2]>,
    arena_bytes: Vec<u8>,
    rejected: u64,
    overflow: u64,
}

type ErasedCpuHandler = Arc<dyn Fn(&[u8], u32, u32) -> ServiceBatch + Send + Sync>;

enum Handler {
    Cpu {
        petition: &'static str,
        payload_bytes: usize,
        result_bytes: usize,
        run: ErasedCpuHandler,
    },
    Node {
        pipeline: PipelineParts,
        numthreads_x: u32,
        extra: Vec<(Parcel, NodeAccess)>,
    },
}

/// The handler, capacity, and policy bound to one continuation of a yielding script.
///
/// See [`crate::SchemeNodeBuilder::yield_point`].
pub struct YieldPoint {
    capacity: u32,
    arena_len: u32,
    backpressure: Backpressure,
    handler: Handler,
}

impl YieldPoint {
    /// Service petitions on the host.
    ///
    /// `capacity` is the mailbox size (lanes that may be suspended at this continuation
    /// at once); `arena_len` is the number of `P::Result` elements all fulfilments of one
    /// round may use together. `handler` is called once per petition, in mailbox order,
    /// on the thread that submits the scheme.
    pub fn cpu<P, F>(capacity: u32, arena_len: u32, handler: F) -> Self
    where
        P: Petition,
        F: Fn(&P, Promised<'_, P::Result>) + Send + Sync + 'static,
    {
        let run: ErasedCpuHandler = Arc::new(move |payloads: &[u8], count: u32, arena_cap: u32| {
            let petitions: &[P] = bytemuck::cast_slice(payloads);
            let mut resolutions = vec![[crate::slang::yielding::RESOLVED_NULL, 0u32]; count as usize];
            let mut arena: Vec<P::Result> = Vec::new();
            let mut overflow = 0u64;
            for (p, resolution) in petitions.iter().zip(resolutions.iter_mut()) {
                handler(
                    p,
                    Promised {
                        arena: &mut arena,
                        arena_cap: arena_cap as usize,
                        resolution,
                        overflow: &mut overflow,
                    },
                );
            }
            let rejected = resolutions
                .iter()
                .filter(|r| r[0] == crate::slang::yielding::RESOLVED_NULL)
                .count() as u64;
            ServiceBatch {
                resolutions,
                arena_bytes: bytemuck::cast_slice(&arena).to_vec(),
                rejected,
                overflow,
            }
        });
        Self {
            capacity,
            arena_len,
            backpressure: Backpressure::default(),
            handler: Handler::Cpu {
                petition: P::SLANG_NAME,
                payload_bytes: std::mem::size_of::<P>(),
                result_bytes: std::mem::size_of::<P::Result>(),
                run,
            },
        }
    }

    /// Service petitions with a compute dispatch.
    ///
    /// The handler shader's `[goldy_compute]` entry must take, in order,
    /// `Scattered<P> petitions` (or `BufRO<P>`), `Scattered<Resolution> resolutions`,
    /// `Scattered<E> arena`, any parcels added with [`Self::with_parcel`], then
    /// `uint count` and a `ThreadId`. It is dispatched with `count` lanes along x and
    /// must write one resolution per petition (`goldy_resolve` / `goldy_reject`).
    ///
    /// As with [`crate::Scheme::node`], the caller keeps `pipeline` alive while the
    /// scheme is in use.
    pub fn node(capacity: u32, arena_len: u32, pipeline: &ComputePipeline) -> Self {
        let numthreads_x = crate::slang::virtual_main::find_all_entries(&pipeline.provenance.source)
            .into_iter()
            .find(|e| e.stage == crate::slang::virtual_main::Stage::Compute)
            .and_then(|e| e.numthreads)
            .map(|(x, _, _)| x)
            .unwrap_or(64);
        Self {
            capacity,
            arena_len,
            backpressure: Backpressure::default(),
            handler: Handler::Node {
                pipeline: PipelineParts::of(pipeline),
                numthreads_x,
                extra: Vec::new(),
            },
        }
    }

    /// Bind an extra buffer parcel to a [`Self::node`] handler, after the three
    /// mailbox parameters. No effect on CPU handlers.
    pub fn with_parcel(mut self, parcel: &Parcel, access: NodeAccess) -> Self {
        if let Handler::Node { extra, .. } = &mut self.handler {
            extra.push((parcel.clone(), access));
        }
        self
    }

    /// Set the overflow policy (default [`Backpressure::Stall`]).
    pub fn backpressure(mut self, policy: Backpressure) -> Self {
        self.backpressure = policy;
        self
    }
}

/// Continuation pipelines compiled alongside a yielding script's prologue pipeline.
pub(crate) struct YieldPipelines {
    pub(crate) script: Arc<YieldScript>,
    /// One per `reflection.continuations`, same order.
    pub(crate) continuations: Vec<Arc<ComputePipeline>>,
}

impl YieldPipelines {
    fn reflection(&self) -> &YieldReflection {
        &self.script.reflection
    }
}

/// Everything the driver needs about one continuation.
struct PointState {
    cap: u32,
    arena_len: u32,
    backpressure: Backpressure,
    handler: Handler,
    /// Double-buffered mailboxes: `[set]` is read while `[1 - set]` is written.
    pay: [Buffer; 2],
    st: [Buffer; 2],
    res: Buffer,
    arena: Buffer,
    numthreads_x: u32,
    /// Indices into the user's resource bindings, in the continuation's parameter order.
    program_resources: Vec<usize>,
    /// Indices into the user's scalar params, in the continuation's parameter order.
    program_scalars: Vec<usize>,
    /// Continuation indices this body may yield to.
    yields_to: Vec<usize>,
}

/// Host-driven executor recorded in place of a yielding dispatch.
pub(crate) struct YieldDriver {
    label: &'static str,
    ctx: Context,
    pipelines: Arc<YieldPipelines>,
    prologue: PipelineParts,
    user_parcels: Vec<(Parcel, NodeAccess)>,
    user_scalars: Vec<u32>,
    dispatch: (u32, u32, u32),
    /// `Some(groups)` when the prologue is launched in chunks of that many workgroups.
    chunk_groups: Option<u32>,
    prologue_yields_to: Vec<usize>,
    points: Vec<PointState>,
    cnt: Buffer,
    _pool: RetainedPool,
    stats: Arc<Mutex<YieldStats>>,
}

/// Record-time inputs collected by [`crate::SchemeNodeBuilder`].
pub(crate) struct YieldRecord {
    pub label: &'static str,
    pub pipelines: Arc<YieldPipelines>,
    pub prologue: PipelineParts,
    /// One entry per `with_parcel` call; `None` when the bindable was not a buffer parcel.
    pub parcels: Vec<Option<(Parcel, NodeAccess)>>,
    pub scalars: Vec<u32>,
    pub points: Vec<(String, YieldPoint)>,
    pub dispatch: (u32, u32, u32),
}

fn err<T>(msg: impl Into<String>) -> Result<T, String> {
    Err(format!("yield_point: {}", msg.into()))
}

fn scalar_result_bytes(elem: &str) -> Option<usize> {
    match elem {
        "uint" | "int" | "float" => Some(4),
        _ => None,
    }
}

impl YieldDriver {
    /// Validate a recorded yielding dispatch against its script and allocate its mailboxes.
    pub(crate) fn build(ctx: &Context, record: YieldRecord) -> Result<(Self, Arc<Mutex<YieldStats>>), String> {
        let YieldRecord {
            label,
            pipelines,
            prologue,
            parcels,
            scalars,
            points,
            dispatch,
        } = record;
        let refl = pipelines.reflection();

        // User bindings must match the prologue's program parameters.
        let prologue_resources: Vec<&crate::slang::yielding::ProgramParam> =
            refl.prologue_params.iter().filter(|p| !p.is_scalar).collect();
        let prologue_scalars = refl.prologue_params.len() - prologue_resources.len();
        if parcels.len() != prologue_resources.len() {
            return err(format!(
                "`{label}` binds {} parcel(s) but the prologue declares {} buffer parameter(s)",
                parcels.len(),
                prologue_resources.len()
            ));
        }
        if scalars.len() != prologue_scalars {
            return err(format!(
                "`{label}` passes {} scalar param(s) but the prologue declares {prologue_scalars}",
                scalars.len()
            ));
        }
        let mut user_parcels = Vec::with_capacity(parcels.len());
        for (i, p) in parcels.into_iter().enumerate() {
            match p {
                Some(p) => user_parcels.push(p),
                None => {
                    return err(format!(
                        "`{label}` parameter `{}`: yielding scripts bind buffer parcels only (Parcel / Buffer)",
                        prologue_resources[i].name
                    ))
                }
            }
        }

        // Every continuation needs exactly one yield point, and vice versa.
        let mut bound: Vec<Option<YieldPoint>> = (0..refl.continuations.len()).map(|_| None).collect();
        for (name, yp) in points {
            let Some(idx) = refl.continuation_index(&name) else {
                let known: Vec<&str> = refl.continuations.iter().map(|c| c.fn_name.as_str()).collect();
                return err(format!(
                    "`{label}` has no [goldy_resume] continuation named `{name}` (continuations: {known:?})"
                ));
            };
            if bound[idx].is_some() {
                return err(format!("`{label}` binds continuation `{name}` twice"));
            }
            bound[idx] = Some(yp);
        }
        if let Some(missing) = refl
            .continuations
            .iter()
            .zip(&bound)
            .find_map(|(c, b)| b.is_none().then_some(&c.fn_name))
        {
            return err(format!(
                "`{label}` yields to `{missing}` but no yield_point(\"{missing}\", ..) was bound"
            ));
        }

        let (nx, ny, nz) = refl.prologue_numthreads;
        let device = ctx.device().clone();
        let mut pool = RetainedPool::new(Arc::new(device));
        let alloc = |pool: &mut RetainedPool, what: &str, elems: u64, stride: u32| -> Result<Buffer, String> {
            pool.acquire_buffer(
                elems.max(1) * stride as u64,
                BufferKind::Scattered,
                Some(stride),
                BufferFlags::empty(),
                None,
            )
            .map_err(|e| format!("yield_point: allocating {what} for `{label}`: {e}"))
        };

        let mut states = Vec::with_capacity(refl.continuations.len());
        let mut stall_cap: Option<u32> = None;
        for (c, yp) in refl.continuations.iter().zip(bound) {
            let yp = yp.expect("checked above");
            let name = &c.fn_name;
            if yp.capacity == 0 {
                return err(format!("`{name}`: capacity must be at least 1"));
            }
            if c.numthreads.1 != 1 || c.numthreads.2 != 1 {
                return err(format!(
                    "`{name}`: continuations must use [numthreads(N, 1, 1)], got {:?}",
                    c.numthreads
                ));
            }
            let petition = refl.petition(&c.petition).expect("reflection validated petitions");
            let result_bytes = match &yp.handler {
                Handler::Cpu {
                    petition: rust_name,
                    payload_bytes,
                    result_bytes,
                    ..
                } => {
                    if *rust_name != c.petition {
                        return err(format!(
                            "`{name}` is resumed for petition `{}` but the handler's Petition::SLANG_NAME is `{rust_name}`",
                            c.petition
                        ));
                    }
                    if *payload_bytes != petition.payload_bytes as usize {
                        return err(format!(
                            "`{name}`: Rust petition `{rust_name}` is {payload_bytes} bytes but the Slang struct is {} bytes",
                            petition.payload_bytes
                        ));
                    }
                    if let Some(expect) = scalar_result_bytes(&c.result_elem) {
                        if *result_bytes != expect {
                            return err(format!(
                                "`{name}`: Petition::Result is {result_bytes} bytes but the Slang result element `{}` is {expect} bytes",
                                c.result_elem
                            ));
                        }
                    }
                    *result_bytes as u32
                }
                Handler::Node { .. } => scalar_result_bytes(&c.result_elem).unwrap_or(4) as u32,
            };
            if yp.backpressure == Backpressure::Stall {
                if yp.capacity < nx {
                    return err(format!(
                        "`{name}`: Backpressure::Stall needs capacity >= prologue numthreads.x ({nx}), got {}",
                        yp.capacity
                    ));
                }
                stall_cap = Some(stall_cap.map_or(yp.capacity, |c| c.min(yp.capacity)));
            }

            let mut program_resources = Vec::new();
            let mut program_scalars = Vec::new();
            for p in &c.program_params {
                let pos = refl
                    .prologue_params
                    .iter()
                    .position(|q| q.name == p.name)
                    .expect("reflection validated names");
                let kind_index = refl.prologue_params[..pos]
                    .iter()
                    .filter(|q| q.is_scalar == p.is_scalar)
                    .count();
                if p.is_scalar {
                    program_scalars.push(kind_index);
                } else {
                    program_resources.push(kind_index);
                }
            }
            let yields_to: Vec<usize> = c
                .yields_to
                .iter()
                .map(|t| refl.continuation_index(t).expect("validated"))
                .collect();

            // Slot budget of the continuation dispatch (see `continuation_entry`).
            let extra_slots = if yields_to.is_empty() { 0 } else { 1 };
            let resources = program_resources.len() + 4 + 2 * yields_to.len() + extra_slots;
            if resources > MAX_BINDLESS_SLOTS {
                return err(format!(
                    "`{name}` needs {resources} resource slots (max {MAX_BINDLESS_SLOTS}); bind fewer buffers"
                ));
            }
            let user_words = program_scalars.len() + 1 + yields_to.len();
            if user_words > MAX_USER_SLOTS {
                return err(format!(
                    "`{name}` needs {user_words} scalar params (max {MAX_USER_SLOTS})"
                ));
            }

            let cap = yp.capacity as u64;
            let pay = [
                alloc(&mut pool, "payload mailbox", cap, petition.payload_bytes)?,
                alloc(&mut pool, "payload mailbox", cap, petition.payload_bytes)?,
            ];
            let st = [
                alloc(&mut pool, "state mailbox", cap, c.state_bytes)?,
                alloc(&mut pool, "state mailbox", cap, c.state_bytes)?,
            ];
            let res = alloc(&mut pool, "resolution table", cap, 8)?;
            let arena = alloc(&mut pool, "result arena", yp.arena_len as u64, result_bytes)?;
            states.push(PointState {
                cap: yp.capacity,
                arena_len: yp.arena_len,
                backpressure: yp.backpressure,
                handler: yp.handler,
                pay,
                st,
                res,
                arena,
                numthreads_x: c.numthreads.0,
                program_resources,
                program_scalars,
                yields_to,
            });
        }

        // Prologue slot budget.
        let prologue_yields_to: Vec<usize> = refl
            .prologue_yields_to
            .iter()
            .map(|t| refl.continuation_index(t).expect("validated"))
            .collect();
        let extra_slots = if prologue_yields_to.is_empty() { 0 } else { 1 };
        let resources = user_parcels.len() + 2 * prologue_yields_to.len() + extra_slots;
        if resources > MAX_BINDLESS_SLOTS {
            return err(format!(
                "`{label}` prologue needs {resources} resource slots (max {MAX_BINDLESS_SLOTS}); bind fewer buffers"
            ));
        }
        let user_words = scalars.len() + prologue_yields_to.len() + 1;
        if user_words > MAX_USER_SLOTS {
            return err(format!(
                "`{label}` prologue needs {user_words} scalar params (max {MAX_USER_SLOTS})"
            ));
        }

        // Chunking for Stall: the whole live-lane population never exceeds the smallest
        // Stall capacity when each chunk launches at most that many lanes.
        let chunk_groups = match stall_cap {
            Some(cap) if cap / nx < dispatch.0 => {
                if !refl.prologue_has_thread_id {
                    return err(format!(
                        "`{label}`: Backpressure::Stall must launch {} workgroups in chunks of {}, which requires a \
                         `ThreadId` parameter on the prologue",
                        dispatch.0,
                        cap / nx
                    ));
                }
                if refl.prologue_uses_group_ids {
                    return err(format!(
                        "`{label}`: Backpressure::Stall chunks the prologue, which breaks GroupId / GroupThreadId; \
                         use ThreadId or raise the capacity to {}",
                        dispatch.0 * nx
                    ));
                }
                if dispatch.1 != 1 || dispatch.2 != 1 || ny != 1 || nz != 1 {
                    return err(format!(
                        "`{label}`: Backpressure::Stall chunks along x only; dispatch ({}, {}, {}) with numthreads \
                         ({nx}, {ny}, {nz}) must be one-dimensional or fit one chunk",
                        dispatch.0, dispatch.1, dispatch.2
                    ));
                }
                Some(cap / nx)
            }
            _ => None,
        };

        let cnt = alloc(&mut pool, "yield counters", refl.continuations.len() as u64, 4)?;
        let stats = Arc::new(Mutex::new(YieldStats::default()));
        Ok((
            Self {
                label,
                ctx: ctx.clone(),
                pipelines,
                prologue,
                user_parcels,
                user_scalars: scalars,
                dispatch,
                chunk_groups,
                prologue_yields_to,
                points: states,
                cnt,
                _pool: pool,
                stats: Arc::clone(&stats),
            },
            stats,
        ))
    }

    fn continuation(&self, idx: usize) -> &ContinuationDecl {
        &self.pipelines.reflection().continuations[idx]
    }

    /// Run one submission's worth of yield/service/resume rounds.
    pub(crate) fn run(&self) -> Result<(), GoldyError> {
        let _tz = crate::tracy_zone!("goldy.yield.run");
        let mut stats = YieldStats::default();
        let (gx, gy, gz) = self.dispatch;
        let nx = self.pipelines.reflection().prologue_numthreads.0;
        let chunks: Vec<(u32, u32)> = match self.chunk_groups {
            None => vec![(0, gx)],
            Some(cg) => (0..gx).step_by(cg as usize).map(|g| (g * nx, cg.min(gx - g))).collect(),
        };

        for (base, groups) in chunks {
            stats.chunks += 1;
            let mut set = 0usize;
            let (mut counts, mut payloads) = self.launch_prologue(base, groups, gy, gz, set)?;
            loop {
                let mut pending: Vec<(usize, u32)> = Vec::new();
                for (i, &n) in counts.iter().enumerate() {
                    if n == 0 {
                        continue;
                    }
                    let pt = &self.points[i];
                    if n > pt.cap {
                        // `Stall` never gets here (chunking bounds the live population);
                        // `Drop` lanes past the capacity wrote nothing.
                        debug_assert_eq!(pt.backpressure, Backpressure::Drop);
                        stats.dropped += u64::from(n - pt.cap);
                        pending.push((i, pt.cap));
                    } else {
                        pending.push((i, n));
                    }
                }
                if pending.is_empty() {
                    break;
                }
                stats.rounds += 1;
                let next = 1 - set;
                (counts, payloads) = self.resume_round(&pending, &payloads, set, next, &mut stats)?;
                set = next;
            }
        }

        *self.stats.lock().unwrap() = stats;
        Ok(())
    }

    /// Submit the prologue over `groups` workgroups starting at lane `base`; return the
    /// per-continuation yield counts and the payload bytes of every CPU-handled mailbox.
    fn launch_prologue(
        &self,
        base: u32,
        groups: u32,
        gy: u32,
        gz: u32,
        set: usize,
    ) -> Result<RoundOutcome, GoldyError> {
        let mut sub = Scheme::new(&self.ctx);
        let mx = MemoryExchange::new(&self.ctx);
        sub.clear_parcel(self.cnt.whole(), 0, 0)?;
        let mut node = sub.node_from_parts(self.label, &self.prologue);
        for (p, access) in &self.user_parcels {
            node = node.with_parcel(p, *access);
        }
        for &t in &self.prologue_yields_to {
            node = node
                .with_parcel(self.points[t].pay[set].whole(), NodeAccess::Write)
                .with_parcel(self.points[t].st[set].whole(), NodeAccess::Write);
        }
        if !self.prologue_yields_to.is_empty() {
            node = node.with_parcel(self.cnt.whole(), NodeAccess::ReadWrite);
        }
        for &s in &self.user_scalars {
            node = node.with_param(s);
        }
        for &t in &self.prologue_yields_to {
            node = node.with_param(self.points[t].cap);
        }
        node = node.with_param(base);
        node.dispatch(groups, gy, gz);
        let withdraws = self.bind_withdraws(&mut sub, &mx, &self.prologue_yields_to, set)?;
        let mut submission = sub.submit()?;
        withdraws.claim(&mut submission, self.points.len())
    }

    /// Service `pending` continuations from mailbox set `set` and resume them, writing
    /// re-yields into set `next`.
    fn resume_round(
        &self,
        pending: &[(usize, u32)],
        payloads: &[Option<Vec<u8>>],
        set: usize,
        next: usize,
        stats: &mut YieldStats,
    ) -> Result<RoundOutcome, GoldyError> {
        let mut sub = Scheme::new(&self.ctx);
        let mx = MemoryExchange::new(&self.ctx);

        for &(i, n) in pending {
            stats.petitions += u64::from(n);
            let pt = &self.points[i];
            match &pt.handler {
                Handler::Cpu { payload_bytes, run, .. } => {
                    let bytes = payloads[i].as_deref().ok_or_else(|| {
                        GoldyError::Backend(anyhow::anyhow!(
                            "yield `{}`: payload mailbox was not withdrawn",
                            self.continuation(i).fn_name
                        ))
                    })?;
                    let batch = {
                        let _tz = crate::tracy_zone!("goldy.yield.cpu_handler");
                        run(&bytes[..n as usize * payload_bytes], n, pt.arena_len)
                    };
                    stats.rejected += batch.rejected;
                    stats.arena_overflow += batch.overflow;
                    let res_bytes: &[u8] = bytemuck::cast_slice(&batch.resolutions);
                    mx.bind_deposit_buffer(&mut sub, pt.res.whole(), res_bytes.len() as u64)?
                        .write_bytes(&mut sub, res_bytes)?;
                    if !batch.arena_bytes.is_empty() {
                        mx.bind_deposit_buffer(&mut sub, pt.arena.whole(), batch.arena_bytes.len() as u64)?
                            .write_bytes(&mut sub, &batch.arena_bytes)?;
                    }
                }
                Handler::Node {
                    pipeline,
                    numthreads_x,
                    extra,
                } => {
                    let mut node = sub
                        .node_from_parts("yield_handler", pipeline)
                        .with_parcel(pt.pay[set].whole(), NodeAccess::Read)
                        .with_parcel(pt.res.whole(), NodeAccess::Write)
                        .with_parcel(pt.arena.whole(), NodeAccess::Write);
                    for (p, access) in extra {
                        node = node.with_parcel(p, *access);
                    }
                    node.with_param(n).dispatch(n.div_ceil(*numthreads_x), 1, 1);
                }
            }
        }

        sub.clear_parcel(self.cnt.whole(), 0, 0)?;
        let mut targets = BTreeSet::new();
        for &(i, n) in pending {
            let pt = &self.points[i];
            let mut node = sub.node("yield_resume", &self.pipelines.continuations[i]);
            for &r in &pt.program_resources {
                let (p, access) = &self.user_parcels[r];
                node = node.with_parcel(p, *access);
            }
            node = node
                .with_parcel(pt.pay[set].whole(), NodeAccess::Read)
                .with_parcel(pt.st[set].whole(), NodeAccess::Read)
                .with_parcel(pt.res.whole(), NodeAccess::Read)
                .with_parcel(pt.arena.whole(), NodeAccess::Read);
            for &t in &pt.yields_to {
                targets.insert(t);
                node = node
                    .with_parcel(self.points[t].pay[next].whole(), NodeAccess::Write)
                    .with_parcel(self.points[t].st[next].whole(), NodeAccess::Write);
            }
            if !pt.yields_to.is_empty() {
                node = node.with_parcel(self.cnt.whole(), NodeAccess::ReadWrite);
            }
            for &s in &pt.program_scalars {
                node = node.with_param(self.user_scalars[s]);
            }
            node = node.with_param(n);
            for &t in &pt.yields_to {
                node = node.with_param(self.points[t].cap);
            }
            node.dispatch(n.div_ceil(pt.numthreads_x), 1, 1);
            stats.resumed += u64::from(n);
        }

        let targets: Vec<usize> = targets.into_iter().collect();
        let withdraws = self.bind_withdraws(&mut sub, &mx, &targets, next)?;
        let mut submission = sub.submit()?;
        withdraws.claim(&mut submission, self.points.len())
    }

    /// Withdraw the counters and, for CPU-handled `targets`, their payload mailboxes in `set`.
    fn bind_withdraws(
        &self,
        sub: &mut Scheme,
        mx: &MemoryExchange,
        targets: &[usize],
        set: usize,
    ) -> Result<RoundWithdraws, GoldyError> {
        let counts = mx.bind_withdraw(sub, self.cnt.whole())?;
        let mut payloads = Vec::new();
        for &t in targets {
            if matches!(self.points[t].handler, Handler::Cpu { .. }) {
                payloads.push((t, mx.bind_withdraw(sub, self.points[t].pay[set].whole())?));
            }
        }
        Ok(RoundWithdraws { counts, payloads })
    }
}

/// Per-continuation yield counts and, for CPU-handled mailboxes, the withdrawn payload bytes.
type RoundOutcome = (Vec<u32>, Vec<Option<Vec<u8>>>);

struct RoundWithdraws {
    counts: crate::WithdrawTransaction,
    payloads: Vec<(usize, crate::WithdrawTransaction)>,
}

impl RoundWithdraws {
    fn claim(self, submission: &mut crate::Submission, n_points: usize) -> Result<RoundOutcome, GoldyError> {
        let bytes = self.counts.claim(submission)?.consume()?;
        let counts: Vec<u32> = bytemuck::cast_slice(&bytes[..n_points * 4]).to_vec();
        drop(bytes);
        let mut payloads: Vec<Option<Vec<u8>>> = (0..n_points).map(|_| None).collect();
        for (t, w) in self.payloads {
            payloads[t] = Some(w.claim(submission)?.consume()?.into_vec());
        }
        Ok((counts, payloads))
    }
}

/// Type-erased driver entry used by the CPU dispatch machinery.
pub(crate) fn driver_main(
    driver: Arc<YieldDriver>,
) -> impl Fn(&Context) -> Result<(), GoldyError> + Send + Sync + 'static {
    move |_ctx| driver.run()
}
