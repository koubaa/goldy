# Schemes

A **scheme** is Goldy's retained unit of work: a recorded graph of dispatches and precedences you submit again without re-recording while it stays clean. Create one with [`Scheme::new`](https://docs.rs/goldy/latest/goldy/struct.Scheme.html), bind parcels on nodes, and call `submit` every frame. Compute pipelines and record-time constant buffers are interned on the scheme, so you can drop the objects you used only to record.

```rust
let mut scheme = Scheme::new(&ctx);
scheme
    .node("double", &pipeline)
    .with_parcel(&data, NodeAccess::ReadWrite)
    .dispatch(1, 1, 1);
scheme.submit()?;
```

Structural mutation (new nodes, new bindings, `include`) drops retained command lists. Params-only mutation (pipeline, scalars, dispatch dims) re-records only the partitions whose baked payload changed. A clean scheme resubmits with neither recording nor fingerprint hashing.

## Nesting

`Scheme::include` copies a child's **description** into the parent as one group. The copy is a snapshot: mutating the child afterward does not change the parent, and the child remains independently submittable.

```rust
let mut attn = Scheme::new(&ctx);
attn.node("rmsnorm", &rmsnorm).with_parcel(&x, NodeAccess::ReadWrite).dispatch(groups, 1, 1);
// ... more child nodes ...

let mut layer = Scheme::new(&ctx);
layer.include("attn", &attn)?.finish();
layer.submit()?;
attn.submit()?; // still legal
```

`Scheme::group` is sugar for a temporary child on the same context plus include:

```rust
layer.group("ffn", |s| {
    s.node("up", &up).with_parcel(&x, NodeAccess::ReadWrite).dispatch(groups, 1, 1);
    Ok(())
})?;
```

`GroupBuilder::after` adds a group-level precedence (`A` completes before `B`). Record order is the total order: only forward precedences are admitted. Expansion to node pairs happens when the schedule cache is rebuilt, not on the clean submit path.

Included nodes keep group provenance. Backend markers and validation text show the path (`layer0/attn/rmsnorm`).

## Include restrictions

Anything not proven correct is rejected at include time (`GoldyError::Validation`, or `StaleResource` if a child stamp is dead). The parent is left untouched. v1 admits only:

- The same `Context` (cross-context include is rejected)
- No pending record errors on the child
- No `cpu_node` (closures and per-occurrence staging are instance state)
- No deposits, present leases, or swapchain outputs (exchanges belong to the submitting root)
- No yielding dispatches
- No transient parcels (lease-epoch semantics across two submitters are not yet admitted)

Interleaving parent and child submits on shared parcels is correct: the ledger orders them like any two schemes. The parent will `topology_dirty` and re-record barriers — correct, not free.
