# Machine Specification

**Status**: Draft v0.12

This chapter specifies the **Fondaco abstract machine**: what a merchant is, what parcels and ownership mean, what execution is, and what a runtime must do. It does **not** specify a particular hardware mapping, calling convention, or API. Those belong to implementations — Goldy's mapping is in [Goldy Runtime Mapping](./goldy-runtime.md).

The machine is a *positive* specification: it states what *is*, and what is not stated does not exist. Analogies to other abstract machines appear only in the appendix and are leaky. The Fondaco terms in the body are authoritative. See also [Terminology](./terminology.md).

## 1. The machine

The Fondaco machine is an abstract machine for cooperative computation, implemented by a **runtime**.

**Parcels** are data. **Schemes** are computation.

A **merchant** describes parcels and executes schemes via **dispatches**. The merchant is the **sovereign** — every parcel belongs to it; the runtime holds them in trust. The merchant assigns ownership claims through schemes; the runtime manages their physical realization.

The remainder of this chapter specifies schemes, dispatches, scripts (including yielding), parcels and ownership, exchanges, gates, petitions, and the latitude runtimes have to transform schemes.

## 2. The scheme

A **scheme** consists of:

- A set of **dispatches**
- A set of **precedences** between dispatches

A precedence `A → B` asserts that dispatch `A` must complete before dispatch `B` begins. The transitive closure is a partial order over the dispatches.

Two dispatches are **unordered** if neither is downstream of the other. The runtime may execute unordered dispatches in any order, including concurrently or fused.

A scheme is first-class data. Schemes compose: a scheme may contain another as a sub-scheme, and the composition of two schemes is a scheme.

A scheme is **well-formed** if its precedence set is consistent with its ownership claims (§5). Specifically: no two unordered dispatches may claim the same parcel where at least one holds private ownership. A merchant may add precedences beyond those ownership requires — for throughput or occupancy — but may not omit a required precedence.

Presenting an ill-formed scheme has *unspecified behavior*. Conforming runtimes may refuse to admit it (§10). Goldy refuses ill-formed schemes rather than producing unspecified results.

The precedence set need not be acyclic. A cyclic scheme describes a non-halting computation; whether to admit it is left to the runtime.

## 3. Dispatches

A **dispatch** is the atomic unit of work admitted to the machine. Once begun, it runs to completion. It is not preempted or reordered internally.

A dispatch runs over parcels. If arbitrary computation is required, the runtime schedules a **script** (the execution model is left unspecified); otherwise the runtime performs the dispatch directly.

## 4. Scripts and yielding

A **script** is the procedure a computing dispatch evaluates. The language is unspecified by the machine; runtimes may fix one or more. The script is opaque to the runtime: control flow is not visible. A script may keep private resources that are invisible to the runtime during a dispatch.

### Yielding

A script may **yield** if it contains **yield points** that transfer control to the runtime and later resume.

At a yield point:

1. The dispatch reaches the yield point and **petitions** the runtime (§7).
2. The script's private resources are undisturbed while suspended.
3. The runtime may modify claims while suspended, as long as it restores the state the dispatch observed before the yield.
4. The runtime unsuspends the dispatch; the script resumes from the yield point.

A yield point does **not** create a gate. The dispatch has not ended. Visibility is limited to servicing that petition and scheduling — not the full gate powers of §6.

### Non-yielding scripts

A script with no yield points runs start to finish with no runtime intervention. The runtime is invisible for the duration of a non-yielding script.

Goldy today ships only non-yielding scripts; yielding is **Designed** (see [Goldy Runtime Mapping](./goldy-runtime.md)).

## 5. Parcels and ownership

A **parcel** is a stable identity for data held by the runtime in trust for the merchant. Parcels are the sole channel of communication between dispatches within a scheme or across schemes. Dispatches are not aware of physical realization; they rely on stable identity.

Each parcel has:

- A **type**, fixed at creation (type system unspecified by the machine)
- A **size** — extent or shape, possibly hinted, possibly fixed
- A **claim** — ownership granted by the merchant

### Ownership

**Ownership** is expressed as **claims**: access rights for the duration of a dispatch.

- **Public** — read. Multiple dispatches may hold public ownership of the same parcel concurrently.
- **Private** — read and write. Exclusive: no other concurrent claim on that parcel.
- **Private-inaugural** — write without depending on prior contents. Exclusive, but no precedence from a prior owner is required; prior state is abandoned.

### Ownership transfer

If dispatch A holds a private claim on parcel X and dispatch B (ordered after A) claims X, ownership transfers at the gate between them. If no later dispatch claims a parcel after its last holder completes, claims drop and ownership reverts to the runtime, which may destroy the parcel if the merchant no longer needs it.

### The ledger

The **ledger** is the runtime's standing record of claims over parcels. Well-formedness (§2) is a property of one scheme in isolation; the ledger is the aggregate account *between* schemes. When one scheme completes and another is admitted, later claims serialize against prior owners the ledger records.

At every instant, at most one private claim — or any number of public claims — stands over a given parcel. The runtime mutates the ledger only by admitting schemes, acting at gates (§6), and settling exchanges. Pending foreign access from an exchange is a standing constraint until that access expires.

The ledger is an invariant, invisible to the merchant. The machine does not require a particular data structure — only that the runtime behave as though such a record is conserved.

### Exchanges

An **exchange** is a stable, runtime-mediated relationship between a scheme and an entity outside the machine. Through an exchange, a scheme may periodically hand parcels to, or receive parcels from, a foreign subsystem.

Establishing an exchange records the relationship and its ownership constraints; it does not itself perform a foreign operation. Executions may produce **exchange claims**, distinct from ownership claims.

Settlement is either:

- **Consume** — perform the foreign operation defined by the exchange
- **Discard** — settle without exercising it

Settlement is terminal even if it reports failure. A runtime must define safe settlement for claims abandoned by the merchant. Representation and delivery of claims are defined by the exchange, not the machine.

Foreign access constrains ownership until the runtime knows the access has expired (reads) or completed with a usable produced state (writes). Which foreign subsystems exist and how completion is observed are unspecified by the machine. A runtime with no exchange conventions simply permits no foreign I/O.

### The warehouse

The runtime need not provide an infinite warehouse. It may impose a **warehouse** — a bound on total parcel extent the merchant may hold. The merchant remains sovereign within that bound.

Exceeding the warehouse is a runtime-defined condition; the runtime need not admit the scheme. The warehouse may expand or contract. On contraction, the runtime may reclaim medium at the next gate for parcels whose claims have been relinquished, but must not destroy a parcel whose claim the merchant still holds.

A preferred warehouse size is a hint: the runtime honors `min(declared, available)`. No preference means the runtime default.

### The physical medium

Physical backing is managed entirely by the runtime. At any gate it may reorganize, relocate, or reclaim medium. None of this is observable to the merchant: **only claims and parcel identity are preserved across physical activity**.

## 6. Gates

A **gate** is the interval between two adjacent dispatches in an execution order.

At a gate, the runtime may:

- Transfer ownership between dispatches
- Inspect or modify contents of parcels it holds in trust
- Relocate physical medium
- Reclaim medium for parcels whose claims have been relinquished
- Insert additional dispatches into the scheme

Within a dispatch (including at yield points), the runtime may exercise gate powers only if they are not observable to the dispatch upon resumption.

## 7. Petitions

A **petition** is how a dispatch requests a service from the runtime. It may be filed only at a **yield point** (§4). The dispatch suspends, the runtime services the petition, then the dispatch resumes.

### At a yield point

1. The script reaches the yield point and signs the petition.
2. The runtime may perform the service — including scheduling other work, delivering parcels to the merchant, or internal bookkeeping.
3. The runtime resumes the script with script-state intact.

Petition conventions (encodings, services offered, how yield points are declared) are unspecified by the machine. A runtime that defines none provides no services beyond execution.

## 8. Scheme transformations

The runtime may transform a scheme before or during execution if observable behavior is unchanged. For any well-formed scheme it admits, it may:

- **Fuse** adjacent dispatches into one, producing the same parcel states
- **Split** one dispatch into several with precedences, producing the same parcel states
- **Reorder** unordered dispatches freely, including concurrent execution
- **Elide** dispatches whose effects can be derived without execution
- **Insert** bookkeeping dispatches that do not alter merchant-observable parcel states

These are algorithms over schemes (§9). Merchants express natural granularity; the runtime reshapes for hardware. Final parcel states are the invariant.

## 9. Algorithms over schemes

A scheme is first-class data. **Execution** — producing parcel states consistent with the partial order — is the defining algorithm, but not the only one.

Others include (without limit): fusion, splitting, specialization, differentiation, distribution across runtimes, scheduling annotations, analysis without execution, and composition.

A runtime need only provide execution; the machine admits all possible algorithms over schemes.

## 10. Conformance

A runtime conforms if and only if, for every well-formed scheme it admits, it produces parcel states consistent with at least one execution that respects:

- The scheme's partial order
- Ownership rules in §5
- Exchange constraints in §5
- Parcel-identity contracts in §5 and §6
- Yield-point petition constraints in §7 (no full gate powers at yield points)

A conforming runtime may decline to admit a scheme. It need not support every script language, parcel type, or physical extent — only those it advertises. It need not support yielding scripts.

The machine does not specify performance, latency, energy, or resource consumption — only the *meaning* of what the merchant executes.

## Appendix: Glossary (non-normative)

Loose analogues for readers familiar with other models. The Fondaco terms above are authoritative.

| Fondaco term | Loose analogue | Note |
|---|---|---|
| Merchant | Program, application, client | Sovereign owner of all parcels |
| Scheme | Command buffer, render graph, dataflow graph | First-class; algorithms operate over it |
| Dispatch | Kernel launch, shader invocation | Atomic; runs to completion |
| Script | Kernel body, shader source | Opaque except at yield points |
| Yielding script | Coroutine, fiber body | Structured suspend/resume |
| Yield point | Suspension point, syscall boundary | Collectively transfers control to the runtime |
| Claims | Descriptor set, root signature, argument buffer | Names mapped to parcels with ownership |
| Parcel | Buffer, texture, resource | Stable identity; medium may move |
| Ledger | Resource-state / hazard tracker | Conserved across schemes; not merchant-addressable |
| Exchange | Swapchain present, DMA, host-visible mapping, pixel blit | Linear per-execution settlement |
| Ownership (public) | SRV, sampled image, read-only binding | Concurrent reads |
| Ownership (private) | UAV, storage image, read-write binding | Exclusive tenant |
| Ownership (private-inaugural) | Discard/clear load op | Exclusive write; prior state abandoned |
| Warehouse | Device memory budget | Bound relative to other merchants |
| Gate | Fence, barrier, semaphore | Full intervention between dispatches |
| Petition | System call, trap | Mid-dispatch; limited service, not full gate powers |
| Scheme transformation | Compiler pass, kernel fusion | Observable parcel states must be preserved |
| Runtime | OS kernel, driver, command queue | Conceptually one agent; may be distributed |

Analogues are not equivalences. In particular: a scheme *is* the program, not merely a scheduling artifact; parcel identity must stay opaque (descriptor heaps are backend-private); petitions at yield points do not grant full gate powers.
