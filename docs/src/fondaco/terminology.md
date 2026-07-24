# Terminology

Vocabulary and status labels used throughout the Fondaco chapters and the rest of this book.

| Authority | Document |
|-----------|----------|
| Machine semantics | [Machine Specification](./specification.md) |
| Goldy realization | [Goldy Runtime Mapping](./goldy-runtime.md) |
| Shipped behavior | Goldy source, tests, and examples |

## Status labels

| Label | Meaning |
|-------|---------|
| **Shipped** | Available in the public Goldy crate today (0.2.x) |
| **Designed** | Specified and intended; not yet implemented, or only partially implemented |
| **Experimental** | Behind a feature flag, alpha binding, or unstable API |
| **Speculative** | Research or exploration; not committed to the roadmap |
| **Historical** | Superseded design kept for context |

Do not describe **Designed**, **Experimental**, or **Speculative** capabilities as if they were **Shipped**.

## Fondaco machine terms

| Term | Definition |
|------|------------|
| **Merchant** | The sovereign client: describes parcels and executes schemes; owns every parcel |
| **Scheme** | First-class computation: dispatches plus precedences (a partial order) |
| **Dispatch** | Atomic unit of work admitted to the machine |
| **Script** | Opaque procedure evaluated by a computing dispatch |
| **Yielding script** | Script with structured yield points where the runtime may be petitioned |
| **Parcel** | Stable identity for data held by the runtime in trust for the merchant |
| **Ownership / claim** | Access right over a parcel for the duration of a dispatch (public, private, or private-inaugural) |
| **Ledger** | Runtime standing record of claims across schemes (not merchant-addressable) |
| **Gate** | Interval between adjacent dispatches where the runtime has full intervention powers |
| **Exchange** | Stable mediated relationship with a foreign subsystem; each execution may publish a linear claim |
| **Warehouse** | Runtime-imposed bound on the total extent of parcels the merchant may hold |
| **Petition** | Structured service request filed at a yield point |

## Goldy API map

| Fondaco term | Goldy type / concept | Notes |
|--------------|----------------------|-------|
| Scheme | `Scheme`, internal `GraphIR` | Public recording and submission API |
| Dispatch | Scheme node (compute, render, copy, clear, present) | Workgroup grid for compute/render |
| Script | Slang via `[goldy_*]` virtual entry points | Sole script language (Goldy choice) |
| Parcel | `Parcel`, `Buffer`, `Texture` | Stable handles; bindless indexing is backend-internal |
| Ownership | `NodeAccess` on scheme nodes | Precedences derived from access modes |
| Ledger | Cross-submission sync (`ParcelStamp`, timeline) | Crate-private; clients use settlement APIs |
| Gate | Submission gate, `Context::boundary_crossed` | Epoch-driven reclamation |
| Exchange | `SurfaceExchange`, `MemoryExchange` | `Transaction` → `Claim` → `consume` / `discard` |
| Warehouse | `BudgetPolicy`, `VramAllocator` | Bound on committed parcel extent |
| Lease | `Lease<T>`, `LeaseRenderTarget` | Temporary view of a parcel for scheme recording |

### Internal terms

| Term | Location | Role |
|------|----------|------|
| **GraphIR** | `task_graph` | Internal scheme representation |
| **Wave / partition analysis** | `task_graph::analysis` | Submission partitioning and transient coloring |
| **Bindless heap** | Backends | Descriptor indexing; not public ABI |

## Reading order

1. [Machine Specification](./specification.md) — normative semantics
2. [Goldy Runtime Mapping](./goldy-runtime.md) — what Goldy ships vs designs
3. [Design Thesis](./design-thesis.md) — why this model on modern GPUs
4. The rest of this book — tutorials, programming model, and APIs
