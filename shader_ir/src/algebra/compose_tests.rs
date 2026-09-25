//! Composition, storage inversion and lowering, checked against sequential evaluation.

use super::tests::{bits, body, data, run};
use super::*;
use crate::{BinOp, Expr, Stmt};

/// `y[i] = sum{k}(W[i, k] * h[k])` from packed `W` and `h` into packed `y`.
fn gemv(m: u32, n: u32, parcels: [u32; 3], order: ReduceOrder) -> Region {
    let mut r = Region::new();
    let i = r.index("i");
    let k = r.index("k");
    let w = r.input("W", &[m, n], Storage::packed(ParcelId(parcels[0]), &[m, n]));
    let h = r.input("h", &[n], Storage::packed(ParcelId(parcels[1]), &[n]));
    r.output(
        "y",
        &[m],
        &[i],
        Term::reduce_in(ReduceOp::Sum, order, k, n, Term::read(w, [i, k]) * Term::read(h, [k])),
        Storage::packed(ParcelId(parcels[2]), &[m]),
    );
    r
}

/// `out[i] = a[i] + b[i]` over packed parcels.
fn add(m: u32, parcels: [u32; 3]) -> Region {
    let mut r = Region::new();
    let i = r.index("i");
    let a = r.input("a", &[m], Storage::packed(ParcelId(parcels[0]), &[m]));
    let b = r.input("b", &[m], Storage::packed(ParcelId(parcels[1]), &[m]));
    r.output(
        "out",
        &[m],
        &[i],
        Term::read(a, [i]) + Term::read(b, [i]),
        Storage::packed(ParcelId(parcels[2]), &[m]),
    );
    r
}

/// Every region in turn, each seeing what the previous ones stored.
fn run_all(regions: &[&Region], env: &Environment) -> Environment {
    regions.iter().fold(env.clone(), |env, r| run(r, &env))
}

const GEMV_LANES: ReduceOrder = ReduceOrder::Lanes {
    lanes: 32,
    accumulators: 2,
};

#[test]
fn lane_order_is_the_gemv_kernel_association() {
    let (m, n) = (3, 100);
    let r = gemv(m, n, [0, 1, 2], GEMV_LANES);
    let (w, h) = (data(1, m * n), data(2, n));
    let env = Environment::new()
        .with_parcel(ParcelId(0), w.clone())
        .with_parcel(ParcelId(1), h.clone())
        .with_parcel(ParcelId(2), vec![0.0; m as usize]);
    let got = run(&r, &env);
    // `gemv_f32`, lane by lane: two strided accumulators, then a tree over 32 lanes.
    for row in 0..m as usize {
        let at = |j: usize| w[row * n as usize + j] * h[j];
        let mut partial: Vec<f32> = (0..32)
            .map(|lane| {
                let (mut acc0, mut acc1, mut j) = (0.0f32, 0.0f32, lane);
                while j + 32 < n as usize {
                    acc0 += at(j);
                    acc1 += at(j + 32);
                    j += 64;
                }
                if j < n as usize {
                    acc0 += at(j);
                }
                acc0 + acc1
            })
            .collect();
        let mut s = 16;
        while s > 0 {
            for l in 0..s {
                partial[l] += partial[l + s];
            }
            s /= 2;
        }
        assert_eq!(
            got.parcels[&ParcelId(2)][row].to_bits(),
            partial[0].to_bits(),
            "row {row}"
        );
    }
    // The sequential order is a different term.
    let sequential = gemv(m, n, [0, 1, 2], ReduceOrder::Sequential);
    assert!(!body(&r, ValueId(2)).alpha_eq(body(&sequential, ValueId(2))));
}

#[test]
fn composition_forwards_an_epilogue() {
    let m = 9;
    let mut fused = gemv(m, 40, [0, 1, 2], GEMV_LANES);
    let first = fused.clone();
    // The residual reads `y` and updates `x` in place.
    let second = add(m, [3, 2, 3]);
    let appended = fused.then(&second).unwrap();
    assert_eq!((appended.forwards.len(), appended.covered.len()), (1, 0));
    let env = Environment::new()
        .with_parcel(ParcelId(0), data(1, m * 40))
        .with_parcel(ParcelId(1), data(2, 40))
        .with_parcel(ParcelId(2), vec![0.0; m as usize])
        .with_parcel(ParcelId(3), data(3, m));
    let want = bits(&run_all(&[&first, &second], &env));
    assert_eq!(bits(&run(&fused, &env)), want);

    let mut inlined = fused.clone();
    inlined.inline_temporaries();
    inlined.substitute(ValueId(2)).unwrap();
    let out = appended.values[2].unwrap();
    assert_eq!(
        inlined.show_definition(out).unwrap(),
        "out[i] = a[i] + sum{k'<40 by 32x2}(W[i, k'] * h[k'])"
    );
    assert_eq!(bits(&run(&inlined, &env)), want);

    let lowered = lower(&fused, &[]).unwrap();
    assert_eq!(
        lowered.schedule,
        Schedule::Lanes {
            lanes: 32,
            elements: 4,
            exchange: Exchange::Workgroup
        }
    );
    assert_eq!(lowered.groups, [3, 1, 1]);
    assert_eq!(lowered.kernel.workgroup_size, [128, 1, 1]);
    assert_eq!(
        lowered.parcels,
        [
            (ParcelId(0), false),
            (ParcelId(1), false),
            (ParcelId(2), true),
            (ParcelId(3), true)
        ]
    );
    let slang = crate::emit_canonical_compute_source(&lowered.kernel)
        .source
        .canonical_slang;
    // One reduction feeds both outputs.
    assert_eq!(slang.matches("groupshared float").count(), 1, "{slang}");
}

#[test]
fn composition_relocates_into_a_selected_row() {
    let (d, rows) = (6u32, 5i64);
    let mut fused = gemv(d, 7, [0, 1, 2], GEMV_LANES);
    let first = fused.clone();
    // cache[p, j] = v[0, j]: a one-row view of `y` stored into row `p`.
    let mut second = Region::new();
    let a = second.index("a");
    let j = second.index("j");
    let p = second.index_param("p", 0..rows);
    let v = second.input("v", &[1, d], Storage::packed(ParcelId(2), &[1, d]));
    second.output(
        "cache",
        &[1, d],
        &[a, j],
        Term::read(v, [a, j]),
        Storage::strided(ParcelId(3), p * i64::from(d), &[i64::from(d), 1]),
    );
    let appended = fused.then(&second).unwrap();
    assert_eq!(appended.forwards.len(), 1);
    let p = appended.index_params[0];
    let env = Environment::new()
        .with_parcel(ParcelId(0), data(1, d * 7))
        .with_parcel(ParcelId(1), data(2, 7))
        .with_parcel(ParcelId(2), vec![0.0; d as usize])
        .with_parcel(ParcelId(3), vec![0.0; rows as usize * d as usize])
        .with_index_param(p, 3);
    assert_eq!(bits(&run(&fused, &env)), bits(&run_all(&[&first, &second], &env)));

    let source = IndexSource {
        parcel: ParcelId(4),
        element: 0,
        element_type: crate::ElementType::I32,
    };
    let lowered = lower(&fused, &[source]).unwrap();
    assert_eq!(lowered.parcels.last(), Some(&(ParcelId(4), false)));
    let slang = crate::emit_canonical_compute_source(&lowered.kernel)
        .source
        .canonical_slang;
    assert!(slang.contains("BufRO<int> p4"), "{slang}");
    assert_eq!(slang.matches("groupshared float").count(), 1, "{slang}");
    assert!(matches!(lower(&fused, &[]), Err(LowerError::IndexSource { .. })));
}

#[test]
fn composition_covers_an_overwritten_output() {
    let m = 8;
    let scale = |op: fn(Term) -> Term, name: &str| {
        let mut r = Region::new();
        let i = r.index("i");
        let x = r.input("x", &[m], Storage::packed(ParcelId(0), &[m]));
        r.output(
            name,
            &[m],
            &[i],
            op(Term::read(x, [i])),
            Storage::packed(ParcelId(0), &[m]),
        );
        r
    };
    let first = scale(|x| x * Term::lit(2.0), "x2");
    let second = scale(|x| x + Term::lit(1.0), "x3");
    let mut fused = first.clone();
    let appended = fused.then(&second).unwrap();
    assert_eq!(appended.covered, [ValueId(1)]);
    assert_eq!(fused.value(ValueId(1)).unwrap().role, Role::Temporary);
    let env = Environment::new().with_parcel(ParcelId(0), data(4, m));
    assert_eq!(bits(&run(&fused, &env)), bits(&run_all(&[&first, &second], &env)));
    let lowered = lower(&fused, &[]).unwrap();
    assert_eq!(lowered.schedule, Schedule::Threads { workgroup: 256 });
    assert_eq!(lowered.parcels, [(ParcelId(0), true)]);
}

#[test]
fn composition_rejects_partial_overlap() {
    let window = |start: i64, parcel: u32| Storage::strided(ParcelId(parcel), start, &[1]);
    let copy = |len: u32, from: Storage, to: Storage| {
        let mut r = Region::new();
        let i = r.index("i");
        let x = r.input("x", &[len], from);
        r.output("y", &[len], &[i], Term::read(x, [i]), to);
        r
    };
    let first = copy(4, window(0, 0), window(0, 1));
    // Reads elements 2..6 of what `first` wrote as 0..4.
    let shifted_read = copy(4, window(2, 1), window(0, 2));
    assert!(matches!(
        first.clone().then(&shifted_read),
        Err(ComposeError::Unforwardable { .. })
    ));
    // A read inside the written window forwards.
    let inside = copy(2, window(1, 1), window(0, 2));
    assert_eq!(first.clone().then(&inside).unwrap().forwards.len(), 1);
    // Writes elements 2..6 of what `first` wrote as 0..4.
    let shifted_write = copy(4, window(0, 0), window(2, 1));
    assert!(matches!(
        first.clone().then(&shifted_write),
        Err(ComposeError::OverlappingOutputs { .. })
    ));
}

#[test]
fn preimage_inverts_injective_storage() {
    let mut r = Region::new();
    let i = r.index("i");
    let j = r.index("j");
    let rows = Storage::strided(ParcelId(0), 8, &[4, 1]);
    let range = |s: Sym| match s {
        Sym::Index(v) if v == i => Some((0, 1)),
        Sym::Index(v) if v == j => Some((0, 3)),
        _ => None,
    };
    assert_eq!(
        rows.preimage(&[2, 4], &(i * 4 + 11), &range),
        Some(vec![Affine::from(i), Affine::constant(3)])
    );
    assert_eq!(
        rows.preimage(&[2, 4], &(Affine::from(j) + 8), &range),
        Some(vec![Affine::constant(0), Affine::from(j)])
    );
    // Crosses from row 0 into row 1: not affine in `j`.
    assert_eq!(rows.preimage(&[2, 4], &(Affine::from(j) + 10), &range), None);
    assert!(!Storage::strided(ParcelId(0), 0, &[2, 1]).injective(&[2, 4]));
}

#[test]
fn lowering_guards_an_output_spanning_part_of_the_domain() {
    // A GEMV over 9 rows, then an epilogue over its first 6.
    let mut fused = gemv(9, 40, [0, 1, 2], GEMV_LANES);
    fused.then(&add(6, [2, 3, 4])).unwrap();
    let lowered = lower(&fused, &[]).unwrap();
    let Some(Stmt::If { then_body, .. }) = lowered.kernel.body.last() else {
        panic!("stores run under the validity guard");
    };
    let guarded: Vec<&Stmt> = then_body
        .iter()
        .filter(|s| matches!(s, Stmt::If { cond: Expr::Binary { op: BinOp::Lt, right, .. }, .. } if **right == Expr::LitI32(6)))
        .collect();
    // The epilogue's value and its store, each within its six elements.
    assert_eq!(guarded.len(), 2, "{then_body:?}");

    // A narrower product's own lane reduction would read past its rows on the
    // epilogue's domain, so it runs beside it, on its own.
    let mut partial = add(9, [0, 1, 2]);
    partial.then(&gemv(6, 40, [3, 4, 5], GEMV_LANES)).unwrap();
    let siblings = lower(&partial, &[]).unwrap();
    assert_eq!(siblings.parts, 2);
    assert!(matches!(siblings.schedule, Schedule::Lanes { elements: 4, .. }));
    // Nine elements a thread each, then six a lane group each: one workgroup and two.
    assert_eq!(siblings.groups, [3, 1, 1]);
}

#[test]
fn lowering_rejects_cross_thread_reads_and_runs_mixed_ranks_side_by_side() {
    let m = 4;
    let mut r = Region::new();
    let i = r.index("i");
    let k = r.index("k");
    let w = r.input("W", &[m, m], Storage::packed(ParcelId(0), &[m, m]));
    let x = r.input("x", &[m], Storage::packed(ParcelId(1), &[m]));
    // Each element reads all of `x` while other threads overwrite it.
    r.output(
        "x2",
        &[m],
        &[i],
        Term::read(x, [i]) + Term::sum(k, m, Term::read(w, [i, k]) * Term::read(x, [k])),
        Storage::packed(ParcelId(1), &[m]),
    );
    assert!(matches!(lower(&r, &[]), Err(LowerError::Race { .. })));

    let mut matrix = Region::new();
    let (i, j) = (matrix.index("i"), matrix.index("j"));
    let a = matrix.input("a", &[2, 3], Storage::packed(ParcelId(0), &[2, 3]));
    matrix.output(
        "b",
        &[2, 3],
        &[i, j],
        Term::read(a, [i, j]).abs(),
        Storage::packed(ParcelId(1), &[2, 3]),
    );
    matrix.then(&add(4, [2, 3, 4])).unwrap();
    let siblings = lower(&matrix, &[]).unwrap();
    assert_eq!(siblings.parts, 2);
    assert_eq!(siblings.groups, [2, 1, 1]);

    // The same storage read in place by its own thread in one part, and at the same
    // index by a thread of another part, which cannot see it before it is overwritten.
    let copy_beside = |scale_in_place: bool| {
        let mut r = Region::new();
        let (i, j) = (r.index("i"), r.index("j"));
        let flat = r.input("a", &[4], Storage::packed(ParcelId(0), &[4]));
        let square = r.input("a2", &[2, 2], Storage::packed(ParcelId(0), &[2, 2]));
        let scaled = if scale_in_place { ParcelId(0) } else { ParcelId(2) };
        r.output(
            "scaled",
            &[4],
            &[i],
            Term::read(flat, [i]) * Term::lit(2.0),
            Storage::packed(scaled, &[4]),
        );
        r.output(
            "copy",
            &[2, 2],
            &[i, j],
            Term::read(square, [i, j]),
            Storage::packed(ParcelId(1), &[2, 2]),
        );
        lower(&r, &[])
    };
    assert_eq!(copy_beside(false).unwrap().parts, 2);
    assert!(matches!(copy_beside(true), Err(LowerError::Race { .. })));
}
