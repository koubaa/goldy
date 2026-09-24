//! Exact laws on the shapes that motivate semantic fusion, checked bit for bit against
//! the reference interpreter.

use super::*;
use std::collections::BTreeSet;

/// Deterministic values in `[-1, 1)`.
pub(super) fn data(seed: u32, n: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

pub(super) fn run(region: &Region, env: &Environment) -> Environment {
    let mut env = env.clone();
    region.evaluate(&mut env).unwrap();
    env
}

pub(super) fn bits(env: &Environment) -> Vec<(ParcelId, Vec<u32>)> {
    env.parcels
        .iter()
        .map(|(&p, d)| (p, d.iter().map(|x| x.to_bits()).collect()))
        .collect()
}

/// Asserts every parcel is bit-identical after the rewrite.
fn assert_exact(before: &Region, after: &Region, env: &Environment) {
    after.validate().unwrap();
    assert_eq!(bits(&run(before, env)), bits(&run(after, env)));
}

pub(super) fn body(region: &Region, id: ValueId) -> &Term {
    &region.value(id).unwrap().definition.as_ref().unwrap().body
}

fn rewritten(region: &Region, rewrite: impl FnOnce(&mut Region)) -> Region {
    let mut out = region.clone();
    rewrite(&mut out);
    out
}

#[test]
fn contraction_epilogue() {
    let (m, n) = (5, 7);
    let mut r = Region::new();
    let i = r.index("i");
    let k = r.index("k");
    let w = r.input("W", &[m, n], Storage::packed(ParcelId(0), &[m, n]));
    let h = r.input("h", &[n], Storage::packed(ParcelId(1), &[n]));
    let x = r.input("x", &[m], Storage::packed(ParcelId(2), &[m]));
    let y = r.temporary(
        "y",
        &[m],
        &[i],
        Term::sum(k, n, Term::read(w, [i, k]) * Term::read(h, [k])),
    );
    // Stored over the input it reads.
    let x2 = r.output(
        "x2",
        &[m],
        &[i],
        Term::read(x, [i]) + Term::read(y, [i]),
        Storage::packed(ParcelId(2), &[m]),
    );
    r.validate().unwrap();
    let env = Environment::new()
        .with_parcel(ParcelId(0), data(1, m * n))
        .with_parcel(ParcelId(1), data(2, n))
        .with_parcel(ParcelId(2), data(3, m));

    let fused = rewritten(&r, |f| assert_eq!(f.inline_temporaries(), 1));
    assert!(fused.value(y).is_none());
    assert_eq!(
        fused.show_definition(x2).unwrap(),
        "x2[i] = x[i] + sum{k'<7}(W[i, k'] * h[k'])"
    );
    assert_exact(&r, &fused, &env);

    let out = run(&fused, &env);
    let (wd, hd, xd) = (
        &env.parcels[&ParcelId(0)],
        &env.parcels[&ParcelId(1)],
        &env.parcels[&ParcelId(2)],
    );
    for row in 0..m as usize {
        let dot = (0..n as usize).fold(0.0f32, |acc, c| acc + wd[row * n as usize + c] * hd[c]);
        assert_eq!(out.parcels[&ParcelId(2)][row].to_bits(), (xd[row] + dot).to_bits());
    }
}

#[test]
fn shared_operand_reductions() {
    let (m, n) = (6, 5);
    let mut r = Region::new();
    let i = r.index("i");
    let k = r.index("k");
    let w1 = r.input("W1", &[m, n], Storage::packed(ParcelId(0), &[m, n]));
    let w3 = r.input("W3", &[m, n], Storage::packed(ParcelId(1), &[m, n]));
    let x = r.input("x", &[n], Storage::packed(ParcelId(2), &[n]));
    let g = r.temporary(
        "g",
        &[m],
        &[i],
        Term::sum(k, n, Term::read(w1, [i, k]) * Term::read(x, [k])),
    );
    let u = r.temporary(
        "u",
        &[m],
        &[i],
        Term::sum(k, n, Term::read(w3, [i, k]) * Term::read(x, [k])),
    );
    let gi = || Term::read(g, [i]);
    let gated = gi() * (Term::lit(1.0) + (-gi()).exp()).recip();
    let out = r.output(
        "out",
        &[m],
        &[i],
        gated * Term::read(u, [i]),
        Storage::packed(ParcelId(3), &[m]),
    );
    let env = Environment::new()
        .with_parcel(ParcelId(0), data(1, m * n))
        .with_parcel(ParcelId(1), data(2, m * n))
        .with_parcel(ParcelId(2), data(3, n))
        .with_parcel(ParcelId(3), vec![0.0; m as usize]);

    let fused = rewritten(&r, |f| assert_eq!(f.inline_temporaries(), 3));
    let sums = body(&fused, out).reductions();
    assert_eq!(sums.len(), 3);
    // Substitution duplicated `g`. The copies are one computation.
    assert!(sums[0].alpha_eq(sums[1]));
    assert!(!sums[0].alpha_eq(sums[2]));
    // Every reduction runs over the same extent at the same output index, so one loop
    // can serve them all and load `x[k]` once.
    for sum in &sums {
        assert_eq!(sum.free_indices(), BTreeSet::from([i]));
        assert!(matches!(sum, Term::Reduce { extent: 5, .. }));
    }
    assert_exact(&r, &fused, &env);
}

#[test]
fn recompute_prologue() {
    let (m, n) = (5, 8);
    let mut r = Region::new();
    let i = r.index("i");
    let j = r.index("j");
    let k = r.index("k");
    let eps = r.scalar("eps");
    let w = r.input("W", &[m, n], Storage::packed(ParcelId(0), &[m, n]));
    let x = r.input("x", &[n], Storage::packed(ParcelId(1), &[n]));
    let gamma = r.input("gamma", &[n], Storage::packed(ParcelId(2), &[n]));
    let ss = r.temporary("ss", &[], &[], Term::sum(j, n, Term::read(x, [j]) * Term::read(x, [j])));
    let ss_value = Term::Read {
        value: ss,
        index: Vec::new(),
    };
    let scale = (ss_value * Term::lit(1.0 / n as f32) + Term::scalar(eps))
        .sqrt()
        .recip();
    let normed = r.temporary(
        "normed",
        &[n],
        &[k],
        (Term::read(x, [k]) * Term::read(gamma, [k])) * scale,
    );
    let y = r.output(
        "y",
        &[m],
        &[i],
        Term::sum(k, n, Term::read(w, [i, k]) * Term::read(normed, [k])),
        Storage::packed(ParcelId(3), &[m]),
    );
    let env = Environment::new()
        .with_parcel(ParcelId(0), data(1, m * n))
        .with_parcel(ParcelId(1), data(2, n))
        .with_parcel(ParcelId(2), data(3, n))
        .with_parcel(ParcelId(3), vec![0.0; m as usize])
        .with_scalar(eps, 1e-5);

    let fused = rewritten(&r, |f| assert_eq!(f.inline_temporaries(), 2));
    let rows = body(&fused, y).invariants(&[i]);
    assert_eq!(rows.len(), 1);
    // The normalized operand does not depend on the output row: one copy per
    // workgroup serves every row it computes.
    assert_eq!(rows[0].free, BTreeSet::from([k]));
    let scalars = rows[0].term.invariants(&[k]);
    assert_eq!(scalars.len(), 1);
    // The norm's scale depends on no index: compute it once rather than per term.
    assert!(scalars[0].free.is_empty());
    assert_eq!(scalars[0].term.reductions().len(), 1);
    assert_exact(&r, &fused, &env);
}

#[test]
fn destination_composition() {
    let (d, n, rows) = (4u32, 3u32, 5u32);
    let mut r = Region::new();
    let i = r.index("i");
    let k = r.index("k");
    let pos = r.index_param("pos", 0..i64::from(rows));
    let w = r.input("Wv", &[d, n], Storage::packed(ParcelId(0), &[d, n]));
    let x = r.input("x", &[n], Storage::packed(ParcelId(1), &[n]));
    let v = r.temporary(
        "v",
        &[d],
        &[i],
        Term::sum(k, n, Term::read(w, [i, k]) * Term::read(x, [k])),
    );
    // One row of a `[rows, d]` cache, chosen at run time.
    let row = r.output(
        "row",
        &[d],
        &[i],
        Term::read(v, [i]),
        Storage::strided(ParcelId(2), pos * i64::from(d), &[1]),
    );
    let env = Environment::new()
        .with_parcel(ParcelId(0), data(1, d * n))
        .with_parcel(ParcelId(1), data(2, n))
        .with_parcel(ParcelId(2), data(3, rows * d))
        .with_index_param(pos, 2);

    let fused = rewritten(&r, |f| assert_eq!(f.inline_temporaries(), 1));
    assert!(fused.value(v).is_none());
    assert_eq!(
        fused.show_definition(row).unwrap(),
        "row[i] = sum{k'<3}(Wv[i, k'] * x[k'])"
    );
    let storage = fused.value(row).unwrap().storage.as_ref().unwrap();
    assert_eq!(storage.element(&[i.into()]), Affine::from(i) + pos * 4);
    assert_exact(&r, &fused, &env);

    let out = run(&fused, &env);
    for (at, (a, b)) in env.parcels[&ParcelId(2)]
        .iter()
        .zip(&out.parcels[&ParcelId(2)])
        .enumerate()
    {
        if !(8..12).contains(&at) {
            assert_eq!(a.to_bits(), b.to_bits(), "element {at} is outside row 2");
        }
    }
}

#[test]
fn concatenated_contraction_splits() {
    let (a, b, n) = (3u32, 2u32, 4u32);
    let mut r = Region::new();
    let i = r.index("i");
    let k = r.index("k");
    let wq = r.input("Wq", &[a, n], Storage::packed(ParcelId(0), &[a, n]));
    let wk = r.input("Wk", &[b, n], Storage::packed(ParcelId(1), &[b, n]));
    let x = r.input("x", &[n], Storage::packed(ParcelId(2), &[n]));
    // The rows of `Wq` followed by the rows of `Wk`, never stored.
    let wcat = r.temporary(
        "Wcat",
        &[a + b, n],
        &[i, k],
        Term::select(
            i,
            CmpOp::Lt,
            i64::from(a),
            Term::read(wq, [i, k]),
            Term::read(wk, [i - i64::from(a), Affine::from(k)]),
        ),
    );
    let out = r.output(
        "qk",
        &[a + b],
        &[i],
        Term::sum(k, n, Term::read(wcat, [i, k]) * Term::read(x, [k])),
        Storage::packed(ParcelId(3), &[a + b]),
    );
    r.validate().unwrap();
    let env = Environment::new()
        .with_parcel(ParcelId(0), data(1, a * n))
        .with_parcel(ParcelId(1), data(2, b * n))
        .with_parcel(ParcelId(2), data(3, n))
        .with_parcel(ParcelId(3), vec![0.0; (a + b) as usize]);

    assert_eq!(Law::HoistSelect.exactness(), Exactness::Exact);
    let fused = rewritten(&r, |f| {
        assert_eq!(f.inline_temporaries(), 1);
        assert_eq!(f.apply(out, Law::HoistSelect), Ok(2));
    });
    assert_eq!(
        fused.show_definition(out).unwrap(),
        "qk[i] = (i < 3 ? sum{k<4}(Wq[i, k] * x[k]) : sum{k<4}(Wk[i - 3, k] * x[k]))"
    );
    assert_exact(&r, &fused, &env);
}

#[test]
fn substituting_an_output_forwards_it() {
    let n = 4;
    let mut r = Region::new();
    let i = r.index("i");
    let x = r.input("x", &[n], Storage::packed(ParcelId(0), &[n]));
    let y = r.output(
        "y",
        &[n],
        &[i],
        Term::read(x, [i]) * Term::lit(2.0),
        Storage::packed(ParcelId(1), &[n]),
    );
    let z = r.output(
        "z",
        &[n],
        &[i],
        Term::read(y, [i]) + Term::lit(1.0),
        Storage::packed(ParcelId(2), &[n]),
    );
    let env = Environment::new()
        .with_parcel(ParcelId(0), data(1, n))
        .with_parcel(ParcelId(1), vec![0.0; n as usize])
        .with_parcel(ParcelId(2), vec![0.0; n as usize]);

    let fused = rewritten(&r, |f| {
        assert_eq!(f.substitute(y), Ok(1));
        assert!(f.eliminate_dead().is_empty());
    });
    assert_eq!(fused.readers(y), 0);
    assert!(fused.value(y).is_some(), "an output is still stored");
    assert_eq!(fused.show_definition(z).unwrap(), "z[i] = x[i] * 2.0 + 1.0");
    assert_exact(&r, &fused, &env);
}

#[test]
fn validation_rejects_ill_formed_regions() {
    let mut base = Region::new();
    let i = base.index("i");
    let k = base.index("k");
    let x = base.input("x", &[4], Storage::packed(ParcelId(0), &[4]));

    let mut shifted = base.clone();
    shifted.temporary("y", &[4], &[i], Term::read(x, [i + 1]));
    assert_eq!(
        shifted.validate(),
        Err(RegionError::OutOfBounds {
            value: "y".into(),
            read: "x".into(),
            axis: 0,
            bounds: (1, 4),
            extent: 4,
        })
    );

    // A guard that excludes the last element makes the same read legal.
    let mut guarded = base.clone();
    guarded.temporary(
        "y",
        &[4],
        &[i],
        Term::select(i, CmpOp::Lt, 3i64, Term::read(x, [i + 1]), Term::lit(0.0)),
    );
    guarded.validate().unwrap();

    let mut unbound = base.clone();
    unbound.temporary("y", &[4], &[i], Term::read(x, [k]));
    assert!(matches!(unbound.validate(), Err(RegionError::Unbound { .. })));

    let mut rebound = base.clone();
    rebound.temporary("y", &[4], &[i], Term::sum(i, 4, Term::read(x, [i])));
    assert!(matches!(rebound.validate(), Err(RegionError::Rebound { .. })));

    let mut unstored = base.clone();
    unstored.output(
        "y",
        &[4],
        &[i],
        Term::read(x, [i]),
        Storage::packed(ParcelId(1), &[4, 1]),
    );
    assert!(matches!(unstored.validate(), Err(RegionError::Storage { .. })));
}

#[test]
fn alpha_equivalence_respects_free_indices() {
    let mut r = Region::new();
    let i = r.index("i");
    let j = r.index("j");
    let k = r.index("k");
    let x = r.input("x", &[4, 4], Storage::packed(ParcelId(0), &[4, 4]));
    let a = Term::sum(k, 4, Term::read(x, [i, k]));
    assert!(a.alpha_eq(&Term::sum(j, 4, Term::read(x, [i, j]))));
    assert!(!a.alpha_eq(&Term::sum(j, 4, Term::read(x, [j, j]))));
    assert!(!a.alpha_eq(&Term::sum(k, 3, Term::read(x, [i, k]))));
    // `i` is free on one side and bound on the other.
    assert!(!a.alpha_eq(&Term::sum(i, 4, Term::read(x, [i, i]))));
}

#[test]
fn region_prints_the_evaluation_tree() {
    let mut r = Region::new();
    let i = r.index("i");
    let x = r.input("x", &[3], Storage::packed(ParcelId(0), &[3]));
    let t = r.temporary(
        "t",
        &[3],
        &[i],
        Term::read(x, [i]) + (Term::read(x, [i]) + Term::lit(1.0)),
    );
    r.output(
        "u",
        &[3],
        &[i],
        -Term::read(t, [i]).max(Term::lit(0.0)),
        Storage::packed(ParcelId(1), &[3]),
    );
    assert_eq!(
        r.to_string(),
        "input x[3]\ntemp t[i] = x[i] + (x[i] + 1.0)\noutput u[i] = -max(t[i], 0.0)\n"
    );
}
