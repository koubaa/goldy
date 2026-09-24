//! Named operations, their expansion, and graphs of them, checked against sequential
//! evaluation.

use super::tests::{bits, data, run};
use super::*;

const GEMV_LANES: ReduceOrder = ReduceOrder::Lanes {
    lanes: 32,
    accumulators: 2,
};

fn operand(name: &str, parcel: u32, shape: &[u32]) -> Operand {
    Operand::new(name, shape, Storage::packed(ParcelId(parcel), shape))
}

/// Packed operands on `parcels`, shaped by their labels.
fn contraction(
    names: [&str; 3],
    parcels: [u32; 3],
    extents: &[u32],
    labels: [&[usize]; 3],
    order: ReduceOrder,
) -> Contraction {
    let shape = |labels: &[usize]| labels.iter().map(|&l| extents[l]).collect::<Vec<_>>();
    Contraction {
        extents: extents.to_vec(),
        lhs: operand(names[0], parcels[0], &shape(labels[0])),
        rhs: operand(names[1], parcels[1], &shape(labels[1])),
        out: operand(names[2], parcels[2], &shape(labels[2])),
        labels: labels.map(<[usize]>::to_vec),
        order,
    }
}

fn op(kind: OpKind) -> Op {
    Op::new(Params::default(), kind)
}

/// `names[2][i] = sum{s}(names[0][i, s] * names[1][s])`.
fn gemv(names: [&str; 3], parcels: [u32; 3], m: u32, k: u32) -> Op {
    op(OpKind::Contraction(contraction(
        names,
        parcels,
        &[m, k],
        [&[0, 1], &[1], &[0]],
        GEMV_LANES,
    )))
}

/// `out = f(inputs..)` element-wise over `[n]`, from and into packed parcels.
fn map(inputs: &[(&str, u32)], out: (&str, u32), n: u32, body: Term) -> Op {
    op(OpKind::Map(Map {
        shape: vec![n],
        inputs: inputs.iter().map(|&(name, p)| operand(name, p, &[n])).collect(),
        outputs: vec![(operand(out.0, out.1, &[n]), body)],
    }))
}

fn env(parcels: &[(u32, Vec<f32>)]) -> Environment {
    parcels.iter().fold(Environment::new(), |env, (p, d)| {
        env.with_parcel(ParcelId(*p), d.clone())
    })
}

/// The graph's region stores what its operations store, one after another.
fn assert_sequential(graph: &Graph, env: &Environment) {
    let sequential = graph
        .ops()
        .iter()
        .fold(env.clone(), |env, op| run(&op.expand().unwrap().region, &env));
    assert_eq!(bits(&run(graph.region(), env)), bits(&sequential));
}

#[test]
fn contraction_labels_classify_axes() {
    // C[b, i, j] = sum{s}(A[b, i, s] * B[b, s, j]), labels b, i, j, s.
    let batched = contraction(
        ["A", "B", "C"],
        [0, 1, 2],
        &[2, 3, 4, 5],
        [&[0, 1, 3], &[0, 3, 2], &[0, 1, 2]],
        ReduceOrder::Sequential,
    );
    assert_eq!(batched.batch(), [0]);
    assert_eq!(batched.free(Factor::Lhs), [1]);
    assert_eq!(batched.free(Factor::Rhs), [2]);
    assert_eq!(batched.summed(), [3]);
    op(OpKind::Contraction(batched.clone())).expand().unwrap();

    let expand = |c: Contraction| op(OpKind::Contraction(c)).expand().map(drop);
    // A label only one factor carries must index the result.
    let dangling = contraction(
        ["A", "B", "C"],
        [0, 1, 2],
        &[3, 4, 5],
        [&[0, 2], &[1], &[0, 1]],
        ReduceOrder::Sequential,
    );
    assert_eq!(expand(dangling), Err(OpError::Labels));
    // An outer product sums nothing.
    let outer = contraction(
        ["a", "b", "C"],
        [0, 1, 2],
        &[3, 4],
        [&[0], &[1], &[0, 1]],
        ReduceOrder::Sequential,
    );
    assert_eq!(expand(outer), Err(OpError::Labels));
    // A lane order names one sum.
    let two_sums = contraction(
        ["A", "B", "c"],
        [0, 1, 2],
        &[3, 4, 5],
        [&[0, 1, 2], &[1, 2], &[0]],
        GEMV_LANES,
    );
    assert_eq!(expand(two_sums), Err(OpError::Order));
    let mut short = batched;
    short.out.shape[2] = 9;
    assert!(matches!(expand(short), Err(OpError::Shape { .. })));
}

#[test]
fn contraction_expands_to_its_sum_of_products() {
    let (m, k) = (3, 100);
    let expanded = gemv(["A", "x", "y"], [0, 1, 2], m, k).expand().unwrap();
    assert_eq!(
        expanded.region.show_definition(expanded.outputs[0]).unwrap(),
        "y[i0] = sum{s1<100 by 32x2}(A[i0, s1] * x[s1])"
    );

    // A transposed GEMM is strides: A is stored k by m.
    let (m, n, k) = (4, 3, 5);
    let mut gemm = contraction(
        ["A", "B", "C"],
        [0, 1, 2],
        &[m, n, k],
        [&[0, 2], &[2, 1], &[0, 1]],
        ReduceOrder::Sequential,
    );
    gemm.lhs.storage = Storage::strided(ParcelId(0), 0, &[1, i64::from(m)]);
    let (a, b) = (data(1, m * k), data(2, k * n));
    let got = run(
        &op(OpKind::Contraction(gemm)).expand().unwrap().region,
        &env(&[(0, a.clone()), (1, b.clone()), (2, vec![0.0; (m * n) as usize])]),
    );
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut acc = 0.0f32;
            for p in 0..k as usize {
                acc += a[p * m as usize + i] * b[p * n as usize + j];
            }
            let at = i * n as usize + j;
            assert_eq!(got.parcels[&ParcelId(2)][at].to_bits(), acc.to_bits(), "C[{i}, {j}]");
        }
    }
}

#[test]
fn maps_and_reductions_expand_element_wise() {
    let mut params = Params::default();
    let k = params.scalar("k");
    let fma = Op::new(
        params,
        OpKind::Map(Map {
            shape: vec![4],
            inputs: vec![operand("a", 0, &[4]), operand("b", 1, &[4])],
            outputs: vec![(operand("out", 2, &[4]), Term::arg(0) * Term::scalar(k) + Term::arg(1))],
        }),
    );
    let expanded = fma.expand().unwrap();
    assert_eq!(
        expanded.region.show_definition(expanded.outputs[0]).unwrap(),
        "out[i0] = a[i0] * k + b[i0]"
    );

    let mean = op(OpKind::Reduction(Reduction {
        op: ReduceOp::Sum,
        order: ReduceOrder::Sequential,
        axis: 1,
        input: operand("x", 0, &[2, 4]),
        out: operand("m", 1, &[2, 1]),
        finish: Term::arg(0) / Term::lit(4.0),
    }));
    let expanded = mean.expand().unwrap();
    assert_eq!(
        expanded.region.show_definition(expanded.outputs[0]).unwrap(),
        "m[i0, i1] = sum{s<4}(x[i0, s]) / 4.0"
    );

    let out_of_range = map(&[("a", 0)], ("out", 1), 4, Term::arg(1));
    assert_eq!(out_of_range.expand().map(drop), Err(OpError::Body));
    let unnamed_scalar = map(&[("a", 0)], ("out", 1), 4, Term::arg(0) * Term::scalar(k));
    assert_eq!(unnamed_scalar.expand().map(drop), Err(OpError::Body));
}

#[test]
fn a_graph_forwards_between_operations() {
    let m = 9;
    let mut graph = Graph::new();
    graph.push(gemv(["W", "h", "y"], [0, 1, 2], m, 40)).unwrap();
    // The residual reads `y` and updates `x` in place.
    graph
        .push(map(&[("x", 3), ("y", 2)], ("x", 3), m, Term::arg(0) + Term::arg(1)))
        .unwrap();
    assert_eq!(
        graph.edges(),
        [Edge {
            from: (0, 0),
            to: (1, 1)
        }]
    );
    assert_sequential(
        &graph,
        &env(&[
            (0, data(1, m * 40)),
            (1, data(2, 40)),
            (2, vec![0.0; m as usize]),
            (3, data(3, m)),
        ]),
    );

    let structure = graph.structure();
    assert_eq!(
        structure.to_string(),
        "contraction y[i0] = sum{s1<40 by 32x2}(W[i0, s1] * h[s1])\noutput x[i0] = x[i0] + y[i0]\n"
    );
    assert_eq!(structure.readers(0), [graph.values(1).1[0]]);

    // An operation that does not compose leaves the graph as it was.
    let before = graph.clone();
    let overlapping = map(&[("y", 2)], ("z", 2), m - 1, Term::arg(0));
    assert!(matches!(
        graph.push(overlapping),
        Err(GraphError::Compose(ComposeError::OverlappingOutputs { .. }))
    ));
    assert_eq!(graph, before);
}

#[test]
fn sibling_contractions_share_their_input() {
    let (m, k) = (6, 50);
    let mut graph = Graph::new();
    graph.push(gemv(["W1", "x", "gate"], [0, 2, 3], m, k)).unwrap();
    graph.push(gemv(["W3", "x", "up"], [1, 2, 4], m, k)).unwrap();
    // silu(gate) * up.
    let g = Term::arg(0);
    let silu = g.clone() / ((-g).exp() + Term::lit(1.0));
    graph
        .push(map(&[("gate", 3), ("up", 4)], ("out", 5), m, silu * Term::arg(1)))
        .unwrap();
    assert_eq!(graph.edges().len(), 2);
    assert_sequential(
        &graph,
        &env(&[
            (0, data(1, m * k)),
            (1, data(2, m * k)),
            (2, data(3, k)),
            (3, vec![0.0; m as usize]),
            (4, vec![0.0; m as usize]),
            (5, vec![0.0; m as usize]),
        ]),
    );

    let structure = graph.structure();
    assert_eq!(structure.contractions.len(), 2);
    assert_eq!(
        structure.shared_factors(),
        [SharedFactor {
            first: (0, Factor::Rhs),
            second: (1, Factor::Rhs),
        }]
    );
    assert!(structure.to_string().ends_with("shared rhs of gate and rhs of up\n"));

    // The same weights against different inputs share the weights, row for row.
    let mut rows = Graph::new();
    rows.push(gemv(["W", "a", "ya"], [0, 1, 2], m, k)).unwrap();
    rows.push(gemv(["W", "b", "yb"], [0, 3, 4], m, k)).unwrap();
    assert_eq!(
        rows.structure().shared_factors(),
        [SharedFactor {
            first: (0, Factor::Lhs),
            second: (1, Factor::Lhs),
        }]
    );

    // Both lane-ordered sums run in one strided loop that loads `x` once per step, and
    // in one tree.
    let lowered = lower(graph.region(), &[]).unwrap();
    assert!(matches!(
        lowered.schedule,
        Schedule::Lanes {
            lanes: 32,
            exchange: Exchange::Workgroup,
            ..
        }
    ));
    assert_eq!(lowered.workgroup_bytes, 2 * LANE_WORKGROUP_THREADS * 4);
    let slang = source(&lowered);
    assert_eq!(slang.matches("while (").count(), 2, "{slang}");
    // Two accumulator steps in the loop and one in the tail.
    assert_eq!(slang.matches("p2[").count(), 3, "{slang}");
    assert_eq!(slang.matches("p0[").count(), 3, "{slang}");

    // A subgroup that holds whole lane groups exchanges without workgroup memory.
    let subgroup = Target {
        subgroup: Some(32),
        matrix: false,
    };
    let shuffled = lower_on(graph.region(), &[], &subgroup).unwrap();
    assert!(matches!(
        shuffled.schedule,
        Schedule::Lanes {
            exchange: Exchange::Subgroup,
            ..
        }
    ));
    assert_eq!(shuffled.workgroup_bytes, 0);
    let slang = source(&shuffled);
    assert!(
        slang.contains("WaveReadLaneAt(") && !slang.contains("groupshared"),
        "{slang}"
    );
    // A subgroup narrower than a lane group cannot.
    let narrow = Target {
        subgroup: Some(16),
        matrix: false,
    };
    assert!(matches!(
        lower_on(graph.region(), &[], &narrow).unwrap().schedule,
        Schedule::Lanes {
            exchange: Exchange::Workgroup,
            ..
        }
    ));
}

fn source(lowered: &Lowered) -> String {
    crate::emit_canonical_compute_source(&lowered.kernel)
        .source
        .canonical_slang
}

#[test]
fn sequential_siblings_share_a_loop() {
    let (m, k) = (6, 50);
    let sequential = |names, parcels| {
        op(OpKind::Contraction(contraction(
            names,
            parcels,
            &[m, k],
            [&[0, 1], &[1], &[0]],
            ReduceOrder::Sequential,
        )))
    };
    let mut graph = Graph::new();
    graph.push(sequential(["W1", "x", "gate"], [0, 2, 3])).unwrap();
    graph.push(sequential(["W3", "x", "up"], [1, 2, 4])).unwrap();
    graph
        .push(map(
            &[("gate", 3), ("up", 4)],
            ("out", 5),
            m,
            Term::arg(0) * Term::arg(1),
        ))
        .unwrap();
    let lowered = lower(graph.region(), &[]).unwrap();
    assert_eq!(
        lowered.schedule,
        Schedule::Threads {
            workgroup: WORKGROUP_THREADS
        }
    );
    let slang = source(&lowered);
    assert_eq!(slang.matches("while (").count(), 1, "{slang}");
    assert_eq!(slang.matches("p2[").count(), 1, "{slang}");
}

/// `C[i, j] = sum{s}(A[i, s] * B[s, j])`, then `D = max(C, 0)`.
fn gemm_relu(m: u32, n: u32, k: u32, order: ReduceOrder) -> Graph {
    let mut graph = Graph::new();
    graph
        .push(op(OpKind::Contraction(contraction(
            ["A", "B", "C"],
            [0, 1, 2],
            &[m, n, k],
            [&[0, 2], &[2, 1], &[0, 1]],
            order,
        ))))
        .unwrap();
    graph
        .push(op(OpKind::Map(Map {
            shape: vec![m, n],
            inputs: vec![operand("C", 2, &[m, n])],
            outputs: vec![(operand("D", 3, &[m, n]), Term::arg(0).max(Term::lit(0.0)))],
        })))
        .unwrap();
    graph
}

#[test]
fn matrix_units_take_contractions_only_when_rounding_is_admitted() {
    let (m, n, k) = (20, 24, 40);
    let graph = gemm_relu(m, n, k, ReduceOrder::Sequential);
    let device = Target {
        subgroup: Some(32),
        matrix: true,
    };
    let lowered = lower_graph(&graph, &[], &device, ContractionPrecision::F16Factors).unwrap();
    assert_eq!(lowered.schedule, Schedule::Matrix { subgroup: 32 });
    assert_eq!(lowered.groups, [4, 1, 1]);
    assert_eq!(lowered.kernel.workgroup_size, [32, 1, 1]);
    let slang = source(&lowered);
    assert_eq!(slang.matches("linalg.coopMatMulAdd").count(), 1, "{slang}");
    assert_eq!(slang.matches("groupshared half").count(), 2, "{slang}");
    // The epilogue reads the tile, and both results are stored.
    assert!(slang.contains("max("), "{slang}");
    assert_eq!(slang.matches("p2[").count(), 1, "{slang}");
    assert_eq!(slang.matches("p3[").count(), 1, "{slang}");

    // Exact contractions, or a device without matrix units, keep the exact schedule.
    let exact = lower_graph(&graph, &[], &device, ContractionPrecision::Exact).unwrap();
    assert_eq!(
        exact.schedule,
        Schedule::Threads {
            workgroup: WORKGROUP_THREADS
        }
    );
    let portable = lower_graph(&graph, &[], &Target::default(), ContractionPrecision::F16Factors).unwrap();
    assert_eq!(portable.schedule, exact.schedule);
    assert_eq!(source(&portable), source(&exact));
    // So does a contraction that is not two-dimensional.
    let mut gemv_graph = Graph::new();
    gemv_graph.push(gemv(["W", "x", "y"], [0, 1, 2], m, k)).unwrap();
    assert!(matches!(
        lower_graph(&gemv_graph, &[], &device, ContractionPrecision::F16Factors)
            .unwrap()
            .schedule,
        Schedule::Lanes {
            exchange: Exchange::Subgroup,
            ..
        }
    ));
}

#[test]
fn a_reduction_prologue_stays_inside_a_factor() {
    let (m, n) = (5, 16);
    let mut graph = Graph::new();
    graph
        .push(map(&[("x", 0)], ("sq", 1), n, Term::arg(0) * Term::arg(0)))
        .unwrap();
    graph
        .push(op(OpKind::Reduction(Reduction {
            op: ReduceOp::Sum,
            order: ReduceOrder::Sequential,
            axis: 0,
            input: operand("sq", 1, &[n]),
            out: operand("ms", 2, &[1]),
            finish: Term::arg(0) / Term::lit(n as f32),
        })))
        .unwrap();
    // The mean broadcast over `x`: a zero stride.
    let broadcast = Operand::new("ms", &[n], Storage::strided(ParcelId(2), 0, &[0]));
    graph
        .push(op(OpKind::Map(Map {
            shape: vec![n],
            inputs: vec![operand("x", 0, &[n]), broadcast],
            outputs: vec![(
                operand("xn", 3, &[n]),
                Term::arg(0) * (Term::arg(1) + Term::lit(1e-5)).sqrt().recip(),
            )],
        })))
        .unwrap();
    graph.push(gemv(["W", "xn", "y"], [4, 3, 5], m, n)).unwrap();
    assert_sequential(
        &graph,
        &env(&[
            (0, data(1, n)),
            (1, vec![0.0; n as usize]),
            (2, vec![0.0]),
            (3, vec![0.0; n as usize]),
            (4, data(2, m * n)),
            (5, vec![0.0; m as usize]),
        ]),
    );

    let structure = graph.structure();
    let [c] = &structure.contractions[..] else {
        panic!("one contraction: {structure}");
    };
    assert!(matches!(c.lhs, Term::Read { .. }));
    assert_eq!(c.rhs.reductions().len(), 1, "the norm is the rhs prologue");
    // The normalized input does not vary with the output row, and its scale does not
    // vary with the summed index either.
    let rows = c.rhs.invariants(&c.domain);
    assert!(rows.len() == 1 && rows[0].term.alpha_eq(&c.rhs), "{rows:?}");
    let scale = c.rhs.invariants(&[c.summed[0].0]);
    assert!(scale.len() == 1 && scale[0].free.is_empty(), "{scale:?}");
    // One grid of five rows cannot also store the mean.
    assert!(matches!(lower(graph.region(), &[]), Err(LowerError::Domain { .. })));
}
