//! [`Schedule::Matrix`]: the contractions of a two-dimensional domain on matrix units.
//!
//! A workgroup is one subgroup and owns one [`MATRIX_TILE`]-square tile of the domain.
//! For each contraction it stages blocks of the row factor and the column factor in
//! workgroup memory as f16, zero outside the domain and the summed extent, and
//! accumulates their products in f32. Then each element of the tile computes its
//! outputs as a [`Schedule::Threads`] thread would, reading the contractions from the
//! tile. Staging reads what the element threads would read, so the obligations
//! [`Prepared`] checks are the same.

use super::affine::{Affine, IndexVar};
use super::cost::{self, Estimate};
use super::graph::Graph;
use super::schedule::{
    assign, barrier, bin, cast, coord_name, field, float, int, let_float, let_int, var, IndexSource, LowerError,
    Lowered, Prepared, Resources, Schedule,
};
use super::term::{BinaryOp, ReduceOp, Term};
use crate::{BinOp, BuiltinFn, BuiltinMask, Expr, MatrixOp, ShaderKernel, SourceMap, Stmt, MATRIX_TILE};
use std::collections::HashMap;

/// One contraction the schedule runs on matrix units.
struct Tile {
    /// The contraction on the domain's coordinates, as the outputs read it.
    term: Term,
    index: IndexVar,
    extent: u32,
    /// The factor free of the column coordinate, and the one free of the row.
    row: Term,
    column: Term,
}

/// `graph`'s composition on [`Schedule::Matrix`] with subgroups of `subgroup`
/// threads, if its domain is two-dimensional and it has a contraction the schedule
/// applies to.
pub(super) fn lower(graph: &Graph, sources: &[IndexSource], subgroup: u32) -> Option<Lowered> {
    let prepared = Prepared::new(graph.region()).ok()?;
    let &[m, n] = prepared.single()?.extents.as_slice() else {
        return None;
    };
    let tiles = tiles(graph, &prepared, [m, n]);
    if tiles.is_empty() {
        return None;
    }
    emit(&prepared, &tiles, [m, n], subgroup, sources).ok()
}

/// The contractions over the whole domain with one summed index whose factors split
/// into a row factor and a column factor, that an output spanning the whole domain
/// reads outside any select and any other reduction.
fn tiles(graph: &Graph, prepared: &Prepared, extents: [u32; 2]) -> Vec<Tile> {
    let coords = &prepared.parts[0].coords;
    let [row, column] = [coords[0], coords[1]];
    let read = prepared.top_reductions(0, true, |_| true);
    let mut tiles = Vec::new();
    for c in graph.structure().contractions {
        let [(index, extent)] = c.summed[..] else {
            continue;
        };
        if c.shape.iter().copied().filter(|&e| e != 1).ne(extents) {
            continue;
        }
        let mut coords = coords.iter();
        let map: HashMap<IndexVar, Affine> = c
            .domain
            .iter()
            .zip(&c.shape)
            .map(|(&d, &e)| match e {
                1 => (d, Affine::constant(0)),
                _ => (d, Affine::from(*coords.next().expect("two axes of the domain"))),
            })
            .collect();
        let product = Term::binary(BinaryOp::Mul, c.lhs, c.rhs);
        let term = Term::reduce_in(ReduceOp::Sum, c.order, index, extent, product)
            .instantiate(&|v| map.get(&v).cloned(), &HashMap::new());
        if !read.iter().any(|r| r.alpha_eq(&term)) {
            continue;
        }
        let Term::Reduce { index, body, .. } = &term else {
            unreachable!("instantiated from a reduction");
        };
        let Term::Binary { lhs, rhs, .. } = &**body else {
            unreachable!("instantiated from a product");
        };
        let free = |t: &Term, v: IndexVar| t.free_indices().contains(&v);
        let (r, col) = if !free(lhs, column) && !free(rhs, row) {
            (lhs, rhs)
        } else if !free(lhs, row) && !free(rhs, column) {
            (rhs, lhs)
        } else {
            continue;
        };
        tiles.push(Tile {
            index: *index,
            extent,
            row: (**r).clone(),
            column: (**col).clone(),
            term,
        });
    }
    tiles
}

fn emit(
    prepared: &Prepared,
    tiles: &[Tile],
    [m, n]: [u32; 2],
    subgroup: u32,
    sources: &[IndexSource],
) -> Result<Lowered, LowerError> {
    let Resources {
        params,
        parcels,
        scalars,
    } = prepared.resources(sources)?;
    let t = i64::from(MATRIX_TILE);
    let area = MATRIX_TILE * MATRIX_TILE;
    let across = n.div_ceil(MATRIX_TILE);
    let groups = u64::from(m.div_ceil(MATRIX_TILE)) * u64::from(across);
    let elements = u64::from(m) * u64::from(n);
    if elements > i32::MAX as u64 || groups > 65_535 {
        return Err(LowerError::Grid { elements });
    }
    let [row, column] = [prepared.parts[0].coords[0], prepared.parts[0].coords[1]];
    let mut emit = prepared.emitter();
    let mut decls = Vec::new();
    let mut body = vec![
        let_int("local", cast(field(BuiltinFn::LocalId), "int")),
        let_int("tile", cast(field(BuiltinFn::WorkgroupId), "int")),
        let_int(
            "row0",
            bin(
                BinOp::Mul,
                bin(BinOp::Div, var("tile"), int(i64::from(across))?),
                int(t)?,
            ),
        ),
        let_int(
            "col0",
            bin(
                BinOp::Mul,
                bin(BinOp::Rem, var("tile"), int(i64::from(across))?),
                int(t)?,
            ),
        ),
    ];
    prepared.index_params(sources, &mut body);
    // The loop over this lane's elements of a tile.
    let elements_of = |e: &str, inner: Vec<Stmt>| -> Result<Vec<Stmt>, LowerError> {
        let mut inner = inner;
        inner.push(assign(e, bin(BinOp::Add, var(e), int(i64::from(subgroup))?)));
        Ok(vec![
            let_int(e, var("local")),
            Stmt::While {
                cond: bin(BinOp::Lt, var(e), int(i64::from(area))?),
                body: inner,
            },
        ])
    };

    let mut stored = Vec::new();
    for tile in tiles {
        let [a, b, c] = ["ma", "mb", "mc"].map(|p| emit.fresh(p));
        for (name, elem) in [(&a, "half"), (&b, "half"), (&c, "float")] {
            decls.push(Stmt::WorkgroupArray {
                name: name.clone(),
                elem: elem.into(),
                len: area,
            });
        }
        let acc = emit.fresh("acc");
        body.push(Stmt::Matrix(MatrixOp::Accumulator { name: acc.clone() }));
        let k0 = emit.fresh("k");
        body.push(let_int(&k0, int(0)?));
        let e = emit.fresh("e");
        let (r, q) = (emit.fresh("r"), emit.fresh("w"));
        let mut stage = vec![
            let_int(&r, bin(BinOp::Div, var(&e), int(t)?)),
            let_int(&q, bin(BinOp::Rem, var(&e), int(t)?)),
        ];
        // A[r, q] = row factor at (row0 + r, k0 + q); B[r, q] = column factor at
        // (k0 + r, col0 + q).
        let operands = [
            (&a, &tile.row, row, m, "row0", &r, &q),
            (&b, &tile.column, column, n, "col0", &q, &r),
        ];
        emit.depth += 1;
        for (array, factor, axis, bound, origin, along, summed) in operands {
            let at = bin(BinOp::Add, var(origin), var(along));
            let k = bin(BinOp::Add, var(&k0), var(summed));
            let inside = bin(
                BinOp::And,
                bin(BinOp::Lt, at.clone(), int(i64::from(bound))?),
                bin(BinOp::Lt, k.clone(), int(i64::from(tile.extent))?),
            );
            let x = emit.fresh("f");
            let (axis_name, index_name) = (emit.fresh("i"), emit.fresh("k"));
            let mut then_body = vec![let_int(&axis_name, at), let_int(&index_name, k)];
            let old_axis = emit.bind(axis, axis_name, bound);
            let old_index = emit.bind(tile.index, index_name, tile.extent);
            let value = emit.term(factor, &mut then_body);
            emit.unbind(tile.index, old_index);
            emit.unbind(axis, old_axis);
            then_body.push(assign(&x, value?));
            stage.push(let_float(&x, float(0.0)));
            stage.push(Stmt::If {
                cond: inside,
                then_body,
                else_body: None,
            });
            stage.push(Stmt::Assign {
                target: Expr::Index {
                    base: Box::new(var(array)),
                    index: Box::new(var(&e)),
                },
                value: cast(var(&x), "half"),
            });
        }
        emit.depth -= 1;
        let mut step = elements_of(&e, stage)?;
        step.extend([
            barrier(),
            Stmt::Matrix(MatrixOp::MulAdd { acc: acc.clone(), a, b }),
            barrier(),
            assign(&k0, bin(BinOp::Add, var(&k0), int(t)?)),
        ]);
        body.push(Stmt::While {
            cond: bin(BinOp::Lt, var(&k0), int(i64::from(tile.extent))?),
            body: step,
        });
        body.push(Stmt::Matrix(MatrixOp::Store { acc, dest: c.clone() }));
        stored.push(c);
    }
    body.push(barrier());

    let e = emit.fresh("e");
    let mut element = vec![
        let_int(
            &coord_name(row),
            bin(BinOp::Add, var("row0"), bin(BinOp::Div, var(&e), int(t)?)),
        ),
        let_int(
            &coord_name(column),
            bin(BinOp::Add, var("col0"), bin(BinOp::Rem, var(&e), int(t)?)),
        ),
    ];
    let mut values = Vec::new();
    for (tile, c) in tiles.iter().zip(&stored) {
        let v = emit.fresh("v");
        values.push(let_float(
            &v,
            Expr::Index {
                base: Box::new(var(c)),
                index: Box::new(var(&e)),
            },
        ));
        emit.cache.push((tile.term.clone(), v));
    }
    values.extend(prepared.outputs(0, &mut emit)?);
    element.push(Stmt::If {
        cond: bin(
            BinOp::And,
            bin(BinOp::Lt, var(&coord_name(row)), int(i64::from(m))?),
            bin(BinOp::Lt, var(&coord_name(column)), int(i64::from(n))?),
        ),
        then_body: values,
        else_body: None,
    });
    body.extend(elements_of(&e, element)?);

    let workgroup_bytes = tiles.len() as u32 * area * (2 + 2 + 4);
    decls.extend(body);
    // Each tile step stages both factors, a subgroup's lanes taking turns over the
    // tile, then multiplies; the epilogue reads the tiles.
    let (mut serial, mut loads, mut work) = (0, 0, 0);
    let mut macs = 0;
    let turns = u64::from(area.div_ceil(subgroup));
    for tile in tiles {
        let steps = u64::from(tile.extent.div_ceil(MATRIX_TILE));
        let stage = cost::serial(&tile.row, &[]).max(cost::serial(&tile.column, &[])) + 1;
        serial += steps * (turns * stage + 2);
        loads += steps * turns * cost::loads(&tile.row, &[]).max(cost::loads(&tile.column, &[]));
        work += groups * steps * u64::from(area) * (cost::ops(&tile.row, &[]) + cost::ops(&tile.column, &[]) + 2);
        macs += u64::from(m) * u64::from(n) * u64::from(tile.extent);
    }
    let computed: Vec<&Term> = tiles.iter().map(|t| &t.term).collect();
    let epilogue = prepared.estimate(0, None, &[], &computed);
    let estimate = Estimate {
        bytes: prepared.footprint(),
        serial: serial + turns * epilogue.serial,
        loads: loads + turns * epilogue.loads,
        work: work + epilogue.work,
        matrix: macs,
    };
    Ok(Lowered {
        kernel: ShaderKernel {
            name: "tensor_region".into(),
            workgroup_size: [subgroup, 1, 1],
            params,
            builtins: BuiltinMask {
                local_id: true,
                workgroup_id: true,
                ..BuiltinMask::NONE
            },
            body: decls,
            source_map: SourceMap::default(),
            type_decls: Vec::new(),
        },
        schedule: Schedule::Matrix { subgroup },
        groups: [groups as u32, 1, 1],
        parcels,
        scalars,
        workgroup_bytes,
        parts: 1,
        estimate,
    })
}
