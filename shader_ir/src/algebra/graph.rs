//! Named operations composed through the storage they share, and the composition seen
//! around its contractions.

use super::affine::{Affine, IndexVar};
use super::compose::ComposeError;
use super::op::{Factor, Op, OpError, OpKind};
use super::region::{Region, Role};
use super::term::{BinaryOp, ReduceOrder, Term, ValueId};
use std::collections::HashMap;
use std::fmt;

/// An input of a later operation that reads what an output of an earlier one stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    /// The producing operation and the position of the output among its outputs.
    pub from: (usize, usize),
    /// The consuming operation and the position of the input among its inputs.
    pub to: (usize, usize),
}

/// Why an operation does not join a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    Op(OpError),
    Compose(ComposeError),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::Op(e) => e.fmt(f),
            GraphError::Compose(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for GraphError {}

/// Operations in execution order, composed into one [`Region`].
///
/// Each operation is expanded with [`Op::expand`] and appended with [`Region::then`],
/// so an input that reads storage an earlier output writes reads that output's value;
/// every such read is an [`Edge`]. The operations stay available beside the region, so
/// a schedule can use what they state rather than recover it from terms.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    ops: Vec<Op>,
    region: Region,
    /// Per operation, the composed value of each input and each output.
    values: Vec<(Vec<ValueId>, Vec<ValueId>)>,
    edges: Vec<Edge>,
    /// The operation and output position of every output value.
    producers: HashMap<ValueId, (usize, usize)>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends `op`, which runs after every operation already here, and returns its
    /// position. On error the graph is unchanged.
    ///
    /// The region's index parameters and scalars are those of every operation, in
    /// order.
    pub fn push(&mut self, op: Op) -> Result<usize, GraphError> {
        let expanded = op.expand().map_err(GraphError::Op)?;
        let mut region = self.region.clone();
        let appended = region.then(&expanded.region).map_err(GraphError::Compose)?;
        let at = |v: &ValueId| appended.values[v.0 as usize].expect("then appends every value");
        let k = self.ops.len();
        let inputs: Vec<ValueId> = expanded.inputs.iter().map(at).collect();
        let outputs: Vec<ValueId> = expanded.outputs.iter().map(at).collect();
        for &(input, writer) in &appended.forwards {
            let from = self.producers[&writer];
            let position = expanded
                .inputs
                .iter()
                .position(|&v| v == input)
                .expect("a forwarded value is an operand");
            self.edges.push(Edge {
                from,
                to: (k, position),
            });
        }
        for (n, &o) in outputs.iter().enumerate() {
            self.producers.insert(o, (k, n));
        }
        self.region = region;
        self.values.push((inputs, outputs));
        self.ops.push(op);
        Ok(k)
    }

    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    pub fn region(&self) -> &Region {
        &self.region
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// The composed values of operation `op`'s inputs and outputs.
    pub fn values(&self, op: usize) -> (&[ValueId], &[ValueId]) {
        let (inputs, outputs) = &self.values[op];
        (inputs, outputs)
    }

    /// The composition around its contractions; see [`Structure`].
    pub fn structure(&self) -> Structure {
        let contractions: Vec<(usize, ValueId)> = self
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| matches!(op.kind, OpKind::Contraction(_)))
            .map(|(k, _)| (k, self.values[k].1[0]))
            .collect();
        let mut region = self.region.clone();
        let substituted: Vec<ValueId> = region
            .values()
            .filter(|(id, v)| v.definition.is_some() && !contractions.iter().any(|&(_, c)| c == *id))
            .map(|(id, _)| id)
            .collect();
        for id in substituted {
            region.substitute(id).expect("a defined value has a definition");
        }
        region.eliminate_dead();
        let contractions = contractions
            .into_iter()
            .filter_map(|(op, value)| {
                let OpKind::Contraction(c) = &self.ops[op].kind else {
                    unreachable!("filtered to contractions")
                };
                let value_ref = region.value(value)?;
                let def = value_ref.definition.as_ref().expect("a contraction is defined");
                let mut summed = Vec::new();
                let mut order = ReduceOrder::Sequential;
                let mut body = &def.body;
                for _ in c.summed() {
                    let Term::Reduce {
                        index,
                        extent,
                        order: o,
                        body: inner,
                        ..
                    } = body
                    else {
                        unreachable!("a contraction expands to sums of a product")
                    };
                    summed.push((*index, *extent));
                    order = *o;
                    body = inner;
                }
                let Term::Binary {
                    op: BinaryOp::Mul,
                    lhs,
                    rhs,
                } = body
                else {
                    unreachable!("a contraction expands to sums of a product")
                };
                Some(Contracted {
                    op,
                    value,
                    shape: value_ref.shape.clone(),
                    domain: def.domain.clone(),
                    summed,
                    order,
                    lhs: (**lhs).clone(),
                    rhs: (**rhs).clone(),
                })
            })
            .collect();
        Structure { region, contractions }
    }
}

/// A composition around its contractions.
///
/// Every defined value other than a contraction's is substituted into its readers. A
/// contraction's factors are then its prologues: terms over what the graph reads. Every
/// other value reads the contractions it depends on: those are their epilogues. All of
/// it is exact; nothing is reassociated.
#[derive(Debug, Clone, PartialEq)]
pub struct Structure {
    pub region: Region,
    /// The live contractions, in execution order.
    pub contractions: Vec<Contracted>,
}

/// One contraction of a [`Structure`].
#[derive(Debug, Clone, PartialEq)]
pub struct Contracted {
    /// The operation it came from.
    pub op: usize,
    /// Its value in the structure's region.
    pub value: ValueId,
    pub shape: Vec<u32>,
    /// The index variable of each axis of the value.
    pub domain: Vec<IndexVar>,
    /// The summed index variables, outermost first, with their extents.
    pub summed: Vec<(IndexVar, u32)>,
    /// The association of the innermost sum.
    pub order: ReduceOrder,
    /// The factors, over `domain` and `summed`.
    pub lhs: Term,
    pub rhs: Term,
}

impl Contracted {
    pub fn factor(&self, factor: Factor) -> &Term {
        match factor {
            Factor::Lhs => &self.lhs,
            Factor::Rhs => &self.rhs,
        }
    }

    /// `factor` summed over this contraction's summed indices, so that two factors
    /// compare up to the names of those indices.
    fn closed(&self, term: Term) -> Term {
        self.summed
            .iter()
            .rev()
            .fold(term, |term, &(index, extent)| Term::sum(index, extent, term))
    }
}

/// Two contractions over the same summed extents with the same factor.
///
/// When their results have one shape, their output indices are identified axis by axis
/// first; so a factor that depends on the output index is shared only when it is the
/// same element for the same output position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedFactor {
    /// Positions in [`Structure::contractions`], and the factor of each.
    pub first: (usize, Factor),
    pub second: (usize, Factor),
}

impl Structure {
    /// Every pair of contractions that multiply the same factor.
    pub fn shared_factors(&self) -> Vec<SharedFactor> {
        let mut shared = Vec::new();
        for (a, x) in self.contractions.iter().enumerate() {
            for (b, y) in self.contractions.iter().enumerate().skip(a + 1) {
                if x.summed.iter().map(|s| s.1).ne(y.summed.iter().map(|s| s.1)) {
                    continue;
                }
                let align = |t: &Term| match x.shape == y.shape {
                    true => t.instantiate(
                        &|v| y.domain.iter().position(|&d| d == v).map(|p| Affine::from(x.domain[p])),
                        &HashMap::new(),
                    ),
                    false => t.clone(),
                };
                for fx in [Factor::Lhs, Factor::Rhs] {
                    for fy in [Factor::Lhs, Factor::Rhs] {
                        let lhs = x.closed(x.factor(fx).clone());
                        let rhs = y.closed(align(y.factor(fy)));
                        if lhs.alpha_eq(&rhs) {
                            shared.push(SharedFactor {
                                first: (a, fx),
                                second: (b, fy),
                            });
                        }
                    }
                }
            }
        }
        shared
    }

    /// The values that read contraction `c`: its epilogues.
    pub fn readers(&self, c: usize) -> Vec<ValueId> {
        let value = self.contractions[c].value;
        self.region
            .values()
            .filter(|(_, v)| {
                let mut reads = false;
                if let Some(def) = &v.definition {
                    def.body.for_each_read(&mut |r, _| reads |= r == value);
                }
                reads
            })
            .map(|(id, _)| id)
            .collect()
    }
}

impl fmt::Display for Factor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Factor::Lhs => "lhs",
            Factor::Rhs => "rhs",
        })
    }
}

/// One line per defined value, `contraction`, `output` or `temp`, then one per shared
/// factor.
impl fmt::Display for Structure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (id, value) in self.region.values() {
            let Some(def) = self.region.show_definition(id) else {
                continue;
            };
            let role = match value.role {
                _ if self.contractions.iter().any(|c| c.value == id) => "contraction",
                Role::Output => "output",
                Role::Temporary => "temp",
                Role::Input => unreachable!("an input has no definition"),
            };
            writeln!(f, "{role} {def}")?;
        }
        for s in self.shared_factors() {
            let name = |c: usize| {
                let value = self.contractions[c].value;
                self.region.value(value).map_or("", |v| v.name.as_str())
            };
            writeln!(
                f,
                "shared {} of {} and {} of {}",
                s.first.1,
                name(s.first.0),
                s.second.1,
                name(s.second.0)
            )?;
        }
        Ok(())
    }
}
