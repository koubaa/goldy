//! Named tensor operations: what a recorded operation computes, stated as tensor
//! algebra.
//!
//! An [`Op`] keeps the structure its recorder knows. A [`Contraction`] labels its
//! operands' axes in Einstein notation, so which axes are batched, free or summed is
//! recorded rather than recovered from a scalar term. [`Op::expand`] states the same
//! computation as a [`Region`], deterministically; composition, the interpreter and
//! lowering work on that region.

use super::affine::{IndexParam, IndexVar};
use super::region::{ParcelId, Region, RegionError, Storage};
use super::term::{ReduceOp, ReduceOrder, ScalarParam, Term, ValueId};
use std::fmt;
use std::ops::Range;

/// A tensor an operation reads or writes: its shape and where its elements live.
#[derive(Debug, Clone, PartialEq)]
pub struct Operand {
    pub name: String,
    pub shape: Vec<u32>,
    pub storage: Storage,
}

impl Operand {
    pub fn new(name: &str, shape: &[u32], storage: Storage) -> Self {
        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            storage,
        }
    }
}

/// The index and scalar parameters an operation names, by position.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Params {
    index: Vec<(String, Range<i64>)>,
    scalars: Vec<String>,
}

impl Params {
    /// A runtime integer in `range`, for storage offsets.
    pub fn index_param(&mut self, name: &str, range: Range<i64>) -> IndexParam {
        self.index.push((name.to_string(), range));
        IndexParam(self.index.len() as u32 - 1)
    }

    /// A runtime f32, for element-wise terms.
    pub fn scalar(&mut self, name: &str) -> ScalarParam {
        self.scalars.push(name.to_string());
        ScalarParam(self.scalars.len() as u32 - 1)
    }
}

/// Element-wise over one domain: `outputs[o][i] = body_o` for every index `i` of `shape`.
///
/// A body is an element-wise term over [`Term::arg`]s, argument `n` being `inputs[n]`
/// at `i`, and the operation's scalars. Every input and output has `shape`; a
/// broadcast is a zero stride. Inputs are read as of entry, so an output may overwrite
/// an input in place.
#[derive(Debug, Clone, PartialEq)]
pub struct Map {
    pub shape: Vec<u32>,
    pub inputs: Vec<Operand>,
    pub outputs: Vec<(Operand, Term)>,
}

/// `out = finish(op over axis of input)`: a reduction along one axis, which `out`
/// keeps with extent one.
///
/// `finish` is an element-wise term over [`Term::arg`]`(0)`, the reduced value. It
/// states what the realization does with its accumulator, such as the value a kernel
/// folds from, or a mean's division.
#[derive(Debug, Clone, PartialEq)]
pub struct Reduction {
    pub op: ReduceOp,
    pub order: ReduceOrder,
    pub axis: usize,
    pub input: Operand,
    pub out: Operand,
    pub finish: Term,
}

/// One factor of a [`Contraction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Factor {
    Lhs,
    Rhs,
}

/// A sum of products in Einstein notation, `out[o] = Σ_s lhs[..] · rhs[..]`.
///
/// Every axis of every operand carries a label, whose extent is the same wherever it
/// appears. A label of `out` is a batch label when both factors carry it, and a free
/// label of the one factor that does. A label both factors carry and `out` lacks is
/// summed, outermost first by label, in `order`. Each term is `lhs · rhs`, in that
/// order.
#[derive(Debug, Clone, PartialEq)]
pub struct Contraction {
    /// The extent of each label.
    pub extents: Vec<u32>,
    pub lhs: Operand,
    pub rhs: Operand,
    pub out: Operand,
    /// The labels of the axes of `lhs`, `rhs` and `out`.
    pub labels: [Vec<usize>; 3],
    /// The association of the sum; [`ReduceOrder::Lanes`] needs exactly one summed
    /// label.
    pub order: ReduceOrder,
}

impl Contraction {
    fn carries(&self, factor: Factor, label: usize) -> bool {
        self.labels[factor as usize].contains(&label)
    }

    /// Labels summed over, ascending.
    pub fn summed(&self) -> Vec<usize> {
        (0..self.extents.len())
            .filter(|&l| !self.labels[2].contains(&l) && self.carries(Factor::Lhs, l) && self.carries(Factor::Rhs, l))
            .collect()
    }

    /// Labels of `out` both factors carry, in `out`'s order.
    pub fn batch(&self) -> Vec<usize> {
        self.labels[2]
            .iter()
            .copied()
            .filter(|&l| self.carries(Factor::Lhs, l) && self.carries(Factor::Rhs, l))
            .collect()
    }

    /// Labels of `out` only `factor` carries, in `out`'s order.
    pub fn free(&self, factor: Factor) -> Vec<usize> {
        let other = match factor {
            Factor::Lhs => Factor::Rhs,
            Factor::Rhs => Factor::Lhs,
        };
        self.labels[2]
            .iter()
            .copied()
            .filter(|&l| self.carries(factor, l) && !self.carries(other, l))
            .collect()
    }

    fn validate(&self) -> Result<(), OpError> {
        for (operand, labels) in [&self.lhs, &self.rhs, &self.out].into_iter().zip(&self.labels) {
            let fits = labels.len() == operand.shape.len()
                && labels
                    .iter()
                    .zip(&operand.shape)
                    .all(|(&l, &e)| self.extents.get(l) == Some(&e));
            let distinct = labels.iter().enumerate().all(|(a, l)| !labels[..a].contains(l));
            if !fits || !distinct {
                return Err(OpError::Shape {
                    operand: operand.name.clone(),
                });
            }
        }
        let used = |l: usize| self.labels.iter().any(|labels| labels.contains(&l));
        let summed = self.summed();
        let product = (0..self.extents.len()).all(|l| used(l) && (self.labels[2].contains(&l) || summed.contains(&l)));
        let outputs_carried = self.labels[2]
            .iter()
            .all(|&l| self.carries(Factor::Lhs, l) || self.carries(Factor::Rhs, l));
        if !product || !outputs_carried || summed.is_empty() {
            return Err(OpError::Labels);
        }
        if matches!(self.order, ReduceOrder::Lanes { .. }) && summed.len() != 1 {
            return Err(OpError::Order);
        }
        Ok(())
    }

    fn expand(&self, region: &mut Region) -> Result<(Vec<ValueId>, Vec<ValueId>), OpError> {
        self.validate()?;
        let vars: Vec<IndexVar> = (0..self.extents.len())
            .map(|l| {
                let kind = if self.labels[2].contains(&l) { "i" } else { "s" };
                region.index(&format!("{kind}{l}"))
            })
            .collect();
        let lhs = input(region, &self.lhs);
        let rhs = input(region, &self.rhs);
        let read = |value: ValueId, labels: &[usize]| Term::read(value, labels.iter().map(|&l| vars[l]));
        let summed = self.summed();
        let order = match summed.len() {
            1 => self.order,
            _ => ReduceOrder::Sequential,
        };
        let body = summed
            .iter()
            .rev()
            .fold(read(lhs, &self.labels[0]) * read(rhs, &self.labels[1]), |body, &l| {
                Term::reduce_in(ReduceOp::Sum, order, vars[l], self.extents[l], body)
            });
        let domain: Vec<IndexVar> = self.labels[2].iter().map(|&l| vars[l]).collect();
        let out = output(region, &self.out, &domain, body);
        Ok((vec![lhs, rhs], vec![out]))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OpKind {
    Map(Map),
    Reduction(Reduction),
    Contraction(Contraction),
}

/// One named tensor operation over operands in storage.
#[derive(Debug, Clone, PartialEq)]
pub struct Op {
    pub params: Params,
    pub kind: OpKind,
}

/// An operation in index notation: a region, and the value of each of its operands.
#[derive(Debug, Clone, PartialEq)]
pub struct Expanded {
    pub region: Region,
    /// The value of each input operand, in [`Op::inputs`] order.
    pub inputs: Vec<ValueId>,
    /// The value of each output operand, in [`Op::outputs`] order.
    pub outputs: Vec<ValueId>,
}

/// Why an operation is ill-formed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpError {
    /// An operand's shape or labels disagree with the operation.
    Shape {
        operand: String,
    },
    /// A contraction's labels do not state a sum of products over at least one label.
    Labels,
    /// An element-wise term uses something other than element-wise arithmetic over
    /// the operation's arguments and scalars.
    Body,
    /// A lane order on a contraction summing other than one label.
    Order,
    Region(RegionError),
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpError::Shape { operand } => write!(f, "the shape of `{operand}` does not fit the operation"),
            OpError::Labels => f.write_str("the labels do not state a sum of products"),
            OpError::Body => f.write_str("an element-wise term uses more than its arguments and scalars"),
            OpError::Order => f.write_str("a lane order needs exactly one summed label"),
            OpError::Region(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for OpError {}

impl Op {
    pub fn new(params: Params, kind: OpKind) -> Self {
        Self { params, kind }
    }

    /// The operands read, in order.
    pub fn inputs(&self) -> Vec<&Operand> {
        match &self.kind {
            OpKind::Map(m) => m.inputs.iter().collect(),
            OpKind::Reduction(r) => vec![&r.input],
            OpKind::Contraction(c) => vec![&c.lhs, &c.rhs],
        }
    }

    /// The operands stored, in order.
    pub fn outputs(&self) -> Vec<&Operand> {
        match &self.kind {
            OpKind::Map(m) => m.outputs.iter().map(|(o, _)| o).collect(),
            OpKind::Reduction(r) => vec![&r.out],
            OpKind::Contraction(c) => vec![&c.out],
        }
    }

    fn operands_mut(&mut self) -> Vec<&mut Operand> {
        match &mut self.kind {
            OpKind::Map(m) => m
                .inputs
                .iter_mut()
                .chain(m.outputs.iter_mut().map(|(o, _)| o))
                .collect(),
            OpKind::Reduction(r) => vec![&mut r.input, &mut r.out],
            OpKind::Contraction(c) => vec![&mut c.lhs, &mut c.rhs, &mut c.out],
        }
    }

    /// Rebinds every operand to the parcel `parcel` maps its parcel to.
    pub fn map_parcels(&mut self, parcel: impl Fn(ParcelId) -> ParcelId) {
        for operand in self.operands_mut() {
            operand.storage.parcel = parcel(operand.storage.parcel);
        }
    }

    /// Renames every operand to `name` of it.
    pub fn rename_operands(&mut self, name: impl Fn(&Operand) -> String) {
        for operand in self.operands_mut() {
            operand.name = name(operand);
        }
    }

    /// The computation in index notation. Index parameter `n` and scalar `n` of the
    /// region are those of [`Self::params`].
    pub fn expand(&self) -> Result<Expanded, OpError> {
        let mut region = Region::new();
        for (name, range) in &self.params.index {
            region.index_param(name, range.clone());
        }
        for name in &self.params.scalars {
            region.scalar(name);
        }
        let scalars = self.params.scalars.len();
        let (inputs, outputs) = match &self.kind {
            OpKind::Map(m) => {
                let at: Vec<IndexVar> = (0..m.shape.len()).map(|a| region.index(&format!("i{a}"))).collect();
                let mut inputs = Vec::with_capacity(m.inputs.len());
                for operand in &m.inputs {
                    same_shape(operand, &m.shape)?;
                    inputs.push(input(&mut region, operand));
                }
                let mut outputs = Vec::with_capacity(m.outputs.len());
                for (operand, body) in &m.outputs {
                    same_shape(operand, &m.shape)?;
                    let body = element_wise(
                        body,
                        &|n| Some(Term::read(*inputs.get(n)?, at.iter().copied())),
                        scalars,
                    )?;
                    outputs.push(output(&mut region, operand, &at, body));
                }
                (inputs, outputs)
            }
            OpKind::Reduction(r) => {
                let shape = &r.input.shape;
                let extent = *shape.get(r.axis).ok_or_else(|| OpError::Shape {
                    operand: r.input.name.clone(),
                })?;
                let mut kept = shape.clone();
                kept[r.axis] = 1;
                same_shape(&r.out, &kept)?;
                let at: Vec<IndexVar> = (0..shape.len()).map(|a| region.index(&format!("i{a}"))).collect();
                let s = region.index("s");
                let x = input(&mut region, &r.input);
                let index = at.iter().enumerate().map(|(a, &i)| if a == r.axis { s } else { i });
                let reduced = Term::reduce_in(r.op, r.order, s, extent, Term::read(x, index));
                let body = element_wise(&r.finish, &|n| (n == 0).then(|| reduced.clone()), scalars)?;
                let out = output(&mut region, &r.out, &at, body);
                (vec![x], vec![out])
            }
            OpKind::Contraction(c) => c.expand(&mut region)?,
        };
        region.validate().map_err(OpError::Region)?;
        Ok(Expanded {
            region,
            inputs,
            outputs,
        })
    }
}

fn same_shape(operand: &Operand, shape: &[u32]) -> Result<(), OpError> {
    match operand.shape == shape && operand.storage.strides.len() == shape.len() {
        true => Ok(()),
        false => Err(OpError::Shape {
            operand: operand.name.clone(),
        }),
    }
}

fn input(region: &mut Region, operand: &Operand) -> ValueId {
    region.input(&operand.name, &operand.shape, operand.storage.clone())
}

fn output(region: &mut Region, operand: &Operand, domain: &[IndexVar], body: Term) -> ValueId {
    region.output(&operand.name, &operand.shape, domain, body, operand.storage.clone())
}

/// `body` with argument `n` replaced by `arg(n)`, if it is element-wise arithmetic over
/// arguments `arg` accepts and the first `scalars` scalars.
fn element_wise(body: &Term, arg: &dyn Fn(usize) -> Option<Term>, scalars: usize) -> Result<Term, OpError> {
    fn check(t: &Term, scalars: usize) -> bool {
        match t {
            Term::Lit(_) => true,
            Term::Scalar(s) => s.index() < scalars,
            Term::Read { index, .. } => index.is_empty(),
            Term::Unary { arg, .. } => check(arg, scalars),
            Term::Binary { lhs, rhs, .. } => check(lhs, scalars) && check(rhs, scalars),
            Term::IndexValue(_) | Term::Select { .. } | Term::Reduce { .. } => false,
        }
    }
    if !check(body, scalars) {
        return Err(OpError::Body);
    }
    let mut missing = false;
    let term = body.clone().map_reads(&mut |value, _| {
        arg(value.0 as usize).unwrap_or_else(|| {
            missing = true;
            Term::Lit(0.0)
        })
    });
    match missing {
        true => Err(OpError::Body),
        false => Ok(term),
    }
}
