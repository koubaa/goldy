//! Semantic regions: tensor values, their definitions and their storage.

use super::affine::{Affine, IndexParam, IndexVar, Sym};
use super::term::{BinaryOp, CmpOp, ReduceOp, ScalarParam, Term, UnaryOp, ValueId};
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

/// A parcel a region reads or writes. The algebra treats it as an opaque name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParcelId(pub u32);

/// Where a value's elements live: element `offset + Σ strides[a]·index[a]` of `parcel`.
///
/// The offset may depend on index parameters, never on index variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Storage {
    pub parcel: ParcelId,
    pub offset: Affine,
    pub strides: Vec<i64>,
}

impl Storage {
    /// Packed row-major storage from element 0.
    pub fn packed(parcel: ParcelId, shape: &[u32]) -> Self {
        let mut strides = vec![0i64; shape.len()];
        let mut acc = 1i64;
        for (stride, &extent) in strides.iter_mut().zip(shape).rev() {
            *stride = acc;
            acc *= i64::from(extent);
        }
        Self {
            parcel,
            offset: Affine::constant(0),
            strides,
        }
    }

    pub fn strided(parcel: ParcelId, offset: impl Into<Affine>, strides: &[i64]) -> Self {
        Self {
            parcel,
            offset: offset.into(),
            strides: strides.to_vec(),
        }
    }

    /// The element an access at `index` touches.
    pub fn element(&self, index: &[Affine]) -> Affine {
        index
            .iter()
            .zip(&self.strides)
            .fold(self.offset.clone(), |acc, (a, &stride)| acc + a.clone() * stride)
    }
}

/// What a value is to everything outside the region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Read from storage as of region entry.
    Input,
    /// Defined here and observable: its final contents must reach its storage.
    Output,
    /// Defined here and scheme-local: it may be substituted away or never stored.
    Temporary,
}

/// `value[domain] = body`.
#[derive(Debug, Clone, PartialEq)]
pub struct Definition {
    /// One index variable per axis of the value.
    pub domain: Vec<IndexVar>,
    pub body: Term,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    pub name: String,
    pub shape: Vec<u32>,
    pub role: Role,
    /// Required for inputs and outputs. A temporary has none.
    pub storage: Option<Storage>,
    /// Present for outputs and temporaries.
    pub definition: Option<Definition>,
}

/// A set of tensor values in index notation.
///
/// A definition reads only values created before it, so creation order is a valid
/// evaluation order. Rewrites keep [`ValueId`]s stable; an eliminated value is gone
/// from [`Region::values`] but its id is never reused.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Region {
    pub(crate) index_names: Vec<String>,
    pub(crate) index_params: Vec<(String, Range<i64>)>,
    pub(crate) scalar_names: Vec<String>,
    pub(crate) values: Vec<Option<Value>>,
}

/// Why a region is not well formed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegionError {
    /// An input or output has no storage, or its strides do not match its rank.
    Storage { value: String },
    /// A storage offset mentions an index variable.
    StorageOffset { value: String },
    /// The domain does not name one index variable per axis.
    DomainRank { value: String },
    /// An index variable is bound where it is already bound.
    Rebound { value: String, index: String },
    /// An index variable is used where nothing binds it.
    Unbound { value: String, index: String },
    /// A read names a value that is eliminated, or not created before the reader.
    UndefinedRead { value: String },
    /// A read's index count differs from the rank of the value read.
    ReadRank { value: String, read: String },
    /// A read can fall outside the value read. `bounds` is the inclusive index range.
    OutOfBounds {
        value: String,
        read: String,
        axis: usize,
        bounds: (i64, i64),
        extent: u32,
    },
    /// An index parameter's declared range is empty.
    EmptyParamRange { param: String },
}

impl fmt::Display for RegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegionError::Storage { value } => write!(f, "`{value}` needs storage with one stride per axis"),
            RegionError::StorageOffset { value } => {
                write!(f, "the storage offset of `{value}` mentions an index variable")
            }
            RegionError::DomainRank { value } => write!(f, "the domain of `{value}` does not match its rank"),
            RegionError::Rebound { value, index } => write!(f, "`{value}` binds `{index}` where it is already bound"),
            RegionError::Unbound { value, index } => write!(f, "`{value}` uses `{index}` where nothing binds it"),
            RegionError::UndefinedRead { value } => {
                write!(f, "`{value}` reads a value that is eliminated or defined after it")
            }
            RegionError::ReadRank { value, read } => {
                write!(f, "`{value}` indexes `{read}` with the wrong number of indices")
            }
            RegionError::OutOfBounds {
                value,
                read,
                axis,
                bounds,
                extent,
            } => write!(
                f,
                "`{value}` reads `{read}` axis {axis} over {}..={}, outside 0..{extent}",
                bounds.0, bounds.1
            ),
            RegionError::EmptyParamRange { param } => write!(f, "index parameter `{param}` has an empty range"),
        }
    }
}

impl std::error::Error for RegionError {}

impl Region {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn index(&mut self, name: &str) -> IndexVar {
        self.index_names.push(name.to_string());
        IndexVar(self.index_names.len() as u32 - 1)
    }

    /// A runtime integer taking values in `range`.
    pub fn index_param(&mut self, name: &str, range: Range<i64>) -> IndexParam {
        self.index_params.push((name.to_string(), range));
        IndexParam(self.index_params.len() as u32 - 1)
    }

    pub fn scalar(&mut self, name: &str) -> ScalarParam {
        self.scalar_names.push(name.to_string());
        ScalarParam(self.scalar_names.len() as u32 - 1)
    }

    pub fn input(&mut self, name: &str, shape: &[u32], storage: Storage) -> ValueId {
        self.push(Value {
            name: name.to_string(),
            shape: shape.to_vec(),
            role: Role::Input,
            storage: Some(storage),
            definition: None,
        })
    }

    pub fn temporary(&mut self, name: &str, shape: &[u32], domain: &[IndexVar], body: Term) -> ValueId {
        self.push(Value {
            name: name.to_string(),
            shape: shape.to_vec(),
            role: Role::Temporary,
            storage: None,
            definition: Some(Definition {
                domain: domain.to_vec(),
                body,
            }),
        })
    }

    pub fn output(&mut self, name: &str, shape: &[u32], domain: &[IndexVar], body: Term, storage: Storage) -> ValueId {
        self.push(Value {
            name: name.to_string(),
            shape: shape.to_vec(),
            role: Role::Output,
            storage: Some(storage),
            definition: Some(Definition {
                domain: domain.to_vec(),
                body,
            }),
        })
    }

    fn push(&mut self, value: Value) -> ValueId {
        self.values.push(Some(value));
        ValueId(self.values.len() as u32 - 1)
    }

    /// `None` once the value has been eliminated.
    pub fn value(&self, id: ValueId) -> Option<&Value> {
        self.values.get(id.0 as usize).and_then(Option::as_ref)
    }

    /// Live values in creation order.
    pub fn values(&self) -> impl Iterator<Item = (ValueId, &Value)> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(at, v)| v.as_ref().map(|v| (ValueId(at as u32), v)))
    }

    /// Number of reads of `id` in live definitions.
    pub fn readers(&self, id: ValueId) -> usize {
        let mut count = 0;
        for (_, value) in self.values() {
            if let Some(def) = &value.definition {
                def.body.for_each_read(&mut |read, _| count += usize::from(read == id));
            }
        }
        count
    }

    pub fn index_name(&self, v: IndexVar) -> &str {
        &self.index_names[v.0 as usize]
    }

    pub fn param_name(&self, p: IndexParam) -> &str {
        &self.index_params[p.0 as usize].0
    }

    pub fn param_range(&self, p: IndexParam) -> Range<i64> {
        self.index_params[p.0 as usize].1.clone()
    }

    pub fn scalar_name(&self, s: ScalarParam) -> &str {
        &self.scalar_names[s.0 as usize]
    }

    pub(crate) fn index_params(&self) -> impl Iterator<Item = IndexParam> {
        (0..self.index_params.len() as u32).map(IndexParam)
    }

    pub(crate) fn scalars(&self) -> impl Iterator<Item = ScalarParam> {
        (0..self.scalar_names.len() as u32).map(ScalarParam)
    }

    /// A fresh index variable named after `like`.
    pub(crate) fn fresh_index(names: &mut Vec<String>, like: IndexVar) -> IndexVar {
        names.push(format!("{}'", names[like.0 as usize]));
        IndexVar(names.len() as u32 - 1)
    }

    fn sym_name(&self, sym: Sym) -> &str {
        match sym {
            Sym::Index(v) => self.index_name(v),
            Sym::Param(p) => self.param_name(p),
        }
    }

    /// Formats `term` with this region's names.
    pub fn show<'a>(&'a self, term: &'a Term) -> impl fmt::Display + 'a {
        Show { region: self, term }
    }

    /// Formats `id`'s definition as `name[domain] = body`, or `None` for inputs and
    /// eliminated values.
    pub fn show_definition(&self, id: ValueId) -> Option<String> {
        let value = self.value(id)?;
        let def = value.definition.as_ref()?;
        let mut out = value.name.clone();
        if !def.domain.is_empty() {
            let names: Vec<&str> = def.domain.iter().map(|&v| self.index_name(v)).collect();
            out += &format!("[{}]", names.join(", "));
        }
        Some(format!("{out} = {}", self.show(&def.body)))
    }

    /// Checks every well-formedness rule the rewrites and the interpreter rely on.
    pub fn validate(&self) -> Result<(), RegionError> {
        for (name, range) in &self.index_params {
            if range.is_empty() {
                return Err(RegionError::EmptyParamRange { param: name.clone() });
            }
        }
        for (id, value) in self.values() {
            if value.role != Role::Temporary {
                let storage = value
                    .storage
                    .as_ref()
                    .filter(|s| s.strides.len() == value.shape.len())
                    .ok_or_else(|| RegionError::Storage {
                        value: value.name.clone(),
                    })?;
                if storage.offset.indices().next().is_some() {
                    return Err(RegionError::StorageOffset {
                        value: value.name.clone(),
                    });
                }
            }
            let Some(def) = &value.definition else {
                continue;
            };
            if def.domain.len() != value.shape.len() {
                return Err(RegionError::DomainRank {
                    value: value.name.clone(),
                });
            }
            let mut check = Check {
                region: self,
                reader: id,
                name: &value.name,
                ranges: HashMap::new(),
                bound: Vec::new(),
            };
            for (&v, &extent) in def.domain.iter().zip(&value.shape) {
                check.bind(v)?;
                check.ranges.insert(Sym::Index(v), (0, i64::from(extent) - 1));
            }
            // An empty domain never evaluates its body.
            if !value.shape.contains(&0) {
                check.term(&def.body)?;
            }
        }
        Ok(())
    }
}

struct Check<'a> {
    region: &'a Region,
    reader: ValueId,
    name: &'a str,
    ranges: HashMap<Sym, (i64, i64)>,
    bound: Vec<IndexVar>,
}

impl Check<'_> {
    fn bind(&mut self, v: IndexVar) -> Result<(), RegionError> {
        if self.bound.contains(&v) {
            return Err(RegionError::Rebound {
                value: self.name.to_string(),
                index: self.region.index_name(v).to_string(),
            });
        }
        self.bound.push(v);
        Ok(())
    }

    fn range(&self, sym: Sym) -> Option<(i64, i64)> {
        match sym {
            Sym::Index(_) => self.ranges.get(&sym).copied(),
            Sym::Param(p) => {
                let r = self.region.param_range(p);
                Some((r.start, r.end - 1))
            }
        }
    }

    fn affine(&self, a: &Affine) -> Result<(i64, i64), RegionError> {
        if let Some(v) = a.indices().find(|v| !self.bound.contains(v)) {
            return Err(RegionError::Unbound {
                value: self.name.to_string(),
                index: self.region.index_name(v).to_string(),
            });
        }
        Ok(a.bounds(&|sym| self.range(sym)).expect("every bound index has a range"))
    }

    fn term(&mut self, term: &Term) -> Result<(), RegionError> {
        match term {
            Term::Lit(_) | Term::Scalar(_) => Ok(()),
            Term::IndexValue(a) => self.affine(a).map(drop),
            Term::Read { value, index } => self.read(*value, index),
            Term::Unary { arg, .. } => self.term(arg),
            Term::Binary { lhs, rhs, .. } => {
                self.term(lhs)?;
                self.term(rhs)
            }
            Term::Select {
                lhs,
                cmp,
                rhs,
                then,
                otherwise,
            } => {
                self.affine(lhs)?;
                self.affine(rhs)?;
                self.guarded(lhs, *cmp, rhs, true, then)?;
                self.guarded(lhs, *cmp, rhs, false, otherwise)
            }
            Term::Reduce {
                index, extent, body, ..
            } => {
                self.bind(*index)?;
                if *extent > 0 {
                    self.ranges.insert(Sym::Index(*index), (0, i64::from(*extent) - 1));
                    self.term(body)?;
                    self.ranges.remove(&Sym::Index(*index));
                }
                self.bound.pop();
                Ok(())
            }
        }
    }

    fn read(&self, value: ValueId, index: &[Affine]) -> Result<(), RegionError> {
        let target = self
            .region
            .value(value)
            .filter(|_| value < self.reader)
            .ok_or_else(|| RegionError::UndefinedRead {
                value: self.name.to_string(),
            })?;
        if index.len() != target.shape.len() {
            return Err(RegionError::ReadRank {
                value: self.name.to_string(),
                read: target.name.clone(),
            });
        }
        for (axis, (a, &extent)) in index.iter().zip(&target.shape).enumerate() {
            let bounds = self.affine(a)?;
            if bounds.0 < 0 || bounds.1 >= i64::from(extent) {
                return Err(RegionError::OutOfBounds {
                    value: self.name.to_string(),
                    read: target.name.clone(),
                    axis,
                    bounds,
                    extent,
                });
            }
        }
        Ok(())
    }

    /// Checks one branch of a select with the ranges its condition implies.
    fn guarded(
        &mut self,
        lhs: &Affine,
        cmp: CmpOp,
        rhs: &Affine,
        taken: bool,
        branch: &Term,
    ) -> Result<(), RegionError> {
        let Some((sym, lo, hi)) = refine(&(lhs.clone() - rhs.clone()), cmp, taken) else {
            return self.term(branch);
        };
        let Some(old) = self.range(sym) else {
            return self.term(branch);
        };
        let narrowed = (old.0.max(lo), old.1.min(hi));
        if narrowed.0 > narrowed.1 {
            // The branch is never taken.
            return Ok(());
        }
        let restore = self.ranges.insert(sym, narrowed);
        let result = self.term(branch);
        match restore {
            Some(r) => self.ranges.insert(sym, r),
            None => self.ranges.remove(&sym),
        };
        result
    }
}

/// The range of the single unit-coefficient symbol in `d` implied by `d cmp 0` holding
/// (`taken`) or failing. `None` when the condition does not have that shape.
fn refine(d: &Affine, cmp: CmpOp, taken: bool) -> Option<(Sym, i64, i64)> {
    let [(sym, coefficient)] = d.terms() else {
        return None;
    };
    // With `d = c·s + k` and `c = ±1`, `d < 0` is `s < -k` or `s > k`.
    let k = d.constant_term();
    let (bound, flipped) = match coefficient {
        1 => (-k, false),
        -1 => (k, true),
        _ => return None,
    };
    #[derive(Clone, Copy)]
    enum Rel {
        Lt,
        Le,
        Gt,
        Ge,
        Eq,
    }
    // The relation of `d` to zero on this branch.
    let rel = match (cmp, taken) {
        (CmpOp::Lt, true) => Rel::Lt,
        (CmpOp::Lt, false) => Rel::Ge,
        (CmpOp::Le, true) => Rel::Le,
        (CmpOp::Le, false) => Rel::Gt,
        (CmpOp::Eq, true) | (CmpOp::Ne, false) => Rel::Eq,
        (CmpOp::Eq, false) | (CmpOp::Ne, true) => return None,
    };
    let rel = match (rel, flipped) {
        (r, false) | (r @ Rel::Eq, true) => r,
        (Rel::Lt, true) => Rel::Gt,
        (Rel::Le, true) => Rel::Ge,
        (Rel::Gt, true) => Rel::Lt,
        (Rel::Ge, true) => Rel::Le,
    };
    let (lo, hi) = match rel {
        Rel::Lt => (i64::MIN, bound - 1),
        Rel::Le => (i64::MIN, bound),
        Rel::Gt => (bound + 1, i64::MAX),
        Rel::Ge => (bound, i64::MAX),
        Rel::Eq => (bound, bound),
    };
    Some((*sym, lo, hi))
}

struct Show<'a> {
    region: &'a Region,
    term: &'a Term,
}

impl fmt::Display for Show<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_term(self.region, self.term, 0, f)
    }
}

fn write_affine(region: &Region, a: &Affine, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    a.write(f, &|sym| region.sym_name(sym))
}

/// Writes `term`, parenthesized when it binds more loosely than `min`.
///
/// Every right operand needs a strictly tighter binding, so the printed form shows
/// the tree exactly: `a + (b + c)` keeps its parentheses.
fn write_term(region: &Region, term: &Term, min: u8, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    const ATOM: u8 = 4;
    let precedence = match term {
        Term::Binary {
            op: BinaryOp::Add | BinaryOp::Sub,
            ..
        } => 1,
        Term::Binary {
            op: BinaryOp::Mul | BinaryOp::Div,
            ..
        } => 2,
        Term::Unary { op: UnaryOp::Neg, .. } => 3,
        Term::Lit(v) if v.is_sign_negative() => 3,
        _ => ATOM,
    };
    if precedence < min {
        f.write_str("(")?;
    }
    match term {
        Term::Lit(v) => write!(f, "{v:?}")?,
        Term::Scalar(s) => f.write_str(region.scalar_name(*s))?,
        Term::IndexValue(a) => {
            f.write_str("f32(")?;
            write_affine(region, a, f)?;
            f.write_str(")")?;
        }
        Term::Read { value, index } => {
            let name = region.value(*value).map_or("<eliminated>", |v| v.name.as_str());
            f.write_str(name)?;
            if !index.is_empty() {
                f.write_str("[")?;
                for (at, a) in index.iter().enumerate() {
                    if at > 0 {
                        f.write_str(", ")?;
                    }
                    write_affine(region, a, f)?;
                }
                f.write_str("]")?;
            }
        }
        Term::Unary { op: UnaryOp::Neg, arg } => {
            f.write_str("-")?;
            write_term(region, arg, ATOM, f)?;
        }
        Term::Unary { op, arg } => {
            let name = match op {
                UnaryOp::Neg => unreachable!(),
                UnaryOp::Abs => "abs",
                UnaryOp::Exp => "exp",
                UnaryOp::Log => "log",
                UnaryOp::Sqrt => "sqrt",
                UnaryOp::Recip => "recip",
                UnaryOp::Sin => "sin",
                UnaryOp::Cos => "cos",
            };
            write!(f, "{name}(")?;
            write_term(region, arg, 0, f)?;
            f.write_str(")")?;
        }
        Term::Binary {
            op: op @ (BinaryOp::Min | BinaryOp::Max),
            lhs,
            rhs,
        } => {
            f.write_str(if *op == BinaryOp::Min { "min(" } else { "max(" })?;
            write_term(region, lhs, 0, f)?;
            f.write_str(", ")?;
            write_term(region, rhs, 0, f)?;
            f.write_str(")")?;
        }
        Term::Binary { op, lhs, rhs } => {
            let symbol = match op {
                BinaryOp::Add => " + ",
                BinaryOp::Sub => " - ",
                BinaryOp::Mul => " * ",
                BinaryOp::Div => " / ",
                BinaryOp::Min | BinaryOp::Max => unreachable!(),
            };
            write_term(region, lhs, precedence, f)?;
            f.write_str(symbol)?;
            write_term(region, rhs, precedence + 1, f)?;
        }
        Term::Select {
            lhs,
            cmp,
            rhs,
            then,
            otherwise,
        } => {
            f.write_str("(")?;
            write_affine(region, lhs, f)?;
            f.write_str(match cmp {
                CmpOp::Lt => " < ",
                CmpOp::Le => " <= ",
                CmpOp::Eq => " == ",
                CmpOp::Ne => " != ",
            })?;
            write_affine(region, rhs, f)?;
            f.write_str(" ? ")?;
            write_term(region, then, 0, f)?;
            f.write_str(" : ")?;
            write_term(region, otherwise, 0, f)?;
            f.write_str(")")?;
        }
        Term::Reduce {
            op,
            index,
            extent,
            body,
        } => {
            let name = match op {
                ReduceOp::Sum => "sum",
                ReduceOp::Max => "max",
                ReduceOp::Min => "min",
            };
            write!(f, "{name}{{{}<{extent}}}(", region.index_name(*index))?;
            write_term(region, body, 0, f)?;
            f.write_str(")")?;
        }
    }
    if precedence < min {
        f.write_str(")")?;
    }
    Ok(())
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (id, value) in self.values() {
            let role = match value.role {
                Role::Input => "input",
                Role::Output => "output",
                Role::Temporary => "temp",
            };
            match self.show_definition(id) {
                Some(def) => writeln!(f, "{role} {def}")?,
                None => writeln!(f, "{role} {}{:?}", value.name, value.shape)?,
            }
        }
        Ok(())
    }
}
