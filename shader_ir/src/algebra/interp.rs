//! Reference semantics of a region, evaluated on the host.

use super::affine::{Affine, IndexParam, Sym};
use super::region::{ParcelId, Region, RegionError, Role, Storage};
use super::term::{BinaryOp, ScalarParam, Term, UnaryOp};
use std::collections::BTreeMap;
use std::fmt;

/// Parcel contents and parameter values for [`Region::evaluate`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Environment {
    pub parcels: BTreeMap<ParcelId, Vec<f32>>,
    pub index_params: BTreeMap<IndexParam, i64>,
    pub scalars: BTreeMap<ScalarParam, f32>,
}

impl Environment {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_parcel(mut self, parcel: ParcelId, data: Vec<f32>) -> Self {
        self.parcels.insert(parcel, data);
        self
    }

    pub fn with_index_param(mut self, param: IndexParam, value: i64) -> Self {
        self.index_params.insert(param, value);
        self
    }

    pub fn with_scalar(mut self, param: ScalarParam, value: f32) -> Self {
        self.scalars.insert(param, value);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    Invalid(RegionError),
    MissingParcel {
        value: String,
    },
    MissingIndexParam {
        param: String,
    },
    MissingScalar {
        param: String,
    },
    ParamOutOfRange {
        param: String,
        value: i64,
    },
    /// An access through `value`'s storage falls outside its parcel.
    StorageOutOfBounds {
        value: String,
        element: i64,
        len: usize,
    },
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalError::Invalid(e) => e.fmt(f),
            EvalError::MissingParcel { value } => write!(f, "no parcel is bound for `{value}`"),
            EvalError::MissingIndexParam { param } => write!(f, "index parameter `{param}` has no value"),
            EvalError::MissingScalar { param } => write!(f, "scalar parameter `{param}` has no value"),
            EvalError::ParamOutOfRange { param, value } => {
                write!(f, "index parameter `{param}` = {value} is outside its declared range")
            }
            EvalError::StorageOutOfBounds { value, element, len } => {
                write!(f, "`{value}` touches element {element} of a parcel of {len}")
            }
        }
    }
}

impl std::error::Error for EvalError {}

impl Region {
    /// Evaluates the region on the host, in place on `env.parcels`.
    ///
    /// Every read of an input sees its parcel as of entry. Every output is stored at
    /// exit, so an output sharing storage with an input does not affect reads of it.
    /// Each reduction combines its terms one by one in ascending index order,
    /// starting from its identity. Parcel elements no output covers are unchanged.
    pub fn evaluate(&self, env: &mut Environment) -> Result<(), EvalError> {
        self.validate().map_err(EvalError::Invalid)?;
        let params = self
            .index_params()
            .map(|p| {
                let name = self.param_name(p).to_string();
                let value = *env
                    .index_params
                    .get(&p)
                    .ok_or_else(|| EvalError::MissingIndexParam { param: name.clone() })?;
                if !self.param_range(p).contains(&value) {
                    return Err(EvalError::ParamOutOfRange { param: name, value });
                }
                Ok(value)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let scalars = self
            .scalars()
            .map(|s| {
                env.scalars.get(&s).copied().ok_or_else(|| EvalError::MissingScalar {
                    param: self.scalar_name(s).to_string(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut eval = Eval {
            region: self,
            entry: &env.parcels,
            dense: vec![None; self.values.len()],
            params: &params,
            scalars: &scalars,
            idx: vec![0; self.index_names.len()],
        };
        for (id, value) in self.values() {
            let Some(def) = &value.definition else {
                continue;
            };
            let mut data = Vec::with_capacity(numel(&value.shape));
            for linear in 0..numel(&value.shape) {
                for (&v, coord) in def.domain.iter().zip(coords(&value.shape, linear)) {
                    eval.idx[v.0 as usize] = coord;
                }
                data.push(eval.term(&def.body)?);
            }
            eval.dense[id.0 as usize] = Some(data);
        }

        let mut stores = Vec::new();
        for (id, value) in self.values().filter(|(_, v)| v.role == Role::Output) {
            let storage = value.storage.as_ref().expect("validated");
            let data = eval.dense[id.0 as usize].as_ref().expect("evaluated above");
            for (linear, &x) in data.iter().enumerate() {
                let at = element(storage, &coords(&value.shape, linear), &params);
                stores.push((storage.parcel, &value.name, at, x));
            }
        }
        for (parcel, name, at, x) in stores {
            let data = env
                .parcels
                .get_mut(&parcel)
                .ok_or_else(|| EvalError::MissingParcel { value: name.clone() })?;
            let len = data.len();
            let slot = usize::try_from(at)
                .ok()
                .and_then(|at| data.get_mut(at))
                .ok_or_else(|| EvalError::StorageOutOfBounds {
                    value: name.clone(),
                    element: at,
                    len,
                })?;
            *slot = x;
        }
        Ok(())
    }
}

fn numel(shape: &[u32]) -> usize {
    shape.iter().map(|&d| d as usize).product()
}

/// Row-major coordinates of element `linear`.
fn coords(shape: &[u32], mut linear: usize) -> Vec<i64> {
    let mut out = vec![0; shape.len()];
    for (coord, &extent) in out.iter_mut().zip(shape).rev() {
        *coord = (linear % extent as usize) as i64;
        linear /= extent as usize;
    }
    out
}

fn element(storage: &Storage, coords: &[i64], params: &[i64]) -> i64 {
    let offset = storage.offset.eval(&|sym| match sym {
        Sym::Param(p) => params[p.0 as usize],
        Sym::Index(_) => unreachable!("validated: storage offsets mention no index"),
    });
    coords
        .iter()
        .zip(&storage.strides)
        .fold(offset, |acc, (&c, &stride)| acc + c * stride)
}

struct Eval<'a> {
    region: &'a Region,
    entry: &'a BTreeMap<ParcelId, Vec<f32>>,
    dense: Vec<Option<Vec<f32>>>,
    params: &'a [i64],
    scalars: &'a [f32],
    idx: Vec<i64>,
}

impl Eval<'_> {
    fn affine(&self, a: &Affine) -> i64 {
        a.eval(&|sym| match sym {
            Sym::Index(v) => self.idx[v.0 as usize],
            Sym::Param(p) => self.params[p.0 as usize],
        })
    }

    fn term(&mut self, term: &Term) -> Result<f32, EvalError> {
        Ok(match term {
            Term::Lit(v) => *v,
            Term::Scalar(s) => self.scalars[s.0 as usize],
            Term::IndexValue(a) => self.affine(a) as f32,
            Term::Read { value, index } => {
                let target = self.region.value(*value).expect("validated");
                let coords: Vec<i64> = index.iter().map(|a| self.affine(a)).collect();
                match &self.dense[value.0 as usize] {
                    Some(data) => {
                        let linear = coords
                            .iter()
                            .zip(&target.shape)
                            .fold(0i64, |acc, (&c, &extent)| acc * i64::from(extent) + c);
                        data[linear as usize]
                    }
                    None => {
                        let storage = target.storage.as_ref().expect("validated");
                        let at = element(storage, &coords, self.params);
                        let data = self
                            .entry
                            .get(&storage.parcel)
                            .ok_or_else(|| EvalError::MissingParcel {
                                value: target.name.clone(),
                            })?;
                        *usize::try_from(at).ok().and_then(|at| data.get(at)).ok_or_else(|| {
                            EvalError::StorageOutOfBounds {
                                value: target.name.clone(),
                                element: at,
                                len: data.len(),
                            }
                        })?
                    }
                }
            }
            Term::Unary { op, arg } => {
                let x = self.term(arg)?;
                match op {
                    UnaryOp::Neg => -x,
                    UnaryOp::Abs => x.abs(),
                    UnaryOp::Exp => x.exp(),
                    UnaryOp::Log => x.ln(),
                    UnaryOp::Sqrt => x.sqrt(),
                    UnaryOp::Recip => 1.0 / x,
                    UnaryOp::Sin => x.sin(),
                    UnaryOp::Cos => x.cos(),
                }
            }
            Term::Binary { op, lhs, rhs } => {
                let (a, b) = (self.term(lhs)?, self.term(rhs)?);
                match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::Div => a / b,
                    BinaryOp::Min => a.min(b),
                    BinaryOp::Max => a.max(b),
                }
            }
            Term::Select {
                lhs,
                cmp,
                rhs,
                then,
                otherwise,
            } => {
                if cmp.holds(self.affine(lhs), self.affine(rhs)) {
                    self.term(then)?
                } else {
                    self.term(otherwise)?
                }
            }
            Term::Reduce {
                op,
                index,
                extent,
                body,
            } => {
                let mut acc = op.identity();
                for k in 0..*extent {
                    self.idx[index.0 as usize] = i64::from(k);
                    acc = op.combine(acc, self.term(body)?);
                }
                acc
            }
        })
    }
}
