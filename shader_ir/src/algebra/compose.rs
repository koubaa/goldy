//! Sequential composition of regions through the storage they share.

use super::affine::{Affine, IndexParam, IndexVar, Sym};
use super::region::{Definition, Region, RegionError, Role, Storage, Value};
use super::term::{ScalarParam, Term, ValueId};
use std::fmt;

/// How the values of a region appended by [`Region::then`] appear in the result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Appended {
    /// The composed value each value of the appended region became, by its id.
    pub values: Vec<Option<ValueId>>,
    pub index_params: Vec<IndexParam>,
    pub scalars: Vec<ScalarParam>,
    /// Inputs of the appended region that now read a value the region defines rather
    /// than storage.
    pub forwarded: usize,
    /// Outputs the appended region overwrites element for element, now temporaries.
    pub covered: Vec<ValueId>,
}

/// Why two regions do not compose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComposeError {
    Invalid(RegionError),
    /// An input reads storage an output writes, but not as whole elements of exactly
    /// one output.
    Unforwardable {
        input: String,
    },
    /// Two outputs write overlapping storage other than exactly the same elements.
    OverlappingOutputs {
        output: String,
        other: String,
    },
}

impl fmt::Display for ComposeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComposeError::Invalid(e) => e.fmt(f),
            ComposeError::Unforwardable { input } => {
                write!(f, "`{input}` reads storage an earlier output only partly writes")
            }
            ComposeError::OverlappingOutputs { output, other } => {
                write!(f, "`{output}` writes storage `{other}` partly writes")
            }
        }
    }
}

impl std::error::Error for ComposeError {}

impl Storage {
    /// The inclusive range of elements an access over `shape` can touch, or `None`
    /// when `shape` is empty or `range` leaves a parameter unbounded.
    pub fn interval(&self, shape: &[u32], range: &dyn Fn(IndexParam) -> (i64, i64)) -> Option<(i64, i64)> {
        if shape.contains(&0) {
            return None;
        }
        let (mut lo, mut hi) = self.offset.bounds(&|sym| match sym {
            Sym::Param(p) => Some(range(p)),
            Sym::Index(_) => None,
        })?;
        for (&stride, &extent) in self.strides.iter().zip(shape) {
            let span = stride.saturating_mul(i64::from(extent) - 1);
            lo = lo.saturating_add(span.min(0));
            hi = hi.saturating_add(span.max(0));
        }
        Some((lo, hi))
    }

    /// Whether distinct indices within `shape` always touch distinct elements.
    ///
    /// Holds when the strides of the axes longer than one, largest first, each exceed
    /// the span of all smaller ones.
    pub fn injective(&self, shape: &[u32]) -> bool {
        let Some(axes) = self.long_axes(shape) else {
            return false;
        };
        let mut span = 0i64;
        for &a in axes.iter().rev() {
            if self.strides[a] <= span {
                return false;
            }
            span = span.saturating_add(self.strides[a].saturating_mul(i64::from(shape[a]) - 1));
        }
        true
    }

    /// Axes longer than one by decreasing stride, or `None` when one has a
    /// non-positive stride.
    fn long_axes(&self, shape: &[u32]) -> Option<Vec<usize>> {
        let mut axes: Vec<usize> = (0..shape.len()).filter(|&a| shape[a] > 1).collect();
        if axes.iter().any(|&a| self.strides[a] <= 0) {
            return None;
        }
        axes.sort_by_key(|&a| std::cmp::Reverse(self.strides[a]));
        Some(axes)
    }

    /// The index within `shape` whose element is `element`, as affine functions of the
    /// symbols `element` mentions, when one exists for every value of those symbols.
    ///
    /// `range` bounds every symbol of `element`. `None` when this storage is not
    /// injective, or the preimage is not affine or can leave `shape`.
    pub fn preimage(
        &self,
        shape: &[u32],
        element: &Affine,
        range: &dyn Fn(Sym) -> Option<(i64, i64)>,
    ) -> Option<Vec<Affine>> {
        if !self.injective(shape) {
            return None;
        }
        let axes = self.long_axes(shape)?;
        let mut index = vec![Affine::constant(0); shape.len()];
        let rest = element.clone() - self.offset.clone();
        for &(sym, coefficient) in rest.terms() {
            let &a = axes.iter().find(|&&a| coefficient % self.strides[a] == 0)?;
            index[a] = index[a].clone() + Affine::sym(sym) * (coefficient / self.strides[a]);
        }
        let mut constant = rest.constant_term();
        for &a in &axes {
            let q = constant.div_euclid(self.strides[a]);
            index[a] = index[a].clone() + q;
            constant -= q * self.strides[a];
        }
        if constant != 0 {
            return None;
        }
        for (at, &extent) in index.iter().zip(shape) {
            let (lo, hi) = at.bounds(range)?;
            if lo < 0 || hi >= i64::from(extent) {
                return None;
            }
        }
        Some(index)
    }

    fn rename(&self, rename: &dyn Fn(Sym) -> Sym) -> Storage {
        Storage {
            parcel: self.parcel,
            offset: self.offset.rename(rename),
            strides: self.strides.clone(),
        }
    }
}

impl Region {
    /// Appends `next`, which runs after this region, and returns where its values went.
    ///
    /// Both regions' inputs read storage as of their own entry, so an input of `next`
    /// that reads what an output here writes reads that output's value instead: a
    /// temporary that reindexes it, which [`Self::inline_temporaries`] then substitutes
    /// away. An output here that an output of `next` overwrites element for element is
    /// no longer observable and becomes a temporary. Anything else that shares written
    /// storage is an error, and this region is unchanged.
    pub fn then(&mut self, next: &Region) -> Result<Appended, ComposeError> {
        next.validate().map_err(ComposeError::Invalid)?;
        let mut out = self.clone();
        let vars = out.index_names.len() as u32;
        let params = out.index_params.len() as u32;
        let scalars = out.scalar_names.len() as u32;
        out.index_names.extend(next.index_names.iter().cloned());
        out.index_params.extend(next.index_params.iter().cloned());
        out.scalar_names.extend(next.scalar_names.iter().cloned());
        let sym = move |s: Sym| match s {
            Sym::Index(v) => Sym::Index(IndexVar(v.0 + vars)),
            Sym::Param(p) => Sym::Param(IndexParam(p.0 + params)),
        };
        let var = move |v: IndexVar| IndexVar(v.0 + vars);
        let scalar = move |s: ScalarParam| ScalarParam(s.0 + scalars);

        let mut appended = Appended {
            values: vec![None; next.values.len()],
            index_params: (0..next.index_params.len() as u32)
                .map(|p| IndexParam(p + params))
                .collect(),
            scalars: (0..next.scalar_names.len() as u32)
                .map(|s| ScalarParam(s + scalars))
                .collect(),
            forwarded: 0,
            covered: Vec::new(),
        };
        for (id, value) in next.values() {
            let storage = value.storage.as_ref().map(|s| s.rename(&sym));
            let composed = match value.role {
                Role::Input => {
                    let storage = storage.expect("validated");
                    match out.writers(&storage, &value.shape)[..] {
                        [] => out.find_input(&storage, &value.shape).unwrap_or_else(|| {
                            out.push(Value {
                                storage: Some(storage),
                                ..value.clone()
                            })
                        }),
                        [writer] => {
                            let forward =
                                out.forward(writer, &storage, value)
                                    .ok_or_else(|| ComposeError::Unforwardable {
                                        input: value.name.clone(),
                                    })?;
                            appended.forwarded += 1;
                            forward
                        }
                        _ => {
                            return Err(ComposeError::Unforwardable {
                                input: value.name.clone(),
                            })
                        }
                    }
                }
                Role::Temporary | Role::Output => {
                    if let Some(storage) = &storage {
                        for writer in out.writers(storage, &value.shape) {
                            let other = out.value(writer).expect("live");
                            if other.storage.as_ref() != Some(storage) || other.shape != value.shape {
                                return Err(ComposeError::OverlappingOutputs {
                                    output: value.name.clone(),
                                    other: other.name.clone(),
                                });
                            }
                            out.demote(writer);
                            appended.covered.push(writer);
                        }
                    }
                    let def = value.definition.as_ref().expect("validated");
                    let body = def.body.rename(&sym, &scalar, &|v| {
                        appended.values[v.0 as usize].expect("reads precede their reader")
                    });
                    out.push(Value {
                        name: value.name.clone(),
                        shape: value.shape.clone(),
                        role: value.role,
                        storage,
                        definition: Some(Definition {
                            domain: def.domain.iter().copied().map(var).collect(),
                            body,
                        }),
                    })
                }
            };
            appended.values[id.0 as usize] = Some(composed);
        }
        *self = out;
        Ok(appended)
    }

    /// Makes output `id` a temporary: its storage is no longer written.
    ///
    /// Returns whether `id` was an output.
    pub fn demote(&mut self, id: ValueId) -> bool {
        match self.values.get_mut(id.0 as usize).and_then(Option::as_mut) {
            Some(value) if value.role == Role::Output => {
                value.role = Role::Temporary;
                value.storage = None;
                true
            }
            _ => false,
        }
    }

    /// Outputs whose elements can overlap an access to `storage` over `shape`.
    pub(crate) fn writers(&self, storage: &Storage, shape: &[u32]) -> Vec<ValueId> {
        let range = |p: IndexParam| {
            let r = self.param_range(p);
            (r.start, r.end - 1)
        };
        let Some((lo, hi)) = storage.interval(shape, &range) else {
            return Vec::new();
        };
        self.values()
            .filter(|(_, v)| v.role == Role::Output)
            .filter(|(_, v)| {
                let s = v.storage.as_ref().expect("validated");
                s.parcel == storage.parcel && s.interval(&v.shape, &range).is_some_and(|(a, b)| a <= hi && lo <= b)
            })
            .map(|(id, _)| id)
            .collect()
    }

    fn find_input(&self, storage: &Storage, shape: &[u32]) -> Option<ValueId> {
        self.values()
            .find(|(_, v)| v.role == Role::Input && v.storage.as_ref() == Some(storage) && v.shape == shape)
            .map(|(id, _)| id)
    }

    /// A temporary equal to `input` (read through `storage`) read from output `writer`.
    fn forward(&mut self, writer: ValueId, storage: &Storage, input: &Value) -> Option<ValueId> {
        let domain: Vec<IndexVar> = (0..input.shape.len())
            .map(|a| self.index(&format!("{}{a}", input.name)))
            .collect();
        let at: Vec<Affine> = domain.iter().map(|&v| Affine::from(v)).collect();
        let element = storage.element(&at);
        let producer = self.value(writer).expect("live");
        let range = |s: Sym| match s {
            Sym::Index(v) => domain
                .iter()
                .position(|&d| d == v)
                .map(|a| (0, i64::from(input.shape[a]) - 1)),
            Sym::Param(p) => {
                let r = self.param_range(p);
                Some((r.start, r.end - 1))
            }
        };
        let index = producer
            .storage
            .as_ref()
            .expect("validated")
            .preimage(&producer.shape, &element, &range)?;
        Some(self.temporary(&input.name, &input.shape, &domain, Term::read(writer, index)))
    }
}
