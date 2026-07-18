//! Python wrappers for [`goldy::Transaction`] and [`goldy::Claim`].

use crate::error::IntoPyResult;
use crate::scheme::PySchemeSubmission;
use goldy::{Claim, Transaction};
use pyo3::prelude::*;

/// Erased exchange transaction recorded in a scheme.
#[pyclass(name = "Transaction", module = "goldy", unsendable)]
pub struct PyTransaction {
    pub(crate) inner: Transaction,
}

#[pymethods]
impl PyTransaction {
    fn binding_id(&self) -> u32 {
        self.inner.binding_id()
    }

    fn generation(&self) -> u64 {
        self.inner.generation()
    }

    fn claim(&self, submission: &mut PySchemeSubmission) -> PyResult<PyClaim> {
        let claim = self.inner.claim(&mut submission.inner).into_py_result()?;
        Ok(PyClaim { inner: Some(claim) })
    }

    fn __repr__(&self) -> String {
        format!(
            "Transaction(binding_id={}, generation={})",
            self.inner.binding_id(),
            self.inner.generation()
        )
    }
}

/// One submission's claim extracted from a transaction.
#[pyclass(name = "Claim", module = "goldy", unsendable)]
pub struct PyClaim {
    pub(crate) inner: Option<Claim>,
}

#[pymethods]
impl PyClaim {
    fn consume(&mut self) -> PyResult<()> {
        let claim = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("claim already settled"))?;
        claim.consume().into_py_result()
    }

    fn discard(&mut self) -> PyResult<()> {
        let claim = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("claim already settled"))?;
        claim.discard().into_py_result()
    }

    fn __repr__(&self) -> String {
        format!("Claim(settled={})", self.inner.is_none())
    }
}

impl Drop for PyClaim {
    fn drop(&mut self) {
        if let Some(claim) = self.inner.take() {
            let _ = claim.discard();
        }
    }
}
