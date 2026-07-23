//! Python wrappers for [`goldy::MemoryExchange`] withdraw / deposit.

use crate::error::IntoPyResult;
use crate::parcel::PyParcel;
use crate::scheme::{PyContext, PyScheme, PySchemeSubmission};
use crate::texture::PyTexture;
use goldy::{DepositTransaction, MemoryExchange, WithdrawClaim, WithdrawTransaction};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// CPU↔GPU memory exchange: withdrawals (readback) and deposits (upload).
#[pyclass(name = "MemoryExchange", module = "goldy", unsendable)]
pub struct PyMemoryExchange {
    pub(crate) inner: MemoryExchange,
}

#[pymethods]
impl PyMemoryExchange {
    #[new]
    fn new(ctx: &PyContext) -> Self {
        Self {
            inner: MemoryExchange::new(&ctx.inner),
        }
    }

    /// Bind a withdrawal over a buffer or texture deed parcel.
    fn bind_withdraw(&self, scheme: &PyScheme, parcel: &PyParcel) -> PyResult<PyWithdrawTransaction> {
        scheme.ensure_no_active_recorder()?;
        let tx = self
            .inner
            .bind_withdraw(&mut scheme.inner.borrow_mut(), parcel.inner.as_parcel())
            .into_py_result()?;
        Ok(PyWithdrawTransaction { inner: tx })
    }

    /// Bind a withdrawal over a texture deed.
    fn bind_withdraw_texture(&self, scheme: &PyScheme, texture: &PyTexture) -> PyResult<PyWithdrawTransaction> {
        scheme.ensure_no_active_recorder()?;
        let tx = self
            .inner
            .bind_withdraw(&mut scheme.inner.borrow_mut(), &*texture.inner)
            .into_py_result()?;
        Ok(PyWithdrawTransaction { inner: tx })
    }

    /// Bind a deposit into a destination buffer parcel.
    fn bind_deposit_buffer(
        &self,
        scheme: &PyScheme,
        destination: &PyParcel,
        capacity: u64,
    ) -> PyResult<PyDepositTransaction> {
        scheme.ensure_no_active_recorder()?;
        let tx = self
            .inner
            .bind_deposit_buffer(&mut scheme.inner.borrow_mut(), destination.inner.as_parcel(), capacity)
            .into_py_result()?;
        Ok(PyDepositTransaction { inner: tx })
    }

    /// Bind a deposit into a texture region.
    #[pyo3(signature = (scheme, destination, x, y, width, height, capacity, src_row_pitch=0))]
    #[allow(clippy::too_many_arguments)]
    fn bind_deposit_texture(
        &self,
        scheme: &PyScheme,
        destination: &PyTexture,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        capacity: u64,
        src_row_pitch: u32,
    ) -> PyResult<PyDepositTransaction> {
        scheme.ensure_no_active_recorder()?;
        let tx = self
            .inner
            .bind_deposit_texture(
                &mut scheme.inner.borrow_mut(),
                &*destination.inner,
                x,
                y,
                width,
                height,
                capacity,
                src_row_pitch,
            )
            .into_py_result()?;
        Ok(PyDepositTransaction { inner: tx })
    }

    fn __repr__(&self) -> String {
        "MemoryExchange()".to_string()
    }
}

/// Stable withdraw relationship recorded in one scheme.
#[pyclass(name = "WithdrawTransaction", module = "goldy", unsendable)]
pub struct PyWithdrawTransaction {
    pub(crate) inner: WithdrawTransaction,
}

#[pymethods]
impl PyWithdrawTransaction {
    fn byte_size(&self) -> u64 {
        self.inner.byte_size()
    }

    fn claim(&self, submission: &mut PySchemeSubmission) -> PyResult<PyWithdrawClaim> {
        let claim = self.inner.claim(&mut submission.inner).into_py_result()?;
        Ok(PyWithdrawClaim { inner: Some(claim) })
    }

    fn __repr__(&self) -> String {
        format!("WithdrawTransaction(byte_size={})", self.inner.byte_size())
    }
}

/// Linear claim for one submission's memory withdrawal.
#[pyclass(name = "WithdrawClaim", module = "goldy", unsendable)]
pub struct PyWithdrawClaim {
    pub(crate) inner: Option<WithdrawClaim>,
}

#[pymethods]
impl PyWithdrawClaim {
    fn consume<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let claim = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("withdraw claim already settled"))?;
        let bytes = claim.consume().into_py_result()?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn discard(&mut self) -> PyResult<()> {
        let claim = self
            .inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("withdraw claim already settled"))?;
        claim.discard().into_py_result()
    }

    fn __repr__(&self) -> String {
        format!("WithdrawClaim(settled={})", self.inner.is_none())
    }
}

impl Drop for PyWithdrawClaim {
    fn drop(&mut self) {
        if let Some(claim) = self.inner.take() {
            let _ = claim.discard();
        }
    }
}

/// Stable deposit relationship recorded in one scheme.
#[pyclass(name = "DepositTransaction", module = "goldy", unsendable)]
pub struct PyDepositTransaction {
    pub(crate) inner: DepositTransaction,
}

#[pymethods]
impl PyDepositTransaction {
    fn capacity(&self) -> u64 {
        self.inner.capacity()
    }

    fn id(&self) -> u32 {
        self.inner.id()
    }

    #[pyo3(signature = (scheme, data, offset=0))]
    fn write(&self, scheme: &PyScheme, data: &[u8], offset: u64) -> PyResult<()> {
        scheme.ensure_no_active_recorder()?;
        self.inner
            .write(&mut scheme.inner.borrow_mut(), offset, data)
            .into_py_result()
    }

    fn __repr__(&self) -> String {
        format!(
            "DepositTransaction(id={}, capacity={})",
            self.inner.id(),
            self.inner.capacity()
        )
    }
}
