//! Python wrappers for [`goldy::MemoryExchange`] withdraw / deposit.

use crate::error::IntoPyResult;
use crate::parcel::PyParcel;
use crate::scheme::{PyContext, PyScheme, PySchemeSubmission};
use crate::texture::PyTexture;
use goldy::{DepositTarget, DepositTransaction, MemoryExchange, WithdrawClaim, WithdrawTransaction};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Destination of a memory-exchange deposit (buffer range or texture region).
#[pyclass(name = "DepositTarget", module = "goldy", unsendable)]
pub struct PyDepositTarget {
    parcel: Option<Py<PyParcel>>,
    texture: Option<Py<PyTexture>>,
    dst_offset: u64,
    capacity: u64,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    src_row_pitch: u32,
    is_texture: bool,
}

#[pymethods]
impl PyDepositTarget {
    /// Whole-parcel or ranged buffer deposit.
    #[staticmethod]
    #[pyo3(signature = (destination, capacity, dst_offset=0))]
    fn buffer(destination: Py<PyParcel>, capacity: u64, dst_offset: u64) -> Self {
        Self {
            parcel: Some(destination),
            texture: None,
            dst_offset,
            capacity,
            x: 0,
            y: 0,
            width: 0,
            height: 0,
            src_row_pitch: 0,
            is_texture: false,
        }
    }

    /// Texture-region deposit.
    #[staticmethod]
    #[pyo3(signature = (destination, x, y, width, height, capacity, src_row_pitch=0))]
    #[allow(clippy::too_many_arguments)]
    fn texture(
        destination: Py<PyTexture>,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        capacity: u64,
        src_row_pitch: u32,
    ) -> Self {
        Self {
            parcel: None,
            texture: Some(destination),
            dst_offset: 0,
            capacity,
            x,
            y,
            width,
            height,
            src_row_pitch,
            is_texture: true,
        }
    }

    fn __repr__(&self) -> String {
        if self.is_texture {
            format!(
                "DepositTarget.texture(x={}, y={}, width={}, height={}, capacity={})",
                self.x, self.y, self.width, self.height, self.capacity
            )
        } else {
            format!(
                "DepositTarget.buffer(capacity={}, dst_offset={})",
                self.capacity, self.dst_offset
            )
        }
    }
}

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

    /// Bind a deposit into a destination buffer parcel or texture region.
    fn bind_deposit(
        &self,
        py: Python<'_>,
        scheme: &PyScheme,
        target: &PyDepositTarget,
    ) -> PyResult<PyDepositTransaction> {
        scheme.ensure_no_active_recorder()?;
        let tx = if target.is_texture {
            let texture = target
                .texture
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("DepositTarget.texture is missing a texture"))?;
            let texture = texture.bind(py);
            let texture = texture.borrow();
            self.inner
                .bind_deposit(
                    &mut scheme.inner.borrow_mut(),
                    DepositTarget::texture(
                        &*texture.inner,
                        target.x,
                        target.y,
                        target.width,
                        target.height,
                        target.capacity,
                        target.src_row_pitch,
                    ),
                )
                .into_py_result()?
        } else {
            let parcel = target
                .parcel
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("DepositTarget.buffer is missing a parcel"))?;
            let parcel = parcel.bind(py);
            let parcel = parcel.borrow();
            self.inner
                .bind_deposit(
                    &mut scheme.inner.borrow_mut(),
                    DepositTarget::Buffer {
                        destination: parcel.inner.as_parcel(),
                        dst_offset: target.dst_offset,
                        capacity: target.capacity,
                    },
                )
                .into_py_result()?
        };
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

    #[pyo3(signature = (data, offset=0))]
    fn write(&self, data: &[u8], offset: u64) -> PyResult<()> {
        self.inner.write(offset, data).into_py_result()
    }

    fn __repr__(&self) -> String {
        format!(
            "DepositTransaction(id={}, capacity={})",
            self.inner.id(),
            self.inner.capacity()
        )
    }
}
