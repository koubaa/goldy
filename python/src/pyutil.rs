//! Small Python interop helpers shared across Goldy bindings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Parse a Python `range` or `slice` into `(start, count)` with step 1.
pub(crate) fn parse_index_range(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<(u32, u32)> {
    let start: i64 = match obj.getattr("start") {
        Ok(v) => v.extract().unwrap_or(0),
        Err(_) => 0,
    };
    let stop: i64 = obj
        .getattr("stop")
        .map_err(|_| PyValueError::new_err(format!("{name} must be a range or slice")))?
        .extract()?;
    let step: i64 = match obj.getattr("step") {
        Ok(v) => v.extract().unwrap_or(1),
        Err(_) => 1,
    };
    if step != 1 {
        return Err(PyValueError::new_err(format!("{name} range step must be 1")));
    }
    if start < 0 || stop < start {
        return Err(PyValueError::new_err(format!(
            "invalid {name} range: start={start}, stop={stop}"
        )));
    }
    Ok((start as u32, (stop - start) as u32))
}
