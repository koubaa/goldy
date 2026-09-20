//! Metal Performance Shaders realization of [`crate::backend::GpuCommand::MatMul`].

#![allow(deprecated)]

use super::types::MetalState;
use crate::ops::matmul::MatMulOperand;
use crate::ops::MatMulDesc;
use anyhow::{Context, Result};
use cocoa::base::{id, nil};
use objc::{class, msg_send, sel, sel_impl};

#[link(name = "MetalPerformanceShaders", kind = "framework")]
extern "C" {}

const MPS_DATA_TYPE_FLOAT32: u32 = 0x10000 | 32;

pub(super) fn encode(
    state: &MetalState,
    command_buffer: &::metal::CommandBufferRef,
    label: Option<&'static str>,
    desc: MatMulDesc,
    a: MatMulOperand,
    b: MatMulOperand,
    c: MatMulOperand,
) -> Result<()> {
    if desc.dtype != crate::ops::MatMulDType::F32 {
        anyhow::bail!("Metal MatMul: only F32 is supported");
    }
    let name = label.unwrap_or("matmul");
    unsafe {
        let left = mps_matrix(state, &a, matrix_rows_cols_a(&desc), name, "A")?;
        let right = mps_matrix(state, &b, matrix_rows_cols_b(&desc), name, "B")?;
        let result = mps_matrix(state, &c, (desc.m as u64, desc.n as u64), name, "C")?;
        let kernel: id = msg_send![class!(MPSMatrixMultiplication), alloc];
        let kernel: id = msg_send![
            kernel,
            initWithDevice: command_buffer.device()
            transposeLeft: desc.transpose_a
            transposeRight: desc.transpose_b
            resultRows: desc.m as u64
            resultColumns: desc.n as u64
            interiorColumns: desc.k as u64
            alpha: desc.alpha as f64
            beta: desc.beta as f64
        ];
        if kernel == nil {
            anyhow::bail!("Metal MatMul `{name}`: MPSMatrixMultiplication init failed");
        }
        let _: () = msg_send![
            kernel,
            encodeToCommandBuffer: command_buffer
            leftMatrix: left
            rightMatrix: right
            resultMatrix: result
        ];
    }
    Ok(())
}

fn matrix_rows_cols_a(desc: &MatMulDesc) -> (u64, u64) {
    if desc.transpose_a {
        (desc.k as u64, desc.m as u64)
    } else {
        (desc.m as u64, desc.k as u64)
    }
}

fn matrix_rows_cols_b(desc: &MatMulDesc) -> (u64, u64) {
    if desc.transpose_b {
        (desc.n as u64, desc.k as u64)
    } else {
        (desc.k as u64, desc.n as u64)
    }
}

unsafe fn mps_matrix(
    state: &MetalState,
    operand: &MatMulOperand,
    rows_cols: (u64, u64),
    label: &str,
    name: &str,
) -> Result<id> {
    let buf = state
        .buffers
        .get(&operand.buffer)
        .with_context(|| format!("Metal MatMul `{label}`: invalid {name} buffer"))?;
    let view_off = buf.view_byte_offset.unwrap_or(0);
    let elem_off = operand
        .offset_elements
        .checked_mul(4)
        .context("Metal MatMul: element offset overflow")?;
    let offset = view_off + elem_off;
    let row_bytes = u64::from(operand.leading_dim) * 4;
    let (rows, cols) = rows_cols;
    let desc: id = msg_send![
        class!(MPSMatrixDescriptor),
        matrixDescriptorWithRows: rows
        columns: cols
        rowBytes: row_bytes
        dataType: MPS_DATA_TYPE_FLOAT32
    ];
    if desc == nil {
        anyhow::bail!("Metal MatMul `{label}`: MPSMatrixDescriptor failed for {name}");
    }
    let matrix: id = msg_send![class!(MPSMatrix), alloc];
    let matrix: id = msg_send![
        matrix,
        initWithBuffer: &*buf.buffer
        offset: offset
        descriptor: desc
    ];
    if matrix == nil {
        anyhow::bail!("Metal MatMul `{label}`: MPSMatrix init failed for {name}");
    }
    Ok(matrix)
}
