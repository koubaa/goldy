//! Finalize a typed kernel record into a Scheme dispatch node.

use crate::scheme::SchemeNodeBuilder;

/// Intermediate after [`super::PreparedKernel`]-generated `record` binds args.
pub struct DispatchBuilder<'a> {
    pub(crate) builder: SchemeNodeBuilder<'a>,
    pub(crate) workgroup_size: [u32; 3],
}

/// Returned by [`DispatchBuilder::groups`] / [`DispatchBuilder::over_1d`].
#[derive(Debug, Clone, Copy)]
pub struct RecordedDispatch {
    node: crate::NodeId,
}

impl RecordedDispatch {
    /// The dispatch node, for [`crate::Scheme::set_node_param`] and friends.
    pub fn node(&self) -> crate::NodeId {
        self.node
    }
}

impl<'a> DispatchBuilder<'a> {
    pub fn new(builder: SchemeNodeBuilder<'a>, workgroup_size: [u32; 3]) -> Self {
        Self {
            builder,
            workgroup_size,
        }
    }

    /// Exact workgroup/grid counts (CUDA `gridDim` analogue). Workgroup size is fixed
    /// in the pipeline from `KernelDef::workgroup_size`.
    pub fn groups(self, counts: [u32; 3]) -> RecordedDispatch {
        RecordedDispatch {
            node: self.builder.dispatch(counts[0], counts[1], counts[2]),
        }
    }

    /// Cover `n` threads in 1D using the kernel's fixed workgroup size.
    pub fn over_1d(self, n: u32) -> RecordedDispatch {
        let wx = self.workgroup_size[0].max(1);
        let groups = n.div_ceil(wx);
        self.groups([groups, 1, 1])
    }

    /// Cover a 2D thread extent.
    pub fn over_2d(self, width: u32, height: u32) -> RecordedDispatch {
        let wx = self.workgroup_size[0].max(1);
        let wy = self.workgroup_size[1].max(1);
        self.groups([width.div_ceil(wx), height.div_ceil(wy), 1])
    }

    /// Cover a 3D thread extent.
    pub fn over_3d(self, width: u32, height: u32, depth: u32) -> RecordedDispatch {
        let wx = self.workgroup_size[0].max(1);
        let wy = self.workgroup_size[1].max(1);
        let wz = self.workgroup_size[2].max(1);
        self.groups([width.div_ceil(wx), height.div_ceil(wy), depth.div_ceil(wz)])
    }

    /// Cover a tensor view's logical element count in 1D.
    ///
    /// Dispatch geometry follows the view's `numel`, not the parent buffer size.
    #[cfg(feature = "tensor")]
    pub fn over_tensor(self, view: &crate::tensor::TensorView<'_>) -> RecordedDispatch {
        self.over_1d(view.numel_u32())
    }
}
