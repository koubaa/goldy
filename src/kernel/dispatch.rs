//! Finalize a typed kernel record into a Scheme dispatch node.

use crate::scheme::SchemeNodeBuilder;

/// Intermediate after [`super::PreparedKernel`]-generated `record` binds args.
pub struct DispatchBuilder<'a> {
    pub(crate) builder: SchemeNodeBuilder<'a>,
    pub(crate) workgroup_size: [u32; 3],
}

/// Marker returned by [`DispatchBuilder::groups`] / [`DispatchBuilder::over_1d`].
pub struct RecordedDispatch;

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
        self.builder.dispatch(counts[0], counts[1], counts[2]);
        RecordedDispatch
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
}
