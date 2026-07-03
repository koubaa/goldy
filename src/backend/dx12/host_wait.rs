//! Host-side waits run on the submission worker before GPU work.

use anyhow::Result;
use windows::Win32::Graphics::Direct3D12::ID3D12Fence;

/// A wait the submission worker performs on the CPU before `ExecuteCommandLists`.
#[derive(Clone)]
pub(super) enum HostWait {
    Fence { fence: ID3D12Fence, value: u64 },
}

impl HostWait {
    pub(super) fn wait(&self) -> Result<()> {
        let HostWait::Fence { fence, value } = self;
        let result = super::utils::wait_for_fence(fence, *value);
        result
    }
}
