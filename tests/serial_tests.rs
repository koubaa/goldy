#![allow(deprecated)]

#[path = "common/submission.rs"]
mod submission;

#[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
mod imp {
    //! Integration tests that must run serially against a single shared [`Device`].
    //!
    //! Each test here documents, directly above its function, which **device-global**
    //! invariant it depends on. If a test does not need isolation from other trials
    //! on the same device, it belongs in [`compute_integration`] instead.
    //!
    //! [`compute_integration`]: compute_integration

    use crate::submission::submission_context;
    use goldy::{
        types::BufferFlags, Buffer, BufferKind, ComputePipeline, Device, DeviceDescriptor, Instance, NodeAccess,
        RequestAdapterOptions, RetainedPool, Scheme, ShaderModule,
    };
    use std::sync::Arc;

    fn request_default_device(instance: &Instance) -> Device {
        instance
            .request_adapter(&RequestAdapterOptions::default())
            .expect("Failed to request adapter")
            .request_device(&DeviceDescriptor::default())
            .expect("Failed to create device")
    }

    fn make_device() -> Device {
        request_default_device(&Instance::new().expect("Failed to create instance"))
    }

    fn test_alloc_buffer(
        device: &Device,
        size: u64,
        kind: BufferKind,
        stride: Option<u32>,
        flags: BufferFlags,
    ) -> Buffer {
        RetainedPool::new(Arc::new(device.clone()))
            .acquire_buffer(size, kind, stride, flags, None)
            .expect("acquire_buffer")
    }

    /// Isolation reason: [`Device::device_deferred_deletion_pending_count`] is a
    /// device-global counter shared by every context on this device. If any other
    /// trial is running concurrently against the same device and has an in-flight
    /// bindless buffer destroy queued, this assertion can observe a nonzero count
    /// even though this trial's own destroy already drained correctly.
    fn headless_deferred_buffer_destroy_drains_after_timeline_wait(device: &Device) {
        const MINIMAL_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(1, 1, 1)]
    void cs_main(Scattered<uint> _unused, ThreadId id) {
    }
    "#;

        let ctx = submission_context(device);
        let shader = ShaderModule::from_slang(device, MINIMAL_SHADER).expect("compile");
        let pipeline = ComputePipeline::new(device, &shader).expect("pipeline");

        // Buffer must be referenced at submit time so bindless retirement requirements
        // include this context's timeline value; an unreferenced buffer has empty
        // requirements and is drainable immediately by any sibling trial's wait_until.
        let buf = test_alloc_buffer(device, 256, BufferKind::Scattered, None, BufferFlags::empty());

        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n0", &pipeline)
            .with_parcel(&buf, NodeAccess::Write)
            .dispatch(1, 1, 1);
        let tv = scheme.submit().expect("submit").timeline_value();

        assert_eq!(
            ctx.deferred_deletion_pending_count(),
            0,
            "bindless buffer destroys are queued on the device-level deletion queue, not per-context"
        );

        // Dropping the retained buffer must not require dropping the scheme first:
        // destroy evicts retained CBs that pin its slots, and marks the scheme stamp dead.
        drop(buf);

        ctx.wait_until(tv).expect("wait_until");

        assert_eq!(
            ctx.deferred_deletion_pending_count(),
            0,
            "per-context deletion queue should stay empty for bindless buffer destroys"
        );
        assert_eq!(
            device.device_deferred_deletion_pending_count(),
            0,
            "wait_until should drain device deferred destruction for completed timeline values"
        );

        assert!(
            matches!(scheme.submit(), Err(goldy::GoldyError::StaleResource)),
            "submit after dropping a bound retained buffer must fail"
        );
    }

    pub fn run() {
        let device = make_device();

        let mut trials = Vec::new();
        macro_rules! trial {
            ($f:ident) => {{
                let device = device.clone();
                trials.push(libtest_mimic::Trial::test(stringify!($f), move || {
                    $f(&device);
                    Ok(())
                }));
            }};
        }

        trial!(headless_deferred_buffer_destroy_drains_after_timeline_wait);

        let mut args = libtest_mimic::Arguments::from_args();
        args.test_threads = Some(1);
        let conclusion = libtest_mimic::run(&args, trials);

        drop(device);

        conclusion.exit_if_failed();
    }
}

fn main() {
    #[cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
    imp::run();
}
