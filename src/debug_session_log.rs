//! Temporary NDJSON debug logging for agent debug sessions. Remove after investigation.

// #region agent log
use std::io::Write;
use std::sync::Mutex;

static LOG_MUTEX: Mutex<()> = Mutex::new(());

fn should_log() -> bool {
    match std::thread::current().name() {
        None | Some("") | Some("main") => true,
        Some(name) if name.contains("TID_") => true,
        Some(name) => !name.contains("::tests::") && !name.starts_with("test "),
    }
}

pub(crate) fn write(hypothesis_id: &str, location: &str, message: &str, data_json: &str) {
    if !should_log() {
        return;
    }
    let _guard = LOG_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../debug-440e75.log");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let test = std::thread::current().name().unwrap_or("<unnamed>").to_string();
        let test_escaped = test.replace('\\', "\\\\").replace('"', "\\\"");
        let _ = writeln!(
            f,
            r#"{{"sessionId":"440e75","hypothesisId":"{hypothesis_id}","test":"{test_escaped}","location":"{location}","message":"{message}","data":{data_json},"timestamp":{ts}}}"#
        );
    }
}
// #endregion
