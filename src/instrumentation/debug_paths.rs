use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static DEBUG_DIR: OnceLock<PathBuf> = OnceLock::new();
static LOG_PATH: OnceLock<PathBuf> = OnceLock::new();

fn debug_dir() -> &'static Path {
    DEBUG_DIR
        .get_or_init(|| {
            let dir = match std::env::var("GOLDY_DEBUG_DIR") {
                Ok(v) => PathBuf::from(v),
                Err(_) => std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join(".goldy_debug"),
            };
            let _ = std::fs::create_dir_all(&dir);
            dir
        })
        .as_path()
}

/// Returns the debug log path.
///
/// Resolution order:
/// 1. `GOLDY_DEBUG_LOG` env var (exact path)
/// 2. `{GOLDY_DEBUG_DIR}/debug.log`
/// 3. `{CWD}/.goldy_debug/debug.log`
pub fn debug_log_path() -> &'static Path {
    LOG_PATH
        .get_or_init(|| match std::env::var("GOLDY_DEBUG_LOG") {
            Ok(v) => PathBuf::from(v),
            Err(_) => debug_dir().join("debug.log"),
        })
        .as_path()
}

/// Returns `{debug_dir}/{name}` (e.g. `shader_dump_0_cs_main.metal`).
pub fn shader_dump_path(name: &str) -> PathBuf {
    debug_dir().join(name)
}
