//! Owned names for scheme nodes.

use std::borrow::Borrow;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

/// Name of a scheme node, owned by the scheme IR.
///
/// Recording APIs take [`Into<SchemeLabel>`], so string literals (`"embed"`),
/// [`String`] (`format!("attn_{layer}")`), and [`Arc<str>`] all work. The scheme
/// keeps the name for the node's lifetime; callers do not intern or leak.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SchemeLabel(Arc<str>);

impl SchemeLabel {
    /// Borrow the label as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Deref for SchemeLabel {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for SchemeLabel {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for SchemeLabel {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SchemeLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for SchemeLabel {
    fn from(s: &str) -> Self {
        Self(Arc::from(s))
    }
}

impl From<String> for SchemeLabel {
    fn from(s: String) -> Self {
        Self(Arc::from(s))
    }
}

impl From<&String> for SchemeLabel {
    fn from(s: &String) -> Self {
        Self(Arc::from(s.as_str()))
    }
}

impl From<Arc<str>> for SchemeLabel {
    fn from(s: Arc<str>) -> Self {
        Self(s)
    }
}

impl PartialEq<str> for SchemeLabel {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for SchemeLabel {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}
