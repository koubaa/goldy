//! JSON file subscriber for structured instrumentation output.
//!
//! This module provides a tracing subscriber layer that writes structured JSON
//! logs to a file, similar to the debug logging format used during development.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::field::{Field, Visit};
use tracing::{Event, Subscriber};
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// A tracing layer that writes JSON-formatted events to a file.
///
/// Each event is written as a single JSON line with the following structure:
/// ```json
/// {
///   "timestamp": 1234567890123,
///   "name": "render.frame.start",
///   "target": "goldy",
///   "level": "DEBUG",
///   "fields": { "frame_id": 42, "other_field": "value" }
/// }
/// ```
pub struct JsonFileLayer {
    file: Mutex<File>,
}

impl JsonFileLayer {
    /// Create a new JSON file layer that writes to the specified path.
    ///
    /// The file is created if it doesn't exist, or truncated if it does.
    pub fn new(path: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        Ok(Self {
            file: Mutex::new(file),
        })
    }

    /// Create a new JSON file layer that appends to the specified path.
    ///
    /// The file is created if it doesn't exist.
    pub fn new_append(path: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            file: Mutex::new(file),
        })
    }

    fn write_event(&self, json: &str) {
        if let Ok(mut file) = self.file.lock() {
            let _ = writeln!(file, "{}", json);
            let _ = file.flush();
        }
    }
}

/// Visitor that collects event fields into a JSON object.
struct JsonVisitor {
    fields: Vec<(String, String)>,
    name: Option<String>,
}

impl JsonVisitor {
    fn new() -> Self {
        Self {
            fields: Vec::new(),
            name: None,
        }
    }

    fn into_json(self) -> (Option<String>, String) {
        let fields_json = self
            .fields
            .iter()
            .map(|(k, v)| format!(r#""{}": {}"#, k, v))
            .collect::<Vec<_>>()
            .join(", ");

        (self.name, format!("{{{}}}", fields_json))
    }
}

impl Visit for JsonVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let value_str = format!("{:?}", value);
        // Try to parse as number, otherwise quote as string
        let json_value = if value_str.parse::<i64>().is_ok()
            || value_str.parse::<f64>().is_ok()
            || value_str == "true"
            || value_str == "false"
        {
            value_str
        } else {
            // Escape quotes and backslashes for JSON
            let escaped = value_str
                .replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            format!(r#""{}""#, escaped)
        };

        if field.name() == "name" {
            // The "name" field is special - it's the observation point name
            self.name = Some(value_str.trim_matches('"').to_string());
        } else {
            self.fields.push((field.name().to_string(), json_value));
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields.push((field.name().to_string(), value.to_string()));
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields.push((field.name().to_string(), value.to_string()));
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.fields.push((field.name().to_string(), value.to_string()));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        let escaped = value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");

        if field.name() == "name" {
            self.name = Some(value.to_string());
        } else {
            self.fields
                .push((field.name().to_string(), format!(r#""{}""#, escaped)));
        }
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.fields.push((field.name().to_string(), value.to_string()));
    }
}

impl<S> Layer<S> for JsonFileLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        // Only capture events targeting "goldy"
        if !event.metadata().target().starts_with(super::TARGET) {
            return;
        }

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);

        let mut visitor = JsonVisitor::new();
        event.record(&mut visitor);

        let (name, fields_json) = visitor.into_json();
        let name = name.unwrap_or_else(|| event.metadata().name().to_string());

        let json = format!(
            r#"{{"timestamp": {}, "name": "{}", "target": "{}", "level": "{}", "fields": {}}}"#,
            timestamp,
            name,
            event.metadata().target(),
            event.metadata().level(),
            fields_json
        );

        self.write_event(&json);
    }

    fn on_enter(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let metadata = span.metadata();
            if !metadata.target().starts_with(super::TARGET) {
                return;
            }

            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);

            let json = format!(
                r#"{{"timestamp": {}, "name": "{}", "target": "{}", "level": "{}", "event": "span_enter"}}"#,
                timestamp,
                metadata.name(),
                metadata.target(),
                metadata.level()
            );

            self.write_event(&json);
        }
    }

    fn on_exit(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let metadata = span.metadata();
            if !metadata.target().starts_with(super::TARGET) {
                return;
            }

            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0);

            let json = format!(
                r#"{{"timestamp": {}, "name": "{}", "target": "{}", "level": "{}", "event": "span_exit"}}"#,
                timestamp,
                metadata.name(),
                metadata.target(),
                metadata.level()
            );

            self.write_event(&json);
        }
    }
}

