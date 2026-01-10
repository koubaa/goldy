//! Shared Digital Clock rendering logic.
//!
//! This module provides the core rendering components for the digital clock demo:
//! - WGSL shader source
//! - Vertex data structures
//! - Digit pattern generation
//! - Time formatting
//!
//! Both native (Vulkan) and web (WebGPU) examples import this shared code,
//! ensuring identical rendering across platforms.

use crate::types::{Color, VertexBufferLayout, VertexAttribute, VertexFormat};
use bytemuck::{Pod, Zeroable};

/// WGSL shader for the digital clock.
/// 
/// Uses standard vertex coloring with position and color attributes.
pub const SHADER_SOURCE: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4<f32>(in.position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// Vertex with 2D position and RGBA color.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ClockVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

impl ClockVertex {
    pub const fn new(x: f32, y: f32, color: Color) -> Self {
        Self {
            position: [x, y],
            color: [color.r, color.g, color.b, color.a],
        }
    }

    /// Get the vertex buffer layout for this vertex type.
    pub fn layout() -> VertexBufferLayout {
        VertexBufferLayout {
            stride: std::mem::size_of::<Self>() as u32,
            attributes: vec![
                VertexAttribute {
                    location: 0,
                    format: VertexFormat::Float32x2,
                    offset: 0,
                },
                VertexAttribute {
                    location: 1,
                    format: VertexFormat::Float32x4,
                    offset: 8,
                },
            ],
        }
    }
}

/// Seven-segment display patterns.
/// Order: top, top-left, top-right, middle, bottom-left, bottom-right, bottom
pub const SEGMENT_PATTERNS: [[bool; 7]; 11] = [
    [true, true, true, false, true, true, true],     // 0
    [false, false, true, false, false, true, false], // 1
    [true, false, true, true, true, false, true],    // 2
    [true, false, true, true, false, true, true],    // 3
    [false, true, true, true, false, true, false],   // 4
    [true, true, false, true, false, true, true],    // 5
    [true, true, false, true, true, true, true],     // 6
    [true, false, true, false, false, true, false],  // 7
    [true, true, true, true, true, true, true],      // 8
    [true, true, true, true, false, true, true],     // 9
    [false, false, false, false, false, false, false], // 10 = blank (for colon position)
];

/// Color palette for the clock.
pub const COLORS: [Color; 8] = [
    Color { r: 0.2, g: 1.0, b: 0.3, a: 1.0 },    // Green (default)
    Color { r: 1.0, g: 0.3, b: 0.2, a: 1.0 },    // Red
    Color { r: 1.0, g: 0.6, b: 0.0, a: 1.0 },    // Orange
    Color { r: 1.0, g: 1.0, b: 0.2, a: 1.0 },    // Yellow
    Color { r: 0.2, g: 1.0, b: 1.0, a: 1.0 },    // Cyan
    Color { r: 0.4, g: 0.6, b: 1.0, a: 1.0 },    // Blue
    Color { r: 0.8, g: 0.3, b: 1.0, a: 1.0 },    // Purple
    Color { r: 1.0, g: 0.4, b: 0.8, a: 1.0 },    // Pink
];

/// Generate vertices for a filled quad.
pub fn quad_vertices(x: f32, y: f32, w: f32, h: f32, color: Color) -> [ClockVertex; 6] {
    [
        ClockVertex::new(x, y, color),
        ClockVertex::new(x + w, y, color),
        ClockVertex::new(x + w, y + h, color),
        ClockVertex::new(x, y, color),
        ClockVertex::new(x + w, y + h, color),
        ClockVertex::new(x, y + h, color),
    ]
}

/// Convert pixel coordinates to normalized device coordinates.
pub fn pixel_to_ndc(px: f32, py: f32, width: f32, height: f32) -> (f32, f32) {
    let x = (px / width) * 2.0 - 1.0;
    let y = 1.0 - (py / height) * 2.0;
    (x, y)
}

/// Generate vertices for a single digit (or colon).
pub fn digit_vertices(
    digit: u8,
    cx: f32,
    cy: f32,
    scale: f32,
    color: Color,
    width: f32,
    height: f32,
) -> Vec<ClockVertex> {
    let mut vertices = Vec::new();

    let seg_w = 60.0 * scale;
    let seg_h = 12.0 * scale;
    let dig_h = 120.0 * scale;
    let gap = 4.0 * scale;

    // Colon (digit 10)
    if digit == 10 {
        let dot_size = seg_h * 1.5;
        let dot_spacing = dig_h * 0.5;

        let (x, y) = pixel_to_ndc(cx - dot_size / 2.0, cy - dot_spacing - dot_size / 2.0, width, height);
        let (w, h) = (dot_size / width * 2.0, dot_size / height * 2.0);
        vertices.extend_from_slice(&quad_vertices(x, y, w, -h, color));

        let (x, y) = pixel_to_ndc(cx - dot_size / 2.0, cy + dot_spacing - dot_size / 2.0, width, height);
        vertices.extend_from_slice(&quad_vertices(x, y, w, -h, color));

        return vertices;
    }

    let pattern = SEGMENT_PATTERNS[digit as usize];

    let mut add_segment = |px: f32, py: f32, pw: f32, ph: f32| {
        let (x, y) = pixel_to_ndc(px, py, width, height);
        let (w, h) = (pw / width * 2.0, ph / height * 2.0);
        vertices.extend_from_slice(&quad_vertices(x, y, w, -h, color));
    };

    // Segment indices: 0=top, 1=top-left, 2=top-right, 3=middle, 4=bottom-left, 5=bottom-right, 6=bottom
    if pattern[0] { add_segment(cx - seg_w / 2.0, cy - dig_h, seg_w, seg_h); }
    if pattern[1] { add_segment(cx - seg_w / 2.0 - seg_h, cy - dig_h + seg_h + gap, seg_h, dig_h - seg_h - gap * 2.0); }
    if pattern[2] { add_segment(cx + seg_w / 2.0, cy - dig_h + seg_h + gap, seg_h, dig_h - seg_h - gap * 2.0); }
    if pattern[3] { add_segment(cx - seg_w / 2.0, cy - seg_h / 2.0, seg_w, seg_h); }
    if pattern[4] { add_segment(cx - seg_w / 2.0 - seg_h, cy + gap, seg_h, dig_h - seg_h - gap * 2.0); }
    if pattern[5] { add_segment(cx + seg_w / 2.0, cy + gap, seg_h, dig_h - seg_h - gap * 2.0); }
    if pattern[6] { add_segment(cx - seg_w / 2.0, cy + dig_h - seg_h, seg_w, seg_h); }

    vertices
}

/// Time data for rendering.
#[derive(Debug, Clone, Copy, Default)]
pub struct TimeData {
    pub hours: u8,
    pub minutes: u8,
    pub seconds: u8,
}

impl TimeData {
    /// Create from elapsed seconds (timer mode).
    pub fn from_elapsed_secs(elapsed: u64) -> Self {
        Self {
            hours: ((elapsed / 3600) % 100) as u8,
            minutes: ((elapsed % 3600) / 60) as u8,
            seconds: (elapsed % 60) as u8,
        }
    }

    /// Convert to digit array: [h1, h2, colon, m1, m2, colon, s1, s2]
    pub fn to_digits(&self) -> [u8; 8] {
        [
            self.hours / 10, self.hours % 10,
            10, // colon
            self.minutes / 10, self.minutes % 10,
            10, // colon
            self.seconds / 10, self.seconds % 10,
        ]
    }
}

/// Generate all vertices for the clock display.
pub fn generate_clock_vertices(
    time: TimeData,
    color: Color,
    width: u32,
    height: u32,
) -> Vec<ClockVertex> {
    let digits = time.to_digits();

    let scale = height as f32 / 720.0;
    let digit_width = 80.0 * scale;
    let colon_width = 40.0 * scale;
    let spacing = 20.0 * scale;

    let total_width = digit_width * 6.0 + colon_width * 2.0 + spacing * 7.0;

    let cy = height as f32 / 2.0;
    let mut cx = (width as f32 - total_width) / 2.0 + digit_width / 2.0;

    let mut all_vertices = Vec::new();

    for &digit in digits.iter() {
        let w = if digit == 10 { colon_width } else { digit_width };
        let verts = digit_vertices(digit, cx, cy, scale, color, width as f32, height as f32);
        all_vertices.extend_from_slice(&verts);
        cx += w + spacing;
    }

    all_vertices
}

/// Clock state for pause/resume functionality.
#[derive(Debug, Clone)]
pub struct ClockState {
    pub color_index: usize,
    pub paused: bool,
    pub accumulated_secs: u64,
}

impl Default for ClockState {
    fn default() -> Self {
        Self {
            color_index: 0,
            paused: false,
            accumulated_secs: 0,
        }
    }
}

impl ClockState {
    /// Get the current display color.
    pub fn color(&self) -> Color {
        let mut color = COLORS[self.color_index];
        // Dim when paused
        if self.paused {
            color.r *= 0.5;
            color.g *= 0.5;
            color.b *= 0.5;
        }
        color
    }

    /// Get background color.
    pub fn background_color(&self) -> Color {
        let bg = if self.paused { 0.06 } else { 0.02 };
        Color { r: bg, g: bg, b: bg, a: 1.0 }
    }

    /// Cycle to next color.
    pub fn next_color(&mut self) {
        self.color_index = (self.color_index + 1) % COLORS.len();
    }

    /// Toggle pause state.
    /// Returns the elapsed seconds that should be preserved.
    pub fn toggle_pause(&mut self, current_elapsed: u64) {
        if self.paused {
            // Resuming - accumulated_secs stays the same
            self.paused = false;
        } else {
            // Pausing - save current elapsed
            self.accumulated_secs = current_elapsed;
            self.paused = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_data_from_elapsed() {
        let time = TimeData::from_elapsed_secs(3661); // 1h 1m 1s
        assert_eq!(time.hours, 1);
        assert_eq!(time.minutes, 1);
        assert_eq!(time.seconds, 1);
    }

    #[test]
    fn test_time_data_to_digits() {
        let time = TimeData { hours: 12, minutes: 34, seconds: 56 };
        let digits = time.to_digits();
        assert_eq!(digits, [1, 2, 10, 3, 4, 10, 5, 6]);
    }

    #[test]
    fn test_vertex_generation() {
        let time = TimeData::from_elapsed_secs(0);
        let vertices = generate_clock_vertices(time, Color::GREEN, 800, 600);
        assert!(!vertices.is_empty());
    }
}

