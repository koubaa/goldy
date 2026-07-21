//! Shared digital-clock rendering helpers for the `digital_clock` example.

use goldy::buffer::StructuredBufferElement;
use goldy::types::{Color, VertexBufferLayout, VertexFormat};
use bytemuck::{Pod, Zeroable};

/// Slang shader for the digital clock.
pub const SHADER_SOURCE: &str = r#"
struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
"#;

/// Vertex with 2D position and RGBA color.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct ClockVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}
impl StructuredBufferElement for ClockVertex {}

impl ClockVertex {
    pub const fn new(x: f32, y: f32, color: Color) -> Self {
        Self {
            position: [x, y],
            color: [color.r, color.g, color.b, color.a],
        }
    }

    pub fn layout() -> VertexBufferLayout {
        VertexBufferLayout::from_formats::<Self>(&[
            VertexFormat::Float32x2,
            VertexFormat::Float32x4,
        ])
    }
}

/// Seven-segment display patterns.
/// Order: top, top-left, top-right, middle, bottom-left, bottom-right, bottom
pub const SEGMENT_PATTERNS: [[bool; 7]; 11] = [
    [true, true, true, false, true, true, true],       // 0
    [false, false, true, false, false, true, false],   // 1
    [true, false, true, true, true, false, true],      // 2
    [true, false, true, true, false, true, true],      // 3
    [false, true, true, true, false, true, false],     // 4
    [true, true, false, true, false, true, true],      // 5
    [true, true, false, true, true, true, true],       // 6
    [true, false, true, false, false, true, false],    // 7
    [true, true, true, true, true, true, true],        // 8
    [true, true, true, true, false, true, true],       // 9
    [false, false, false, false, false, false, false], // 10 = blank (for colon position)
];

pub const COLORS: [Color; 8] = [
    Color {
        r: 0.2,
        g: 1.0,
        b: 0.3,
        a: 1.0,
    },
    Color {
        r: 1.0,
        g: 0.3,
        b: 0.2,
        a: 1.0,
    },
    Color {
        r: 1.0,
        g: 0.6,
        b: 0.0,
        a: 1.0,
    },
    Color {
        r: 1.0,
        g: 1.0,
        b: 0.2,
        a: 1.0,
    },
    Color {
        r: 0.2,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    },
    Color {
        r: 0.4,
        g: 0.6,
        b: 1.0,
        a: 1.0,
    },
    Color {
        r: 0.8,
        g: 0.3,
        b: 1.0,
        a: 1.0,
    },
    Color {
        r: 1.0,
        g: 0.4,
        b: 0.8,
        a: 1.0,
    },
];

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

pub fn pixel_to_ndc(px: f32, py: f32, width: f32, height: f32) -> (f32, f32) {
    let x = (px / width) * 2.0 - 1.0;
    let y = 1.0 - (py / height) * 2.0;
    (x, y)
}

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

    if pattern[0] {
        add_segment(cx - seg_w / 2.0, cy - dig_h, seg_w, seg_h);
    }
    if pattern[1] {
        add_segment(
            cx - seg_w / 2.0 - seg_h,
            cy - dig_h + seg_h + gap,
            seg_h,
            dig_h - seg_h - gap * 2.0,
        );
    }
    if pattern[2] {
        add_segment(
            cx + seg_w / 2.0,
            cy - dig_h + seg_h + gap,
            seg_h,
            dig_h - seg_h - gap * 2.0,
        );
    }
    if pattern[3] {
        add_segment(cx - seg_w / 2.0, cy - seg_h / 2.0, seg_w, seg_h);
    }
    if pattern[4] {
        add_segment(cx - seg_w / 2.0 - seg_h, cy + gap, seg_h, dig_h - seg_h - gap * 2.0);
    }
    if pattern[5] {
        add_segment(cx + seg_w / 2.0, cy + gap, seg_h, dig_h - seg_h - gap * 2.0);
    }
    if pattern[6] {
        add_segment(cx - seg_w / 2.0, cy + dig_h - seg_h, seg_w, seg_h);
    }

    vertices
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TimeData {
    pub hours: u8,
    pub minutes: u8,
    pub seconds: u8,
}

impl TimeData {
    pub fn from_elapsed_secs(elapsed: u64) -> Self {
        Self {
            hours: ((elapsed / 3600) % 100) as u8,
            minutes: ((elapsed % 3600) / 60) as u8,
            seconds: (elapsed % 60) as u8,
        }
    }

    pub fn to_digits(&self) -> [u8; 8] {
        [
            self.hours / 10,
            self.hours % 10,
            10,
            self.minutes / 10,
            self.minutes % 10,
            10,
            self.seconds / 10,
            self.seconds % 10,
        ]
    }
}

pub fn generate_clock_vertices(time: TimeData, color: Color, width: u32, height: u32) -> Vec<ClockVertex> {
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

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct ClockState {
    pub color_index: usize,
    pub paused: bool,
    pub accumulated_secs: u64,
}

#[allow(dead_code)]
impl ClockState {
    pub fn color(&self) -> Color {
        let mut color = COLORS[self.color_index];
        if self.paused {
            color.r *= 0.5;
            color.g *= 0.5;
            color.b *= 0.5;
        }
        color
    }

    pub fn background_color(&self) -> Color {
        let bg = if self.paused { 0.06 } else { 0.02 };
        Color {
            r: bg,
            g: bg,
            b: bg,
            a: 1.0,
        }
    }

    pub fn next_color(&mut self) {
        self.color_index = (self.color_index + 1) % COLORS.len();
    }

    pub fn toggle_pause(&mut self, current_elapsed: u64) {
        if self.paused {
            self.paused = false;
        } else {
            self.accumulated_secs = current_elapsed;
            self.paused = true;
        }
    }
}
