//! Regenerate FLIP reference PNGs under `tests/screenshots/`.
//!
//! Run explicitly (does not run with `cargo test`):
//! ```text
//! cargo run --bin update-screenshots --features update-screenshots
//! ```

#[path = "../tests/common/render_fixtures.rs"]
mod render_fixtures;

use goldy::{Color, Vertex2D};
use std::path::Path;

fn main() {
    let device = render_fixtures::create_device().expect("GPU device (DiscreteGpu)");

    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/screenshots");

    save_png(
        &out_dir.join("solid_red.png"),
        64,
        64,
        &render_fixtures::render_clear(&device, 64, 64, Color::RED),
    );
    save_png(
        &out_dir.join("solid_blue.png"),
        64,
        64,
        &render_fixtures::render_clear(&device, 64, 64, Color::BLUE),
    );

    let rgb_verts = [
        Vertex2D::new(0.0, -0.8, Color::RED),
        Vertex2D::new(-0.8, 0.8, Color::GREEN),
        Vertex2D::new(0.8, 0.8, Color::BLUE),
    ];
    save_png(
        &out_dir.join("rgb_triangle.png"),
        256,
        256,
        &render_fixtures::render_triangle(&device, 256, 256, Color::BLACK, rgb_verts),
    );

    let white_verts = [
        Vertex2D::new(0.0, -0.5, Color::WHITE),
        Vertex2D::new(-0.5, 0.5, Color::WHITE),
        Vertex2D::new(0.5, 0.5, Color::WHITE),
    ];
    save_png(
        &out_dir.join("white_triangle.png"),
        128,
        128,
        &render_fixtures::render_triangle(&device, 128, 128, Color::BLACK, white_verts),
    );

    save_png(
        &out_dir.join("game_of_life_50.png"),
        512,
        512,
        &render_fixtures::render_game_of_life(&device, 50),
    );
    save_png(
        &out_dir.join("game_of_life_100.png"),
        512,
        512,
        &render_fixtures::render_game_of_life(&device, 100),
    );

    save_png(
        &out_dir.join("depth_occlusion.png"),
        64,
        64,
        &render_fixtures::render_depth_occlusion(&device, 64, 64),
    );

    println!("Updated PNGs in {}", out_dir.display());
}

fn save_png(path: &Path, width: u32, height: u32, rgba_data: &[u8]) {
    let img = image::RgbaImage::from_raw(width, height, rgba_data.to_vec()).expect("Failed to create image");
    img.save(path).expect("Failed to save PNG");
    println!("Saved: {}", path.display());
}
