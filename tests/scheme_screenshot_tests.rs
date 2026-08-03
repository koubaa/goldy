//! Scheme screenshot tests.
//!
//! Reuses the same reference PNGs as the former TaskGraph screenshot suite.
#![cfg(all(feature = "graphics", any(feature = "vulkan", feature = "dx12", feature = "metal")))]

mod common;

#[path = "common/gol_state.rs"]
mod gol_state;
#[path = "common/scheme_render.rs"]
mod scheme_render;
#[path = "common/scheme_render_fixtures.rs"]
mod scheme_render_fixtures;

use std::path::Path;

use common::image::{compare_images, ComparisonType, ImageComparisonError};
use goldy::Color;
use goldy::Vertex2D;
use scheme_render_fixtures::{
    scheme_render_clear, scheme_render_depth_occlusion, scheme_render_game_of_life, scheme_render_triangle,
};

fn run_screenshot_test(
    name: &str,
    reference_path: &str,
    width: u32,
    height: u32,
    comparisons: &[ComparisonType],
    pixels: Vec<u8>,
) {
    println!("Running scheme screenshot test: {}", name);

    let reference_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(reference_path);

    match compare_images(&reference_path, width, height, &pixels, comparisons) {
        Ok(()) => {
            println!("Scheme screenshot test '{}' passed!", name);
        }
        Err(ImageComparisonError::ReferenceNotFound(path)) => {
            panic!(
                "Reference image not found: {}\n\
                 To create it, run: cargo run --bin update-screenshots --features update-screenshots",
                path
            );
        }
        Err(e) => {
            panic!("Scheme screenshot test '{}' failed: {}", name, e);
        }
    }
}

#[test]
fn scheme_solid_red() {
    let Some(device) = scheme_render_fixtures::create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = scheme_render_clear(&device, 64, 64, Color::RED);
    run_screenshot_test(
        "solid_red",
        "tests/screenshots/solid_red.png",
        64,
        64,
        &[ComparisonType::Mean(0.001)],
        pixels,
    );
}

#[test]
fn scheme_solid_blue() {
    let Some(device) = scheme_render_fixtures::create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = scheme_render_clear(&device, 64, 64, Color::BLUE);
    run_screenshot_test(
        "solid_blue",
        "tests/screenshots/solid_blue.png",
        64,
        64,
        &[ComparisonType::Mean(0.001)],
        pixels,
    );
}

#[test]
fn scheme_rgb_triangle() {
    let Some(device) = scheme_render_fixtures::create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let vertices = [
        Vertex2D::new(0.0, -0.8, Color::RED),
        Vertex2D::new(-0.8, 0.8, Color::GREEN),
        Vertex2D::new(0.8, 0.8, Color::BLUE),
    ];

    let pixels = scheme_render_triangle(&device, 256, 256, Color::BLACK, vertices);
    run_screenshot_test(
        "rgb_triangle",
        "tests/screenshots/rgb_triangle.png",
        256,
        256,
        &[ComparisonType::Mean(0.02)],
        pixels,
    );
}

#[test]
fn scheme_white_triangle() {
    let Some(device) = scheme_render_fixtures::create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let vertices = [
        Vertex2D::new(0.0, -0.5, Color::WHITE),
        Vertex2D::new(-0.5, 0.5, Color::WHITE),
        Vertex2D::new(0.5, 0.5, Color::WHITE),
    ];

    let pixels = scheme_render_triangle(&device, 128, 128, Color::BLACK, vertices);
    run_screenshot_test(
        "white_triangle",
        "tests/screenshots/white_triangle.png",
        128,
        128,
        &[ComparisonType::Mean(0.01)],
        pixels,
    );
}

#[test]
fn scheme_game_of_life_update_50() {
    let Some(device) = scheme_render_fixtures::create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = scheme_render_game_of_life(&device, 50);
    run_screenshot_test(
        "game_of_life_50",
        "tests/screenshots/game_of_life_50.png",
        512,
        512,
        &[ComparisonType::Mean(0.012)],
        pixels,
    );
}

#[test]
fn scheme_game_of_life_update_100() {
    let Some(device) = scheme_render_fixtures::create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = scheme_render_game_of_life(&device, 100);
    run_screenshot_test(
        "game_of_life_100",
        "tests/screenshots/game_of_life_100.png",
        512,
        512,
        &[ComparisonType::Mean(0.012)],
        pixels,
    );
}

#[test]
fn scheme_depth_occlusion() {
    let Some(device) = scheme_render_fixtures::create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let Some(pixels) = scheme_render_depth_occlusion(&device, 64, 64) else {
        eprintln!("Skipping test: CUDA raster has no depth (first slice)");
        return;
    };
    run_screenshot_test(
        "depth_occlusion",
        "tests/screenshots/depth_occlusion.png",
        64,
        64,
        &[ComparisonType::Mean(0.001)],
        pixels,
    );
}
