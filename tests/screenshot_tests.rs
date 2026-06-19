//! Screenshot tests for Goldy examples using FLIP perceptual image comparison.
//!
//! Legacy TaskGraph path. Scheme coverage: `scheme_screenshot_tests.rs`.
//! Delete this file when ekrano migrates (Phase 2).
//!
//! These tests render examples to offscreen targets and compare them against
//! reference PNG images using NVIDIA's FLIP algorithm.
//!
#![cfg(any(feature = "vulkan", feature = "dx12", feature = "metal"))]
//! ## Running Tests
//!
//! ```bash
//! cargo test --test screenshot_tests
//! ```
//!
//! ## Creating / updating reference images
//!
//! Run the `update-screenshots` binary (not part of `cargo test`):
//!
//! ```bash
//! cargo run --bin update-screenshots --features update-screenshots
//! ```

mod common;

#[path = "common/render_fixtures.rs"]
mod render_fixtures;

use std::path::Path;

use common::image::{compare_images, ComparisonType, ImageComparisonError};
use goldy::{Color, Vertex2D};
use render_fixtures::{create_device, render_clear, render_depth_occlusion, render_game_of_life, render_triangle};

fn run_screenshot_test(
    name: &str,
    reference_path: &str,
    width: u32,
    height: u32,
    comparisons: &[ComparisonType],
    pixels: Vec<u8>,
) {
    println!("Running screenshot test: {}", name);

    let reference_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(reference_path);

    match compare_images(&reference_path, width, height, &pixels, comparisons) {
        Ok(()) => {
            println!("Screenshot test '{}' passed!", name);
        }
        Err(ImageComparisonError::ReferenceNotFound(path)) => {
            panic!(
                "Reference image not found: {}\n\
                 To create it, run: cargo run --bin update-screenshots --features update-screenshots",
                path
            );
        }
        Err(e) => {
            panic!("Screenshot test '{}' failed: {}", name, e);
        }
    }
}

/// Test rendering a solid red color.
// Legacy TaskGraph — migrated: `scheme_solid_red`
#[test]
fn test_solid_red() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = render_clear(&device, 64, 64, Color::RED);
    run_screenshot_test(
        "solid_red",
        "tests/screenshots/solid_red.png",
        64,
        64,
        &[ComparisonType::Mean(0.001)],
        pixels,
    );
}

/// Test rendering a solid blue color.
// Legacy TaskGraph — migrated: `scheme_solid_blue`
#[test]
fn test_solid_blue() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = render_clear(&device, 64, 64, Color::BLUE);
    run_screenshot_test(
        "solid_blue",
        "tests/screenshots/solid_blue.png",
        64,
        64,
        &[ComparisonType::Mean(0.001)],
        pixels,
    );
}

/// Test rendering the classic RGB triangle.
// Legacy TaskGraph — migrated: `scheme_rgb_triangle`
#[test]
fn test_rgb_triangle() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let vertices = [
        Vertex2D::new(0.0, -0.8, Color::RED),
        Vertex2D::new(-0.8, 0.8, Color::GREEN),
        Vertex2D::new(0.8, 0.8, Color::BLUE),
    ];

    let pixels = render_triangle(&device, 256, 256, Color::BLACK, vertices);
    run_screenshot_test(
        "rgb_triangle",
        "tests/screenshots/rgb_triangle.png",
        256,
        256,
        &[ComparisonType::Mean(0.02)],
        pixels,
    );
}

/// Test rendering a white triangle on black background.
// Legacy TaskGraph — migrated: `scheme_white_triangle`
#[test]
fn test_white_triangle() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let vertices = [
        Vertex2D::new(0.0, -0.5, Color::WHITE),
        Vertex2D::new(-0.5, 0.5, Color::WHITE),
        Vertex2D::new(0.5, 0.5, Color::WHITE),
    ];

    let pixels = render_triangle(&device, 128, 128, Color::BLACK, vertices);
    run_screenshot_test(
        "white_triangle",
        "tests/screenshots/white_triangle.png",
        128,
        128,
        &[ComparisonType::Mean(0.01)],
        pixels,
    );
}

/// Test Game of Life at update 50.
// Legacy TaskGraph — migrated: `scheme_game_of_life_update_50`
#[test]
fn test_game_of_life_update_50() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = render_game_of_life(&device, 50);
    run_screenshot_test(
        "game_of_life_50",
        "tests/screenshots/game_of_life_50.png",
        512,
        512,
        &[ComparisonType::Mean(0.012)],
        pixels,
    );
}

/// Test Game of Life at update 100.
// Legacy TaskGraph — migrated: `scheme_game_of_life_update_100`
#[test]
fn test_game_of_life_update_100() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = render_game_of_life(&device, 100);
    run_screenshot_test(
        "game_of_life_100",
        "tests/screenshots/game_of_life_100.png",
        512,
        512,
        &[ComparisonType::Mean(0.012)],
        pixels,
    );
}

/// Depth occlusion test: a near (red) geometry blocks a far (green) geometry.
// Legacy TaskGraph — migrated: `scheme_depth_occlusion`
#[test]
fn test_depth_occlusion() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let pixels = render_depth_occlusion(&device, 64, 64);
    run_screenshot_test(
        "depth_occlusion",
        "tests/screenshots/depth_occlusion.png",
        64,
        64,
        &[ComparisonType::Mean(0.001)],
        pixels,
    );
}
