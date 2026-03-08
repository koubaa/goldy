//! FLIP-based perceptual image comparison utilities for screenshot tests.
//!
//! Uses NVIDIA's FLIP algorithm to detect perceptually significant differences
//! between rendered output and reference images.

use std::path::Path;

/// Comparison type for FLIP statistics.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum ComparisonType {
    /// Pass if the mean error is less than or equal to the threshold.
    Mean(f32),
    /// Pass if the given percentile is less than or equal to the threshold.
    /// Percentile is in range [0.0, 1.0].
    Percentile { percentile: f32, threshold: f32 },
}

impl ComparisonType {
    #[allow(dead_code)]
    fn check(&self, pool: &mut nv_flip::FlipPool) -> bool {
        match *self {
            ComparisonType::Mean(threshold) => {
                let mean = pool.mean();
                let within = mean <= threshold;
                println!(
                    "    Mean: {:.6} (threshold: {:.6}) - {}",
                    mean,
                    threshold,
                    if within { "PASS" } else { "FAIL" }
                );
                within
            }
            ComparisonType::Percentile {
                percentile,
                threshold,
            } => {
                let value = pool.get_percentile(percentile, true);
                let within = value <= threshold;
                println!(
                    "    {}%: {:.6} (threshold: {:.6}) - {}",
                    percentile * 100.0,
                    value,
                    threshold,
                    if within { "PASS" } else { "FAIL" }
                );
                within
            }
        }
    }
}

/// Error type for image comparison failures.
#[derive(Debug)]
#[allow(dead_code)]
pub enum ImageComparisonError {
    /// Reference image not found at the specified path.
    ReferenceNotFound(String),
    /// Reference image has wrong dimensions.
    DimensionMismatch {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    /// Reference image has wrong format.
    FormatMismatch(String),
    /// IO error reading/writing images.
    IoError(std::io::Error),
    /// FLIP comparison failed - images are perceptually different.
    ComparisonFailed {
        actual_path: String,
        difference_path: String,
    },
}

impl std::fmt::Display for ImageComparisonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImageComparisonError::ReferenceNotFound(path) => {
                write!(f, "Reference image not found: {}", path)
            }
            ImageComparisonError::DimensionMismatch {
                expected_width,
                expected_height,
                actual_width,
                actual_height,
            } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}x{}, got {}x{}",
                    expected_width, expected_height, actual_width, actual_height
                )
            }
            ImageComparisonError::FormatMismatch(msg) => {
                write!(f, "Format mismatch: {}", msg)
            }
            ImageComparisonError::IoError(e) => {
                write!(f, "IO error: {}", e)
            }
            ImageComparisonError::ComparisonFailed {
                actual_path,
                difference_path,
            } => {
                write!(
                    f,
                    "Image comparison failed. Actual: {}, Difference: {}",
                    actual_path, difference_path
                )
            }
        }
    }
}

impl std::error::Error for ImageComparisonError {}

impl From<std::io::Error> for ImageComparisonError {
    fn from(e: std::io::Error) -> Self {
        ImageComparisonError::IoError(e)
    }
}

/// Read a PNG file and return its RGBA pixel data.
#[allow(dead_code)]
fn read_png(
    path: &Path,
    expected_width: u32,
    expected_height: u32,
) -> Result<Vec<u8>, ImageComparisonError> {
    let data = std::fs::read(path)
        .map_err(|_| ImageComparisonError::ReferenceNotFound(path.display().to_string()))?;

    let decoder = png::Decoder::new(std::io::Cursor::new(data));
    let mut reader = decoder.read_info().map_err(|e| {
        ImageComparisonError::FormatMismatch(format!("Failed to read PNG header: {}", e))
    })?;

    let buffer_len = reader
        .output_buffer_size()
        .expect("output buffer size should be known after reading info");
    let mut buffer = vec![0u8; buffer_len];
    let info = reader.next_frame(&mut buffer).map_err(|e| {
        ImageComparisonError::FormatMismatch(format!("Failed to decode PNG: {}", e))
    })?;

    if info.width != expected_width || info.height != expected_height {
        return Err(ImageComparisonError::DimensionMismatch {
            expected_width,
            expected_height,
            actual_width: info.width,
            actual_height: info.height,
        });
    }

    if info.color_type != png::ColorType::Rgba {
        return Err(ImageComparisonError::FormatMismatch(format!(
            "Expected RGBA, got {:?}",
            info.color_type
        )));
    }

    if info.bit_depth != png::BitDepth::Eight {
        return Err(ImageComparisonError::FormatMismatch(format!(
            "Expected 8-bit depth, got {:?}",
            info.bit_depth
        )));
    }

    Ok(buffer)
}

/// Write RGBA pixel data to a PNG file.
#[allow(dead_code)]
fn write_png(
    path: &Path,
    width: u32,
    height: u32,
    data: &[u8],
    compression: png::Compression,
) -> Result<(), ImageComparisonError> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);

    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(compression);

    let mut writer = encoder.write_header().map_err(|e| {
        ImageComparisonError::IoError(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to write PNG header: {}", e),
        ))
    })?;

    writer.write_image_data(data).map_err(|e| {
        ImageComparisonError::IoError(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Failed to write PNG data: {}", e),
        ))
    })?;

    Ok(())
}

/// Remove alpha channel from RGBA data to get RGB data for FLIP.
fn remove_alpha(input: &[u8]) -> Vec<u8> {
    input
        .chunks_exact(4)
        .flat_map(|chunk| &chunk[0..3])
        .copied()
        .collect()
}

/// Add alpha channel (255) to RGB data to get RGBA for PNG output.
fn add_alpha(input: &[u8]) -> Vec<u8> {
    input
        .chunks_exact(3)
        .flat_map(|chunk| [chunk[0], chunk[1], chunk[2], 255])
        .collect()
}

/// Print FLIP statistics for debugging.
#[allow(dead_code)]
fn print_flip_stats(pool: &mut nv_flip::FlipPool) {
    println!("  FLIP Statistics:");
    println!("    Min: {:.6}", pool.min_value());
    println!("    Mean: {:.6}", pool.mean());
    for percentile in [25, 50, 75, 95, 99] {
        println!(
            "    {}%: {:.6}",
            percentile,
            pool.get_percentile(percentile as f32 / 100.0, true)
        );
    }
    println!("    Max: {:.6}", pool.max_value());
}

/// Compare rendered output against a reference image using FLIP.
///
/// # Arguments
/// * `reference_path` - Path to the reference PNG image
/// * `width` - Expected width of both images
/// * `height` - Expected height of both images
/// * `actual_rgba` - RGBA pixel data from rendering (width * height * 4 bytes)
/// * `checks` - Comparison thresholds that must pass
///
/// # Returns
/// * `Ok(())` if all checks pass
/// * `Err(ImageComparisonError::ComparisonFailed)` if any check fails, with paths to debug images
#[allow(dead_code)]
pub fn compare_images(
    reference_path: &Path,
    width: u32,
    height: u32,
    actual_rgba: &[u8],
    checks: &[ComparisonType],
) -> Result<(), ImageComparisonError> {
    assert_eq!(
        actual_rgba.len(),
        (width * height * 4) as usize,
        "Actual image data has wrong size"
    );

    // Load reference image
    let reference_rgba = read_png(reference_path, width, height)?;

    // Convert to RGB for FLIP (it doesn't use alpha)
    let reference_rgb = remove_alpha(&reference_rgba);
    let actual_rgb = remove_alpha(actual_rgba);

    // Create FLIP images
    let reference_flip = nv_flip::FlipImageRgb8::with_data(width, height, &reference_rgb);
    let actual_flip = nv_flip::FlipImageRgb8::with_data(width, height, &actual_rgb);

    // Compute FLIP error map
    let error_map = nv_flip::flip(
        reference_flip,
        actual_flip,
        nv_flip::DEFAULT_PIXELS_PER_DEGREE,
    );

    // Gather statistics
    let mut pool = nv_flip::FlipPool::from_image(&error_map);

    println!("Comparing against reference: {}", reference_path.display());
    print_flip_stats(&mut pool);

    // Run all checks
    let mut all_passed = !checks.is_empty();
    println!("  Checks:");
    for check in checks {
        all_passed &= check.check(&mut pool);
    }

    if all_passed {
        println!("  Result: PASS");
        return Ok(());
    }

    // Comparison failed - write debug images
    let stem = reference_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let parent = reference_path.parent().unwrap_or(Path::new("."));

    let actual_path = parent.join(format!("{}-actual.png", stem));
    let difference_path = parent.join(format!("{}-diff.png", stem));

    // Write actual image
    write_png(
        &actual_path,
        width,
        height,
        actual_rgba,
        png::Compression::Fast,
    )?;

    // Convert error map to magma colormap and write difference image
    let magma_rgb = error_map.apply_color_lut(&nv_flip::magma_lut()).to_vec();
    let magma_rgba = add_alpha(&magma_rgb);
    write_png(
        &difference_path,
        width,
        height,
        &magma_rgba,
        png::Compression::Fast,
    )?;

    println!("  Result: FAIL");
    println!("    Actual image saved to: {}", actual_path.display());
    println!("    Difference map saved to: {}", difference_path.display());

    Err(ImageComparisonError::ComparisonFailed {
        actual_path: actual_path.display().to_string(),
        difference_path: difference_path.display().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_alpha() {
        let rgba = vec![255, 128, 64, 255, 0, 0, 0, 128];
        let rgb = remove_alpha(&rgba);
        assert_eq!(rgb, vec![255, 128, 64, 0, 0, 0]);
    }

    #[test]
    fn test_add_alpha() {
        let rgb = vec![255, 128, 64, 0, 0, 0];
        let rgba = add_alpha(&rgb);
        assert_eq!(rgba, vec![255, 128, 64, 255, 0, 0, 0, 255]);
    }
}
