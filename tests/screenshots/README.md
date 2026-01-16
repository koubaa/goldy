# Screenshot Reference Images

This directory contains reference PNG images for FLIP-based screenshot tests.

## Creating Reference Images

Reference images must be created manually:

1. Run the example or test that generates the image
2. Verify the output looks correct visually
3. Save the output as a PNG file in this directory

## Requirements

- Format: RGBA PNG, 8-bit depth
- Dimensions: Must match the test's expected width/height
- Naming: Should match the test name (e.g., `triangle.png` for `test_triangle_screenshot`)

## Updating Reference Images

If rendering changes intentionally:

1. Delete the old reference image
2. Run the test to generate new `-actual.png` output
3. Verify the new output is correct
4. Rename `-actual.png` to the reference name

## Debugging Failed Tests

When a test fails, two debug images are generated:

- `{name}-actual.png` - What was actually rendered
- `{name}-diff.png` - FLIP difference map (magma colormap: blue=similar, red=different)

