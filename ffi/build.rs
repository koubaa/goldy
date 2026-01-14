//! Build script for goldy-ffi that generates the C header using cbindgen.

fn main() {
    // Only regenerate header when source files change
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    // Generate C header
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let output_dir = std::path::Path::new(&crate_dir)
        .join("..")
        .join("cpp")
        .join("include");

    // Create output directory if it doesn't exist
    std::fs::create_dir_all(&output_dir).ok();

    let output_file = output_dir.join("goldy.h");

    // Run cbindgen
    let config =
        cbindgen::Config::from_file("cbindgen.toml").expect("Failed to read cbindgen.toml");

    cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("Failed to generate C header")
        .write_to_file(&output_file);

    println!("cargo:warning=Generated C header at {:?}", output_file);
}
