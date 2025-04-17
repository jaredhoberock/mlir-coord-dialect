use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Get absolute path to the parent of the crate (i.e., the dialect root)
    let dialect_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .expect("Could not find parent directory")
        .to_path_buf();

    // Always run make
    let status = Command::new("make")
        .arg("-j")
        .current_dir(&dialect_dir)
        .status()
        .expect("Failed to run make");

    if !status.success() {
        panic!("C++ build failed");
    }

    // Link against the shared library
    println!("cargo:rustc-link-search=native={}", dialect_dir.display());
    println!("cargo:rustc-link-lib=dylib=coord_dialect");
}
