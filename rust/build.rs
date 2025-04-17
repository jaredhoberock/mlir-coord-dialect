use std::path::Path;
use std::process::Command;

fn main() {
    let dialect_dir = Path::new("..");

    // build the C++ plugin
    let status = Command::new("make")
        .current_dir(dialect_dir)
        .status()
        .expect("Failed to build coord dialect C++");

    if !status.success() {
        panic!("C++ build failed");
    }

    // link against the shared library
    println!("cargo:rustc-link-search=native={}", dialect_dir.display());
    println!("cargo:rustc-link-lib=dylib=coord_dialect");

    // trigger rebuild if any files in the dialect directory change
    println!("cargo:rerun-if-changed={}", dialect_dir.display());
}
