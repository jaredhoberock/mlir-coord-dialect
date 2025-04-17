use std::path::Path;
use std::process::Command;

fn main() {
    let dialect_dir = Path::new("..");

    // Always run `make`. It handles incremental rebuilds itself.
    let status = Command::new("make")
        .arg("-j")
        .current_dir(dialect_dir)
        .status()
        .expect("Failed to run make");

    if !status.success() {
        panic!("C++ build failed");
    }

    // Link against the shared library
    println!("cargo:rustc-link-search=native={}", dialect_dir.display());
    println!("cargo:rustc-link-lib=dylib=coord_dialect");
}
