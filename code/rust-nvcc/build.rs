use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Absolute path to this crate's directory (set by Cargo).
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // CUDA kernel source file compiled by this build script.
    let kernel_src = manifest_dir.join("kernels").join("conv2d_gray_f32.cu");

    // Cargo-provided output directory for build artifacts.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // Destination PTX path consumed at runtime via include_str!.
    let ptx_out = out_dir.join("conv2d_gray_f32.ptx");

    // Re-run this build script when the kernel source changes.
    println!("cargo:rerun-if-changed={}", kernel_src.display());

    // Compile CUDA source to PTX with optimization enabled.
    let status = Command::new("nvcc")
        .arg("-ptx")
        .arg("-O3")
        .arg(kernel_src.as_os_str())
        .arg("-o")
        .arg(ptx_out.as_os_str())
        .status()
        .expect("Failed to run nvcc. Is CUDA toolkit installed and nvcc in PATH?");

    if !status.success() {
        panic!("nvcc failed compiling {}", kernel_src.display());
    }
}
