use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    println!("cargo:rerun-if-changed=kernels/my_kernel.cl");
    println!("cargo:rerun-if-env-changed=AOC");
    println!("cargo:rerun-if-env-changed=AOC_FLAGS");

    let kernel_name = "conv2d_gray_f32";

    // Optional path to the aoc compiler (defaults to "aoc").
    let aoc = env::var("AOC").unwrap_or_else(|_| "aoc".to_string());
    // Optional flags passed to aoc.
    let aoc_flags = env::var("AOC_FLAGS").unwrap_or_default();

    let kernel_filename = format!("kernels/{}.cl", kernel_name);
    let kernel_src = Path::new(kernel_filename.as_str());

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let built = out_dir.join(format!("{}.aocx", kernel_name));

    // Build the FPGA bitstream into OUT_DIR.
    let mut cmd = Command::new(aoc);

    if !aoc_flags.trim().is_empty() {
        for part in aoc_flags.split_whitespace() {
            cmd.arg(part);
        }
    }

    // Invoke aoc on the kernel source.
    cmd.arg(kernel_src).arg("-o").arg(&built);

    eprintln!("Running: {:?}", cmd);

    let status = cmd.status().expect("failed to run aoc");
    if !status.success() {
        panic!("aoc failed: {status}");
    }

    // Copy into a stable location under target/ (or CARGO_TARGET_DIR if set).
    let target_dir = env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string());
    let profile = env::var("PROFILE").expect("PROFILE not set"); // "debug" or "release"

    let stable_dir = Path::new(&target_dir).join("aoc").join(&profile);
    fs::create_dir_all(&stable_dir).expect("failed to create stable aoc dir");

    let stable_path = stable_dir.join(format!("{}.aocx", kernel_name));
    fs::copy(&built, &stable_path).expect("failed to copy aoc output");

    // Tell the host code where to find the bitstream.
    println!("cargo:rustc-env=FPGA_AOCX_PATH={}", stable_path.display());
}
