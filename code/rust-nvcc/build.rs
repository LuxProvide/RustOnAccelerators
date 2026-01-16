use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let kernel_src = manifest_dir.join("kernels").join("vec_add.cu");

    // Où Cargo met les artefacts de build pour cette cible/profil
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_out = out_dir.join("vec_add.ptx");

    // Rebuild si le kernel change
    println!("cargo:rerun-if-changed={}", kernel_src.display());

    // Compile en PTX
    // -ptx : sortie PTX
    // -O3 : optimisation
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

    // Optionnel: expose le chemin du PTX au code Rust (via env compile-time)
    println!("cargo:rustc-env=VEC_ADD_PTX={}", ptx_out.display());
}

