#!/bin/bash -l
#SBATCH -p gpu
#SBATCH -q default
#SBATCH -N 1
#SBATCH -t 10:00
#SBATCH -G 1
#SBATCH -J rust-nvcc-cuda
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

source setup_rustgpu.sh

#-------------------rust-nvcc
echo "Going to rust-nvcc folder"
cd ${CODE_ROOT}/rust-nvcc
echo "Building executable"
cargo build --release
echo "Execute rust-nvcc"
./target/release/rust-nvcc -orust-nvcc-image.png ../../data/original_image.png

#-------------------rust-cuda
echo "Going to rust-cuda folder"
cd ${CODE_ROOT}/rust-cuda
echo "Building executable"
cargo build --release
echo "Execute rust-cuda"
./target/release/rust-cuda -orust-cuda-image.png ../../data/original_image.png
