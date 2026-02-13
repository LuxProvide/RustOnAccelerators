#!/bin/bash -l
#SBATCH -p fpga
#SBATCH -q default
#SBATCH -N 1
#SBATCH -t 10:00
#SBATCH -J rust-opencl-fpga
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

source setup_rustfpga.sh

echo "#-------------------Emulation"
echo "Going to rust-opencl-fpga folder"
cd ${CODE_ROOT}/rust-opencl-fpga
echo "Building executable"
AOC="$(which aoc)" AOC_FLAGS="-v -march=emulator -legacy-emulator -board=p520_hpc_m210h_g3x16" cargo build --release
echo "Execute the code the emulation code"
CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./target/release/rust-opencl-fpga -orust-opencl-fpga.png ../../data/original_image.png

echo "#-------------------FPGA execution"
echo "Execute the code the hardware image"
LD_PRELOAD=${JEMALLOC_PRELOAD} FPGA_AOCX_PATH=${HARD_IMAGE} ./target/release/rust-opencl-fpga -orust-opencl-fpga.png ../../data/original_image.png
