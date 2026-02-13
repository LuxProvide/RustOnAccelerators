#!/bin/bash -l

module --force purge
module load env/release/2024.1
module load Clang
module load CUDA

export CODE_ROOT=$(realpath $PWD)
export LLVM7_ROOT="/mnt/tier2/project/lxp/ekieffer/Training/Rust-CUDA/LLVM-7.1.0"
export CARGO_HOME="$LOCALSCRATCH/${USER}/cargo"
export RUSTUP_HOME="$LOCALSCRATCH/${USER}/rustup"

echo "Setting LLVM7_ROOT=${LLVM7_ROOT}"
echo "Setting CARGO_HOME=${CARGO_HOME}"
echo "Setting RUSTUP_HOME=${RUSTUP_HOME}"

if [[ ! -d "${LLVM7_ROOT}" ]]; then
  echo "Installing LLVM7 into ${LLVM7_ROOT}"
  echo "Loading required mdoules"
  module load Ninja CMake Python
  CWD=${PWD}
  cd /tmp
  curl -sSf -L -O https://github.com/llvm/llvm-project/releases/download/llvmorg-7.1.0/llvm-7.1.0.src.tar.xz
  tar -xf llvm-7.1.0.src.tar.xz && cd llvm-7.1.0.src
  mkdir -p build && cd build
  ARCH=$(uname -m)
  if [ "$ARCH" = "x86_64" ]; then
    TARGETS="X86;NVPTX"
  else
    TARGETS="AArch64;NVPTX"
  fi
  cmake -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_TARGETS_TO_BUILD="$TARGETS" \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_ENABLE_ZLIB=ON \
    -DLLVM_ENABLE_TERMINFO=ON \
    -DCMAKE_INSTALL_PREFIX=${LLVM7_ROOT} ..
  make -j$(nproc)
  make install
  cd ${CWD}
fi

echo "Found LLVM7 in ${LLVM7_ROOT}"
export PATH="${LLVM7_ROOT}/bin":${PATH}
export CPATH="${LLVM7_ROOT}/include":${CPATH}
export LD_LIBRARY_PATH="${LLVM7_ROOT}/include":${LD_LIBRARY_PATH}

if [[ ! -f "${CARGO_HOME}/bin/rustup" ]]; then
  echo "Installing Rust"
  curl -sSf -L https://sh.rustup.rs | bash -s -- -y --no-modify-path --profile minimal --default-toolchain none
fi

export LD_LIBRARY_PATH="${EBROOTCUDA}/nvvm/lib64:${LD_LIBRARY_PATH}"
export LLVM_LINK_STATIC=2
export RUST_LOG=info

. "${CARGO_HOME}/env"
