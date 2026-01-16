#!/bin/bash -l

check_mel3_node() {
    local host
    host=$(hostname -s)

    if [[ ! $host =~ ^mel3 ]]; then
        echo "Error: This script must be sourced on an FPGA  node (current: $host)" >&2
        exit 1
    fi
}

check_mel3_node


module load util-linux ifpgasdk 520nmx

export CARGO_HOME="$LOCALSCRATCH/${USER}/cargo"
export RUSTUP_HOME="$LOCALSCRATCH/${USER}/rustup"
echo "Setting CARGO_HOME=${CARGO_HOME}"
echo "Setting RUSTUP_HOME=${RUSTUP_HOME}"
if [[ ! -f "${CARGO_HOME}/bin/rustup" ]]; then
 echo "Installing Rust"
 curl -sSf -L https://sh.rustup.rs | bash -s -- -y --no-modify-path --profile minimal --default-toolchain none
fi

. "${CARGO_HOME}/env"











