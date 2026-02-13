# A Rust host code executing OpenCL kernel for FPGA 

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenCL_logo.svg/960px-OpenCL_logo.svg.png){align=right width=30%}

- We are going to build the code in **emulation** mode
- We are going to run with a prepare FPGA image on the real FPGA cards

## Host code

```rust title="./code/rust-opencl-fpga/src/main.rs" linenums="1"
--8<-- "./code/rust-opencl-fpga/src/main.rs"
```

## Device code

```cpp title="./code/rust-opencl-fpga/kernels/conv2d_gray_f32.cl" linenums="1"
--8<-- "./code/rust-opencl-fpga/kernels/conv2d_gray_f32.cl"
```

```rust title="./code/rust-opencl-fpga/build.rs./code/rust-opencl-fpga/build.rs" linenums="1"
--8<-- "./code/rust-opencl-fpga/build.rs"
```


## Execution on MeluXina

!!! warning "What about memory accesses in FPGA ? "
    * In order to use **Direct Memory Access (DMA)**, we need to setup proper data alignment or the offline compiler will output the following warnings:
    ```bash
    Running on device: p520_hpc_m210h_g3x16 : BittWare Stratix 10 MX OpenCL platform (aclbitt_s10mx_pcie0)
    add two vectors of size 256
    ** WARNING: [aclbitt_s10mx_pcie0] NOT using DMA to transfer 1024 bytes from host to device because of lack of alignment
    **                 host ptr (0xb60b350) and/or dev offset (0x400) is not aligned to 64 bytes
    ** WARNING: [aclbitt_s10mx_pcie0] NOT using DMA to transfer 1024 bytes from host to device because of lack of alignment
    **                 host ptr (0xb611910) and/or dev offset (0x800) is not aligned to 64 bytes
    ** WARNING: [aclbitt_s10mx_pcie0] NOT using DMA to transfer 1024 bytes from device to host because of lack of alignment
    **                 host ptr (0xb611d20) and/or dev offset (0xc00) is not aligned to 64 bytes
    ``` 
    To do so, we use [jemalloc](https://jemalloc.net):
    ```bash
    module load jemalloc
    export JEMALLOC_PRELOAD=$(jemalloc-config --libdir)/libjemalloc.so.$(jemalloc-config --revision)
    LD_PRELOAD=${JEMALLOC_PRELOAD} ./exe
    ```

### Interactive execution
 
```bash linenums="1"
salloc -A <project_name> --reservation=<reservation_name> -t 30:00 -q default -p fpga
cd RustOnAccelerators/code
source setup_rustfpga.sh
```

=== "Building for Emulation"
    ```bash linenums="1"
    cd ${CODE_ROOT}/rust-opencl-fpga 
    AOC="$(which aoc)" AOC_FLAGS="-v -march=emulator -legacy-emulator -board=p520_hpc_m210h_g3x16" cargo build --release
    # Execute the code
    CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1 ./target/release/rust-opencl-fpga -orust-opencl-fpga.png ../../data/original_image.png
    ```

=== "Execution on FPGA card"
    ```bash linenums="1"
    cd ${CODE_ROOT}/rust-opencl-fpga 
    # Execute the code
    LD_PRELOAD=${JEMALLOC_PRELOAD} FPGA_AOCX_PATH=${HARD_IMAGE} ./target/release/rust-opencl-fpga -orust-opencl-fpga.png ../../data/original_image.png
    ```

### Batch execution

```bash
cd RustOnAccelerators/code
sbatch -A <project_name> --reservation=<reservation_name> launcher-rust-opencl-fpga.sh
```
## Results


- You should see the following results for both executions:


| <center markdown="1">![](./images/original_image.png)</center> | <center markdown="1">![](./images/rust-nvcc-cuda.png)</center>|
|----------------------------------------------------------------|---------------------------------------------------------------|
| <center>Original</center>                                      | <center>Convolution</center>                                  |


## Explore Further

- Modify the host and the device code to run on GPU using OpenCL

!!! info
    Look inside the `code` folder for the solutions. We have already prepared the solution for you.


