# A Rust host code executing OpenCL kernel for FPGA 

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/OpenCL_logo.svg/960px-OpenCL_logo.svg.png){align=right width=30%}

We are first going to build the code in **emulation** mode and then, run a pre-build FPGA image on the hardware FPGA cards.

!!! warning "Emulation vs Synthesis"
    - **FPGA emulation** refers to the process of using a software or hardware system to mimic the behaviour of an FPGA device.
    - This is usually done to test, validate, and debug FPGA designs before deploying them on actual hardware. 
    - **FPGA synthesis** defaults to building the entire chip every time which can take hours depending on the kernel size. For this reason, we have **already** done the FPGA synthesis for you.

## Host code

- Line 1-12: we import the necessary modules. The **prelude** in Rust is a module that re-exports commonly used types and traits so you can import them all at once.
- Line 17-115: the `OUT_DIR` environment variable contains the path to the device code build using `build.rs`.
- Line 9-81: the run function applies the same kernel (3 X 3):

      $$
        \begin{bmatrix}
          0 & 1  & 0\\\
          1 & -4 & 1\\\
          0 & 1 & 0
        \end{bmatrix}
      $$

- We used the [opencl3](https://crates.io/crates/opencl3) crate, **a Rust implementation of the Khronos OpenCL 3.0 API and extensions**.


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

!!! note "What about memory accesses in FPGA ? "
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
    To do so, we use [jemalloc](https://jemalloc.net/):
    ```bash
    module load jemalloc
    export JEMALLOC_PRELOAD=$(jemalloc-config --libdir)/libjemalloc.so.$(jemalloc-config --revision)
    LD_PRELOAD=${JEMALLOC_PRELOAD} ./exe
    ```

You can choose to execute the build and execute the code within an **1. interactive session** or by **2. submitting a batch job**. 

!!! note "Interactive session"
    - Once you have executed the code, please release the node as you won't need it anymore. 
    - To do so, use `CTRL-d`.

### **1. Interactive execution**

- Apply the following commands:
 
```bash linenums="1"
salloc -A <project_name> -t 30:00 -q default -p fpga
cd ${HOME}/RustOnAccelerators/code
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

### **2. Batch execution**

- Make sure you are not **already** on an **interactive session**.
- Then apply the following commands:

```bash
cd ${HOME}/RustOnAccelerators/code
sbatch -A <project_name> launcher-rust-opencl-fpga.sh
```

- The `sbatch` command should output `Submitted batch job <job_id>`.

- Keep in mind the job id, it will be useful to look at the logs.

- To see the status of your job, please use the following command: `squeue -j <job_id>`.

- Check the output with the following command: `cat rust-opencl-fpga-<job_id>.out`.

- You can also check for error with `cat rust-opencl-fpga-<job_id>.err`.


## Results

- To open the resulting images, you can either:

  - Use the `rsync` command from a terminal: 
  ```bash
  rsync -avz -e 'ssh -p 8822' <u1xxxxx>@login.lxp.lu:"RustOnAccelerators/code/**/rust-*.png"` 
  ```

  - Or use the [OpenOnDemand portal](https://portal.lxp.lu/). Once logged in, select "Files" -> "Home Directory" -> "RustOnAccelerators"
  ![](./images/check_images_fpga.png)
  - Click on the image.


- You should see the following results for both executions:


| <center markdown="1">![](./images/original_image.png)</center> | <center markdown="1">![](./images/rust-nvcc-cuda.png)</center>|
|----------------------------------------------------------------|---------------------------------------------------------------|
| <center>Original</center>                                      | <center>Convolution</center>                                  |


## Explore Further

- Modify the host and the device code to run on GPU using OpenCL

!!! info
    Look inside the `code` folder, we have already prepared the solution for you.


