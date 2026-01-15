# Rust on HPC accelerators: use-cases on MeluXina GPU and FPGA 

## Introduction

High-performance computing on accelerators (GPUs, TPUs, FPGAs) demands extreme performance and correctness, yet traditional accelerator code relies heavily on C/C++ with manual memory management and undefined behavior risks. Rust offers a compelling alternative by enforcing memory safety, data-race freedom, and clear ownership semantics at compile time, without sacrificing low-level control. This is especially valuable in heterogeneous systems where host–device interactions, lifetimes, and synchronization are common sources of bugs. Rust’s zero-cost abstractions allow expressive high-level code that compiles down to predictable machine instructions. Its strong type system makes illegal states unrepresentable, reducing entire classes of runtime errors. For accelerator programming, Rust integrates cleanly with existing ecosystems (CUDA, HIP, OpenCL) via FFI while enabling safer orchestration on the host. As HPC systems scale in complexity and concurrency, Rust helps shift correctness checks from runtime to compile time—exactly where HPC needs them most.


## Content

In this course, you will learn to:

1. How to use the [opencl3](https://crates.io/crates/opencl3) crate to submit kernel on GPUs and FPGA 

2. How to use the [cust](https://crates.io/crates/cust) crate to bind to the CUDA Driver API

3. How to use [rust-cuda](https://github.com/Rust-GPU/rust-cuda) and the [cust](https://crates.io/crates/cust) to write and execute host and kernels with Rust

## Who is the course for

This course is for students, researchers, engineers wishing to discover how to use Rust to program GPUs/FPGAs.

This course is **NOT** a Rust/CUDA/OpenCL3 programming course but intends to show you how to use Rust with Meluxina's accelerators.

We strongly recommend to interested participants the following resources:

- [The Rust book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [OpenCL3 reference guide](https://www.khronos.org/files/opencl30-reference-guide.pdf)
- [CUDA programming]()

## About this course

This course has been developed by the Supercomputing Application Services group at LuxProvide.

© 2026 LuxProvide, All rights reserved. 
