// Copyright (c) 2021 Via Technology Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use getargs::{Arg, Options};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_ACCELERATOR, Device, get_all_devices};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_event, cl_float, cl_uint};
use opencl3::{Result, platform};
use std::ptr;
use utils::{load_gray_f32, save_gray_f32};

const KERNEL_NAME: &str = "conv2d_gray_f32";

fn run(buffer: &mut [f32], width: u32, height: u32) -> Result<()> {
    let buffer_size = (width * height) as usize;

    // Define kernel
    let ksize: cl_uint = 4;
    let weights: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];

    let platforms: Vec<Platform> = get_platforms()?;
    let intel_fpga_platform = platforms
        .into_iter()
        .find(|p| {
            let name = p.name().unwrap_or_default();
            let vendor = p.vendor().unwrap_or_default();
            // Match either the platform name or vendor string.
            // Tweak these substrings to match what you see on your machine.
            name.to_lowercase().contains("fpga")
        })
        .ok_or_else(|| panic!("Intel FPGA OpenCL platform not found"))?;

    // Find a usable device for this application
    let device_id = *intel_fpga_platform
        .get_devices(CL_DEVICE_TYPE_ACCELERATOR)
        .first()
        .expect("no device found in platform");

    let device = Device::new(device_id);

    println!("Executing on {}", device.name()?);

    // Create a Context on an OpenCL device
    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
        .expect("CommandQueue::create_default failed");

    // Read bistream
    let aocx_path = std::env::var("FPGA_AOCX_PATH")
        .unwrap_or_else(|_| "target/aoc/debug/conv2d_gray_f32.aocx".to_string());
    let aocx = std::fs::read(&aocx_path).unwrap();

    // Create program
    let mut program =
        unsafe { Program::create_from_binary(&context, &[device.id()], &[aocx.as_slice()])? };

    // Build the program for the device
    program.build(&[device.id()], "")?;
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    // Create OpenCL device buffers
    let mut input_b = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, buffer_size, ptr::null_mut())?
    };
    let mut weights_b = unsafe {
        Buffer::<cl_float>::create(
            &context,
            CL_MEM_READ_ONLY,
            (ksize * ksize) as usize,
            ptr::null_mut(),
        )?
    };
    let output_b = unsafe {
        Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, buffer_size, ptr::null_mut())?
    };

    let local_x = 16;
    let local_y = 16;
    let global_x = ((width + local_x - 1) / local_x) * local_x;
    let global_y = ((height + local_y - 1) / local_y) * local_y;

    let w: cl_uint = width;
    let h: cl_uint = height;

    // Blocking write
    let _input_write_event =
        unsafe { queue.enqueue_write_buffer(&mut input_b, CL_BLOCKING, 0, buffer, &[])? };

    // Non-blocking write, wait for y_write_event
    let _weights_write_event =
        unsafe { queue.enqueue_write_buffer(&mut weights_b, CL_NON_BLOCKING, 0, &weights, &[])? };

    // Use the ExecuteKernel builder to set the kernel buffer and
    // cl_float value arguments, before setting the one dimensional
    // global_work_size for the call to enqueue_nd_range.
    // Unwraps the Result to get the kernel execution event.
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&input_b)
            .set_arg(&output_b)
            .set_arg(&weights_b)
            .set_arg(&w)
            .set_arg(&h)
            .set_arg(&ksize)
            .set_global_work_sizes(&[global_x as usize, global_y as usize])
            .set_local_work_sizes(&[local_x as usize, local_y as usize])
            .set_wait_event(&_weights_write_event)
            .enqueue_nd_range(&queue)?
    };

    let events: Vec<cl_event> = vec![kernel_event.get()];

    let read_event =
        unsafe { queue.enqueue_read_buffer(&output_b, CL_NON_BLOCKING, 0, buffer, &events)? };

    // Wait for the read_event to complete.
    read_event.wait()?;

    // Calculate the kernel duration, from the kernel_event
    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let duration = end_time - start_time;
    println!("kernel execution duration (ns): {}", duration);

    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();

    if args.is_empty() {
        args.push(String::from("--help")); // help the user out :)
    }
    let mut opts = Options::new(args.iter().map(String::as_str));
    let mut output_path = None;

    while let Some(arg) = opts.next_arg().expect("argument parsing error") {
        match arg {
            Arg::Short('h') | Arg::Long("help") => {
                eprintln!(
                    r"Usage: rust-opencl-gpu [OPTIONS/ARGS] input ...
                     This command execute an OpenCL Convolution kernel on GPU.
                     -h, --help   display this help and exit
                     -o, --output path to record output image"
                );
            }
            Arg::Short('o') | Arg::Long("output") => {
                output_path = opts.value_opt();
            }
            Arg::Positional(arg) => {
                let metadata = std::fs::metadata(arg);
                match metadata {
                    Ok(m) => {
                        if !m.is_file() {
                            panic!("{arg:?} is not a file");
                        }
                    }
                    Err(e) => {
                        panic!("Error: {e:?}");
                    }
                }
                let (mut buffer, w, h) =
                    load_gray_f32(arg).expect("Cannot read image located at {arg}");
                let status = run(&mut buffer, w, h);
                match status {
                    Ok(_) => {
                        if let Some(p) = output_path {
                            save_gray_f32(p, &buffer, w, h).expect("Cannot save image at {p}");
                        } else {
                            save_gray_f32(arg, &buffer, w, h).expect("Cannot save image at {arg}");
                        }
                        println!("Execution complete");
                    }
                    Err(e) => {
                        panic!("ClError: {e:?}");
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}
