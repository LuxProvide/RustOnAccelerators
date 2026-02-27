use getargs::{Arg, Options};
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_NOT_FOUND, CL_DEVICE_TYPE_ACCELERATOR, Device};
use opencl3::error_codes::{CL_INVALID_PLATFORM, ClError};
use opencl3::kernel::Kernel;
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, CL_NON_BLOCKING, cl_event, cl_float, cl_uint};
use opencl3::{Result, platform};
use std::ptr;
use utils::{load_gray_f32, save_gray_f32};

const KERNEL_NAME: &str = "conv2d_gray_f32";
// Path to the built FPGA bitstream image (emulation or hardware).
static FPGA_GEN_IMAGE: &str = concat!(env!("OUT_DIR"), "/conv2d_gray_f32.aocx");

fn run(buffer: &mut [f32], width: u32, height: u32) -> Result<()> {
    let buffer_size = (width * height) as usize;

    // Kernel parameters
    let ksize: cl_uint = 3;
    // 3x3 convolution weights
    let weights: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];

    // Query all OpenCL platforms
    let platforms: Vec<platform::Platform> = platform::get_platforms()?;
    // Select a platform containing "fpga" in its name
    let intel_fpga_platform = platforms
        .into_iter()
        .find(|p| {
            let name = p.name().unwrap_or_default();
            name.to_lowercase().contains("fpga")
        })
        .ok_or_else(|| ClError::from(CL_INVALID_PLATFORM))?;

    // List accelerator devices on that platform
    let device_ids = intel_fpga_platform.get_devices(CL_DEVICE_TYPE_ACCELERATOR)?;

    // Take the first device (error if none)
    let device_id = match device_ids.first() {
        Some(ptr) => *ptr,
        None => return Err(ClError::from(CL_DEVICE_NOT_FOUND)),
    };

    // Create a device handle
    let device = Device::new(device_id);

    println!("Executing on {}", device.name()?);

    // Create a context for the device
    let context = Context::from_device(&device)?;

    // Create a command queue with profiling enabled
    let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)?;

    // Get the path to the FPGA bitstream image
    let aocx_path = std::env::var("FPGA_AOCX_PATH").unwrap_or_else(|_| FPGA_GEN_IMAGE.to_string());

    // Read the bitstream binary
    let aocx = std::fs::read(&aocx_path).unwrap();

    // Create the program from the device-specific binary
    let mut program =
        unsafe { Program::create_from_binary(&context, &[device.id()], &[aocx.as_slice()])? };

    // Build the program for the device
    program.build(&[device.id()], "")?;
    let kernel = Kernel::create(&program, KERNEL_NAME)?;

    // Allocate device buffers
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

    let w: cl_uint = width;
    let h: cl_uint = height;

    // Blocking write: input must be available before kernel launch
    let _input_write_event =
        unsafe { queue.enqueue_write_buffer(&mut input_b, CL_BLOCKING, 0, buffer, &[])? };

    // Non-blocking write: we will wait on its event before launching the kernel
    let _weights_write_event =
        unsafe { queue.enqueue_write_buffer(&mut weights_b, CL_NON_BLOCKING, 0, &weights, &[])? };

    // Set kernel arguments and enqueue the task (waits on weights transfer)
    let kernel_event = unsafe {
        kernel.set_arg(0, &input_b)?;
        kernel.set_arg(1, &output_b)?;
        kernel.set_arg(2, &weights_b)?;
        kernel.set_arg(3, &w)?;
        kernel.set_arg(4, &h)?;
        kernel.set_arg(5, &ksize)?;
        queue.enqueue_task(kernel.get(), &[_weights_write_event.get()])?
    };

    let events: Vec<cl_event> = vec![kernel_event.get()];

    // Enqueue read, waiting for the kernel to complete
    let read_event =
        unsafe { queue.enqueue_read_buffer(&output_b, CL_NON_BLOCKING, 0, buffer, &events)? };

    // Wait for the readback to complete.
    read_event.wait()?;

    // Report kernel execution time from the profiling event
    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let duration = (end_time - start_time) as f64;
    println!("kernel execution duration (ms): {}", duration / 1e6);

    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();

    if args.is_empty() {
        args.push(String::from("--help")); // show usage when no args are provided
    }
    let mut opts = Options::new(args.iter().map(String::as_str));
    let mut input_path = None;
    let mut output_path = None;

    while let Some(arg) = opts.next_arg().expect("argument parsing error") {
        match arg {
            Arg::Short('h') | Arg::Long("help") => {
                eprintln!(
                    r"Usage: rust-opencl-fpga [OPTIONS/ARGS] input ...
                     This command executes an OpenCL convolution kernel on FPGA.
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
                        } else {
                            input_path = Some(arg);
                        }
                    }
                    Err(e) => {
                        panic!("Error: {e:?}");
                    }
                }
            }
            _ => {}
        }
    }

    let (mut buffer, w, h) =
        load_gray_f32(input_path.unwrap()).expect("Cannot read image located at {arg}");
    let status = run(&mut buffer, w, h);
    match status {
        Ok(_) => {
            if let Some(p) = output_path {
                save_gray_f32(p, &buffer, w, h).expect("Cannot save image at {p}");
            } else {
                save_gray_f32(input_path.unwrap(), &buffer, w, h)
                    .expect("Cannot save image at {arg}");
            }
            println!("Execution complete");
        }
        Err(e) => {
            panic!("ClError: {e:?}");
        }
    }
    Ok(())
}
