use cust::prelude::*;
use getargs::{Arg, Options};
use std::error::Error;
use utils::{load_gray_f32, save_gray_f32};

// PTX generated at build time
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/conv2d_gray_f32.ptx"));

fn run(buffer: &mut [f32], width: u32, height: u32) -> Result<(), Box<dyn Error>> {
    // 3x3 Laplacian-style convolution weights.
    let ksize = 3;
    let weights: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];

    // Initialize CUDA and keep the context alive while GPU work is in flight.
    let _ctx = cust::quick_init()?;

    // Load kernel code from PTX.
    let module = Module::from_ptx(PTX, &[])?;

    // Stream used to enqueue asynchronous GPU operations.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Events used for GPU timing.
    let start = Event::new(EventFlags::DEFAULT)?;
    let stop = Event::new(EventFlags::DEFAULT)?;

    // Allocate input buffer on device and copy host data.
    let input_buf = DeviceBuffer::from_slice(buffer)?;
    // Alternate API to allocate/copy a device buffer.
    let weights_buf = weights.as_slice().as_dbuf()?;

    // Allocate output buffer without initialization.
    // This is safe here because the kernel writes all output elements before any read.
    let output_buf = unsafe { DeviceBuffer::<f32>::uninitialized(buffer.len())? };

    // Lookup the kernel function by symbol name.
    let conv2d_gray_f32 = module.get_function("conv2d_gray_f32")?;

    let block_size = (16u32, 16u32); // 16x16 = 256 threads per block.

    // Round up so the grid covers the full image.
    let grid_size = (
        (width + block_size.0 - 1) / block_size.0,
        (height + block_size.1 - 1) / block_size.1,
    );

    println!(
        "using {:?} blocks and {:?} threads per block",
        grid_size, block_size
    );

    println!("Kernel size: {}", ksize);

    // Queue the start marker on this stream.
    start.record(&stream)?;

    // Launch is unsafe due to the Rust -> CUDA FFI boundary.
    // Rust cannot verify kernel signature, pointer validity/sizes,
    // launch configuration, or device-side memory safety.
    // Immutable slices are passed via pointer/length pairs. This is unsafe
    // because the kernel function is unsafe, but also because, like an FFI
    // call, any mismatch between this call and the called kernel could
    // result in incorrect behaviour or even uncontrolled crashes.

    unsafe {
        launch!(
            // Pass device pointers and their lengths plus scalar dimensions.
            conv2d_gray_f32<<<grid_size, block_size, 0, stream>>>(
                input_buf.as_device_ptr(),
                input_buf.len(),
                output_buf.as_device_ptr(),
                weights_buf.as_device_ptr(),
                weights_buf.len(),
                width as usize,
                height as usize,
                ksize as usize)
        )?;
    }

    // Queue the stop marker after the kernel in the same stream order.
    // `record` itself is not a host-side barrier.
    stop.record(&stream)?;

    // Synchronous device-to-host copy of computed output.
    output_buf.copy_to(buffer)?;

    println!(
        "kernel execution duration (ms): {}",
        Event::elapsed_time_f32(&stop, &start)?
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();

    if args.is_empty() {
        args.push(String::from("--help")); // Show usage when invoked without arguments.
    }
    let mut opts = Options::new(args.iter().map(String::as_str));
    let mut input_path = None;
    let mut output_path = None;

    while let Some(arg) = opts.next_arg().expect("argument parsing error") {
        match arg {
            Arg::Short('h') | Arg::Long("help") => {
                eprintln!(
                    r"Usage: rust-cuda [OPTIONS/ARGS] input ...
                     This command execute a CUDA Convolution kernel on GPU.
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
            panic!("Error: {e:?}");
        }
    }

    Ok(())
}
