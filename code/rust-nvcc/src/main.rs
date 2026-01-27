use cust::prelude::*;
use getargs::{Arg, Options};
use std::error::Error;
use utils::{load_gray_f32, save_gray_f32};

// Generated PTX file will be located in OUT_DIR
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/conv2d_gray_f32.ptx"));

fn run(buffer: &mut [f32], width: u32, height: u32) -> Result<(), Box<dyn Error>> {
    let buffer_size = (width * height) as usize;

    // Define kernel
    let ksize = 4;
    let weights: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];

    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    let _ctx = cust::quick_init()?;

    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    let module = Module::from_ptx(PTX, &[])?;

    // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // Create timing events
    let start = Event::new(EventFlags::DEFAULT)?;
    let stop = Event::new(EventFlags::DEFAULT)?;

    // allocate the GPU memory needed to house our numbers and copy them over.
    let input_buf = DeviceBuffer::from_slice(buffer)?;
    let weights_buf = weights.as_slice().as_dbuf()?;

    let output_buf = unsafe { DeviceBuffer::uninitialized(buffer.len() * size_of::<f32>())? };

    // retrieve the `vecadd` kernel from the module so we can calculate the right launch config.
    let conv2d_gray_f32 = module.get_function("conv2d_gray_f32")?;

    let block_size = (16u32, 16u32, 1u32); // 256 threads

    let grid_size = (
        (width + block_size.0 - 1) / block_size.0,
        (height + block_size.1 - 1) / block_size.1,
        1u32,
    );

    println!(
        "using {:?} blocks and {:?} threads per block",
        grid_size, block_size
    );

    start.record(&stream)?;

    // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
    // not block the thread until the kernel is finished.
    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            conv2d_gray_f32<<<grid_size, block_size, 0, stream>>>(
                input_buf.as_device_ptr(),
                output_buf.as_device_ptr(),
                weights_buf.as_device_ptr(),
                width,
                height,
                ksize)
        )?;
    }

    stop.record(&stream)?;

    // copy back the data from the GPU.
    //output_buf.copy_to(&mut out)?;
    output_buf.copy_to(buffer)?;

    println!(
        "kernel execution duration (ms): {}",
        Event::elapsed_time_f32(&start, &stop)?
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();

    if args.is_empty() {
        args.push(String::from("--help")); // help the user out :)
    }
    let mut opts = Options::new(args.iter().map(String::as_str));
    let mut input_path = None;
    let mut output_path = None;

    while let Some(arg) = opts.next_arg().expect("argument parsing error") {
        match arg {
            Arg::Short('h') | Arg::Long("help") => {
                eprintln!(
                    r"Usage: rust-nvcc [OPTIONS/ARGS] input ...
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
