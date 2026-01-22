use core::cmp::{max, min};
use core::mem::MaybeUninit;
use cuda_std::address_space;
use cuda_std::prelude::*;

pub const MAX_K: usize = 31;


#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn conv2d_gray_f32(
    input: &[f32],
    output: *mut f32,
    weights: &[f32],
    width:  usize,
    height: usize,
    k_size: usize,
) {
    let x = (thread::block_idx_x() * thread::block_dim_x() + thread::thread_idx_x()) as usize;
    let y = (thread::block_idx_y() * thread::block_dim_y() + thread::thread_idx_y()) as usize;
    let lx = thread::thread_idx_x() as usize;
    let ly = thread::thread_idx_y() as usize;
    let bx = thread::block_dim_x() as usize;

    if x < width && y < height {


        const MAX_K_2D: usize = MAX_K * MAX_K;

        #[address_space(shared)]
        static mut KLOCAL: [MaybeUninit<f32>; MAX_K_2D] = [MaybeUninit::uninit(); MAX_K_2D];


        let lid = ly * bx + lx;
        if lid < k_size * k_size {
            unsafe {
                KLOCAL[lid].write(weights[lid]);
                
            }
        }

        thread::sync_threads();

        let r = (k_size - 1) / 2;
        let mut acc: f32 = 0.0;

        // Convolution sum
        for ky in 0..k_size {
            let mut iy = y + ky - r;
            iy = max(0usize,min(iy, height-1));
            for kx in 0..k_size {
                let mut ix = x + kx - r;
                ix = max(0usize,min(ix, width-1));
                let p = input[iy * width + ix];
                let w = unsafe { KLOCAL[ky * k_size + kx].assume_init() };
                acc += p*w;
           }
        }
        let o = unsafe { output.add(y * width + x) };
        unsafe { *o = acc };
    }
}
