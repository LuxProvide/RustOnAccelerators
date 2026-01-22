use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn conv2d_gray_f32(input: &[f32], b: &[f32], c: *mut f32) {



    let x = thread::thread_idx_x() as usize;
    let y = thread::thread_idx_y() as usize;

    let lx = thread::block_idx_x() as usize * TILE_SIZE + ty;
    let ly = thread::block_idx_y() as usize * TILE_SIZE + tx;

    if idx < a.len() {
        let elem = unsafe { &mut *c.add(idx) };
        *elem = a[idx] + b[idx];
    }
}
