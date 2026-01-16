use image::{GrayImage, ImageReader, ImageResult, Luma};

/// Loads an image from disk, converts to grayscale, and returns a normalized f32 buffer in [0.0, 1.0].
pub fn load_gray_f32(path: &str) -> image::ImageResult<(Vec<f32>, u32, u32)> {
    // Load + decode (handles PNG/JPG/etc.)
    let img = ImageReader::open(path)?.decode()?;

    // Convert to 8-bit grayscale (Luma8)
    let gray = img.to_luma8();
    let (w, h) = gray.dimensions();

    // Convert to f32 (normalize 0..255 -> 0.0..1.0)
    let buf_u8 = gray.into_raw(); // length = w*h
    let buf_f32: Vec<f32> = buf_u8.into_iter().map(|p| p as f32 / 255.0).collect();

    Ok((buf_f32, w, h))
}

/// Saves a normalized grayscale f32 buffer ([0.0, 1.0]) as an image.
pub fn save_gray_f32(path: &str, buf: &[f32], width: u32, height: u32) -> ImageResult<()> {
    assert_eq!(
        buf.len(),
        (width * height) as usize,
        "Buffer size does not match dimensions"
    );

    // Convert f32 -> u8 with clamping
    let buf_u8: Vec<u8> = buf
        .iter()
        .map(|v| {
            let v = v.clamp(0.0, 1.0);
            (v * 255.0).round() as u8
        })
        .collect();

    // Create grayscale image
    let img: GrayImage =
        GrayImage::from_raw(width, height, buf_u8).expect("Failed to create image");

    // Write to disk (PNG, JPG inferred from extension)
    img.save(path)
}
