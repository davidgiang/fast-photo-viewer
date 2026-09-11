//! File-type classification and still-image decoding.
//!
//! Extracted from `main.rs` so the `media_scan` binary can hammer the
//! exact same decode paths the GUI uses — a bug found by the scanner
//! is a bug the viewer would have hit.
//!
//! Every public decode entry point here is *total*: it returns `Err`
//! rather than panicking or aborting, and it refuses to hand back an
//! image whose dimensions the GPU can't hold. Malformed media is the
//! norm in real photo libraries (iPhone backups are full of truncated
//! JPEGs and zero-byte placeholders), so decode failures must stay
//! recoverable errors instead of taking the process down.

use std::fs;
use std::path::Path;

use image::DynamicImage;
use ffmpeg_the_third as ffmpeg;

pub const IMAGE_EXTENSIONS: &[&str] = &[
    "jpg", "jpeg", "png", "bmp", "webp", "gif", "tiff", "tif", "ico", "svg",
    // HEIF family (decoded via ffmpeg)
    "heic", "heif", "avif",
    // Camera raw (decoded via rawloader + imagepipe)
    "nef", "nrw", "cr2", "arw", "srf", "sr2", "dng", "raf",
    "rw2", "orf", "pef", "srw", "3fr", "mrw", "iiq", "kdc",
    "dcr", "rwl", "x3f", "mef", "mos",
];

pub const RAW_EXTENSIONS: &[&str] = &[
    "nef", "nrw", "cr2", "arw", "srf", "sr2", "dng", "raf",
    "rw2", "orf", "pef", "srw", "3fr", "mrw", "iiq", "kdc",
    "dcr", "rwl", "x3f", "mef", "mos",
];

pub const HEIF_EXTENSIONS: &[&str] = &["heic", "heif", "avif"];

pub const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "wmv", "flv", "webm", "m4v",
    "mpg", "mpeg", "3gp", "3g2", "ogv", "ts", "vob", "mts", "m2ts",
];

/// Hard ceiling on decoded pixel count. `image` will happily allocate
/// a buffer for whatever dimensions a header claims, so a corrupt or
/// hostile header (or a genuine gigapixel panorama) can ask for tens
/// of gigabytes and abort the process on allocation failure — an
/// unrecoverable crash, not a catchable error. 512 MP covers every
/// real camera by a wide margin.
const MAX_DECODED_PIXELS: u64 = 512 * 1024 * 1024;

/// Upper bound on file size we're willing to slurp into memory for a
/// still image. Anything larger is either not really an image or
/// would blow up during decode anyway.
const MAX_IMAGE_FILE_BYTES: u64 = 1024 * 1024 * 1024;

fn has_ext(path: &Path, exts: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| exts.contains(&ext.to_lowercase().as_str()))
        .unwrap_or(false)
}

pub fn is_raw_file(path: &Path) -> bool { has_ext(path, RAW_EXTENSIONS) }
pub fn is_heif_file(path: &Path) -> bool { has_ext(path, HEIF_EXTENSIONS) }
pub fn is_image_file(path: &Path) -> bool { has_ext(path, IMAGE_EXTENSIONS) }
pub fn is_video_file(path: &Path) -> bool { has_ext(path, VIDEO_EXTENSIONS) }

pub fn is_supported_file(path: &Path) -> bool {
    is_image_file(path) || is_video_file(path)
}

/// Reject dimensions that would produce an unusable or dangerous
/// allocation. Zero-sized images are the sharp edge here: they decode
/// "successfully" but blow up later as a wgpu validation error when
/// the texture is created, which aborts the process.
pub fn validate_dimensions(w: u32, h: u32) -> Result<(), String> {
    if w == 0 || h == 0 {
        return Err(format!("zero-sized image ({}x{})", w, h));
    }
    let pixels = w as u64 * h as u64;
    if pixels > MAX_DECODED_PIXELS {
        return Err(format!(
            "image too large ({}x{} = {} MP, cap {} MP)",
            w,
            h,
            pixels / 1_000_000,
            MAX_DECODED_PIXELS / 1_000_000
        ));
    }
    Ok(())
}

/// Scan a byte slice for the largest embedded JPEG (SOI..EOI). Works
/// because within a valid JPEG payload, any `0xFF` byte is followed by
/// `0x00` (stuff byte), so `FF D9` only appears as a real end-of-image.
pub fn find_largest_embedded_jpeg(data: &[u8]) -> Option<&[u8]> {
    let mut best: Option<&[u8]> = None;
    let mut i = 0;
    while i + 2 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xD8 && data[i + 2] == 0xFF {
            let mut j = i + 2;
            while j + 1 < data.len() {
                if data[j] == 0xFF && data[j + 1] == 0xD9 {
                    let slice = &data[i..j + 2];
                    if best.map_or(true, |b| slice.len() > b.len()) {
                        best = Some(slice);
                    }
                    i = j + 2;
                    break;
                }
                j += 1;
            }
            if j + 1 >= data.len() {
                break;
            }
        } else {
            i += 1;
        }
    }
    best
}

/// Decode a camera RAW file (NEF, CR2, ARW, DNG, etc.) to an sRGB image.
/// Fast path: extract the full-resolution JPEG preview every modern
/// camera embeds. Fallback: full rawloader + imagepipe demosaic (slow,
/// and only works for camera models in rawloader's database).
fn decode_raw_image(path: &Path) -> Result<DynamicImage, String> {
    if let Ok(bytes) = fs::read(path) {
        if let Some(jpeg) = find_largest_embedded_jpeg(&bytes) {
            // Require at least 64 KB so we don't pick up a tiny thumbnail
            // when a larger preview exists further in the file.
            if jpeg.len() >= 64 * 1024 {
                if let Ok(img) = image::load_from_memory(jpeg) {
                    return Ok(img);
                }
            }
        }
    }

    // Fallback: full raw decode via rawloader + imagepipe.
    let mut pipeline = imagepipe::Pipeline::new_from_file(path)
        .map_err(|e| format!("raw open: {:?}", e))?;
    let decoded = pipeline
        .output_8bit(None)
        .map_err(|e| format!("raw pipeline: {:?}", e))?;
    let buf = image::RgbImage::from_raw(
        decoded.width as u32,
        decoded.height as u32,
        decoded.data,
    )
    .ok_or_else(|| "raw: buffer size mismatch".to_string())?;
    Ok(DynamicImage::ImageRgb8(buf))
}

/// Decode a still image through ffmpeg, treating the file as a
/// one-frame video: decode the first frame and convert it to RGB24 via
/// swscale.
///
/// This is the only decoder for the HEIF family, and the fallback for
/// everything else. ffmpeg's decoders recover from mid-stream damage
/// that the `image` crate refuses outright — a JPEG with a corrupt
/// scan still yields a mostly-correct picture here, which is what
/// every other viewer on the machine shows for the same file.
fn decode_with_ffmpeg(path: &Path) -> Result<DynamicImage, String> {
    let mut ictx = ffmpeg::format::input(&path)
        .map_err(|e| format!("ffmpeg open: {}", e))?;

    let stream_index = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| "ffmpeg: no image stream".to_string())?
        .index();

    let params = ictx
        .stream(stream_index)
        .ok_or_else(|| "ffmpeg: missing stream".to_string())?
        .parameters();
    let decoder_ctx = ffmpeg::codec::context::Context::from_parameters(params)
        .map_err(|e| format!("ffmpeg codec ctx: {}", e))?;
    let mut decoder = decoder_ctx
        .decoder()
        .video()
        .map_err(|e| format!("ffmpeg decoder: {}", e))?;

    // swscale rejects zero dimensions with an assertion inside the
    // native library rather than an error return, so screen them here.
    validate_dimensions(decoder.width(), decoder.height())
        .map_err(|e| format!("ffmpeg: {}", e))?;

    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        decoder.format(),
        decoder.width(),
        decoder.height(),
        ffmpeg::format::Pixel::RGB24,
        decoder.width(),
        decoder.height(),
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    )
    .map_err(|e| format!("ffmpeg scaler: {}", e))?;

    let extract = |scaler: &mut ffmpeg::software::scaling::context::Context,
                   frame: &ffmpeg::frame::Video|
     -> Result<DynamicImage, String> {
        let mut rgb = ffmpeg::frame::Video::empty();
        scaler
            .run(frame, &mut rgb)
            .map_err(|e| format!("ffmpeg scale: {}", e))?;
        let w = rgb.width();
        let h = rgb.height();
        validate_dimensions(w, h).map_err(|e| format!("ffmpeg: {}", e))?;
        let stride = rgb.stride(0);
        let src = rgb.data(0);
        let row_bytes = w as usize * 3;
        // A plane shorter than stride × height means ffmpeg handed us a
        // buffer that doesn't match the geometry it reported; slicing
        // on those numbers would panic.
        if stride < row_bytes || src.len() < stride * (h as usize - 1) + row_bytes {
            return Err("ffmpeg: plane smaller than reported geometry".to_string());
        }
        let mut buf = Vec::with_capacity(row_bytes * h as usize);
        for y in 0..h as usize {
            let start = y * stride;
            buf.extend_from_slice(&src[start..start + row_bytes]);
        }
        let img = image::RgbImage::from_raw(w, h, buf)
            .ok_or_else(|| "ffmpeg: buffer size mismatch".to_string())?;
        Ok(DynamicImage::ImageRgb8(img))
    };

    let mut frame = ffmpeg::frame::Video::empty();
    for item in ictx.packets() {
        let (stream, packet) = item.map_err(|e| format!("ffmpeg packet: {}", e))?;
        if stream.index() != stream_index {
            continue;
        }
        decoder
            .send_packet(&packet)
            .map_err(|e| format!("ffmpeg send: {}", e))?;
        if decoder.receive_frame(&mut frame).is_ok() {
            return extract(&mut scaler, &frame);
        }
    }

    decoder
        .send_eof()
        .map_err(|e| format!("ffmpeg eof: {}", e))?;
    if decoder.receive_frame(&mut frame).is_ok() {
        return extract(&mut scaler, &frame);
    }

    Err("ffmpeg: no frame decoded".to_string())
}

/// Decode an ordinary raster image, retrying once with an appended
/// end-of-image marker when the file is a JPEG that was cut off
/// mid-write (endemic to interrupted phone transfers).
fn decode_standard_image(path: &Path) -> Result<DynamicImage, String> {
    let meta = fs::metadata(path).map_err(|e| e.to_string())?;
    if meta.len() == 0 {
        return Err("empty file".to_string());
    }
    if meta.len() > MAX_IMAGE_FILE_BYTES {
        return Err(format!("file too large ({} bytes)", meta.len()));
    }

    let mut bytes = fs::read(path).map_err(|e| e.to_string())?;

    // Check the header-declared dimensions before decoding: `image`
    // allocates the full pixel buffer up front, so a bogus header can
    // request an allocation large enough to abort the process.
    if let Ok(reader) = image::ImageReader::new(std::io::Cursor::new(&bytes)).with_guessed_format() {
        if let Ok((w, h)) = reader.into_dimensions() {
            validate_dimensions(w, h)?;
        }
    }

    let first_error = match image::load_from_memory(&bytes) {
        Ok(img) => return Ok(img),
        Err(e) => e.to_string(),
    };

    // A JPEG cut off mid-write — endemic to interrupted phone
    // transfers — decodes fine once it has an end-of-image marker.
    if bytes.len() > 2 && bytes[0] == 0xFF && bytes[1] == 0xD8 {
        let mut repaired = bytes;
        repaired.push(0xFF);
        repaired.push(0xD9);
        repaired.extend(std::iter::repeat(0).take(1024));
        if let Ok(img) = image::load_from_memory(&repaired) {
            return Ok(img);
        }
        bytes = repaired;
    }
    drop(bytes);

    // Last resort: ffmpeg. It tolerates damage the `image` crate
    // rejects outright, which is the difference between a photo the
    // user can still look at and an error message.
    decode_with_ffmpeg(path)
        .map_err(|ffmpeg_err| format!("{}; ffmpeg fallback: {}", first_error, ffmpeg_err))
}

/// Decode any supported still image, dispatching on extension.
///
/// Decoder panics are converted into errors: `rawloader`/`imagepipe`
/// index out of bounds on malformed raw files, and the `image` crate
/// can panic inside individual format decoders. A photo library is
/// exactly where those inputs live, so an unwind here has to become a
/// skipped file rather than a dead process.
pub fn decode_image(path: &Path) -> Result<DynamicImage, String> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if is_raw_file(path) {
            decode_raw_image(path)
        } else if is_heif_file(path) {
            decode_with_ffmpeg(path)
        } else {
            decode_standard_image(path)
        }
    }));

    let image = match result {
        Ok(r) => r?,
        Err(_) => return Err("decoder panicked".to_string()),
    };

    validate_dimensions(image.width(), image.height())?;
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn zero_dimensions_are_rejected() {
        assert!(validate_dimensions(0, 100).is_err());
        assert!(validate_dimensions(100, 0).is_err());
        assert!(validate_dimensions(0, 0).is_err());
    }

    #[test]
    fn ordinary_camera_dimensions_are_accepted() {
        // Nikon Z9, Fujifilm GFX100, an 8K frame, a long panorama.
        assert!(validate_dimensions(8256, 5504).is_ok());
        assert!(validate_dimensions(11648, 8736).is_ok());
        assert!(validate_dimensions(7680, 4320).is_ok());
        assert!(validate_dimensions(60000, 2000).is_ok());
    }

    #[test]
    fn absurd_dimensions_are_rejected_before_allocation() {
        // A corrupt header claiming 65535×65535 would ask `image` for
        // ~17 GB; that must not reach the allocator.
        assert!(validate_dimensions(65535, 65535).is_err());
    }

    #[test]
    fn extension_matching_is_case_insensitive() {
        assert!(is_image_file(&PathBuf::from("a/B/IMG_0001.JPG")));
        assert!(is_image_file(&PathBuf::from("photo.HeIc")));
        assert!(is_video_file(&PathBuf::from("clip.MOV")));
        assert!(is_raw_file(&PathBuf::from("shot.NEF")));
        assert!(!is_supported_file(&PathBuf::from("cache.ithmb")));
        assert!(!is_supported_file(&PathBuf::from("noextension")));
    }

    #[test]
    fn heif_extensions_route_to_the_ffmpeg_decoder() {
        assert!(is_heif_file(&PathBuf::from("x.heic")));
        assert!(is_heif_file(&PathBuf::from("x.avif")));
        assert!(!is_heif_file(&PathBuf::from("x.jpg")));
    }

    #[test]
    fn empty_and_missing_files_error_instead_of_panicking() {
        let dir = std::env::temp_dir().join("fpv_media_tests");
        let _ = fs::create_dir_all(&dir);

        let empty = dir.join("empty.jpg");
        fs::write(&empty, b"").unwrap();
        assert!(decode_image(&empty).is_err(), "empty file must error");

        let garbage = dir.join("garbage.png");
        fs::write(&garbage, b"this is definitely not a PNG").unwrap();
        assert!(decode_image(&garbage).is_err(), "garbage must error");

        let missing = dir.join("does-not-exist.jpg");
        assert!(decode_image(&missing).is_err(), "missing file must error");

        let _ = fs::remove_file(&empty);
        let _ = fs::remove_file(&garbage);
    }

    #[test]
    fn truncated_jpeg_is_repaired_rather_than_rejected() {
        // Encode a real JPEG, lop off the trailing EOI marker, and
        // confirm the repair path still produces an image.
        let dir = std::env::temp_dir().join("fpv_media_tests");
        let _ = fs::create_dir_all(&dir);
        let img = DynamicImage::ImageRgb8(image::RgbImage::new(64, 48));
        let mut encoded: Vec<u8> = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut encoded),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
        encoded.truncate(encoded.len() - 64);

        let path = dir.join("truncated.jpg");
        fs::write(&path, &encoded).unwrap();
        let decoded = decode_image(&path);
        assert!(decoded.is_ok(), "truncated JPEG should be repaired: {:?}", decoded.err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn embedded_jpeg_scanner_finds_the_largest_payload() {
        let mut data = vec![0u8; 16];
        // Small JPEG.
        data.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0x01, 0x02, 0xFF, 0xD9]);
        data.extend_from_slice(&[0u8; 8]);
        // Larger JPEG.
        data.extend_from_slice(&[0xFF, 0xD8, 0xFF]);
        data.extend_from_slice(&[0x11; 40]);
        data.extend_from_slice(&[0xFF, 0xD9]);

        let found = find_largest_embedded_jpeg(&data).expect("should find a payload");
        assert_eq!(found.len(), 45);
        assert_eq!(&found[..3], &[0xFF, 0xD8, 0xFF]);
    }

    #[test]
    fn embedded_jpeg_scanner_handles_input_with_no_jpeg() {
        assert!(find_largest_embedded_jpeg(&[0u8; 128]).is_none());
        assert!(find_largest_embedded_jpeg(&[]).is_none());
        // Start marker with no terminator must not run off the end.
        assert!(find_largest_embedded_jpeg(&[0xFF, 0xD8, 0xFF, 0x00, 0x00]).is_none());
    }
}
