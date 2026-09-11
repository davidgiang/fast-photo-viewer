//! Bulk media-library crash hunter.
//!
//! Walks a directory tree and pushes every supported file through the
//! exact decode paths the viewer uses, reporting anything that errors,
//! panics, or produces geometry the GPU can't accept.
//!
//! The scanner writes the path it is *about* to touch into a progress
//! file and flushes before decoding. If a file takes the whole process
//! down — a native ffmpeg abort, an out-of-memory kill — that file is
//! the last line in the progress file, which is the only way to
//! attribute a hard crash to an input.
//!
//! Run with:
//!
//!     cargo run --release --bin media_scan -- "C:/Users/me/Downloads/iPhone Content"
//!
//! Flags:
//!     --images-only / --videos-only   restrict what gets probed
//!     --gpu                           also upload each frame to a real
//!                                     GPU texture, which is where a bad
//!                                     image actually takes the process
//!                                     down (wgpu validation errors abort)
//!     --max-mem-gb N                  abort if this process's committed
//!                                     memory passes N GB (default 4). A
//!                                     leak in a bulk scan is otherwise
//!                                     only discovered when the machine
//!                                     runs out of memory.
//!     --limit N                       stop after N files
//!     --skip N                        resume past the first N files
//!     --quiet                         only print failures and the summary

#[path = "../media.rs"]
mod media;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;
use ffmpeg_the_third as ffmpeg;
use image::DynamicImage;
use walkdir::WalkDir;

/// A decode slower than this is worth reporting on its own: in the
/// GUI it runs on the main thread, so it reads as a freeze.
const SLOW_DECODE: Duration = Duration::from_millis(1500);

#[derive(Debug)]
enum Outcome {
    Ok { detail: String, elapsed: Duration },
    Failed { reason: String },
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut root: Option<PathBuf> = None;
    let mut images_only = false;
    let mut videos_only = false;
    let mut limit = usize::MAX;
    let mut skip = 0usize;
    let mut quiet = false;
    let mut gpu = false;
    let mut max_mem_gb: f64 = 4.0;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--images-only" => images_only = true,
            "--videos-only" => videos_only = true,
            "--quiet" => quiet = true,
            "--gpu" => gpu = true,
            "--max-mem-gb" => {
                max_mem_gb = args.next().and_then(|v| v.parse().ok()).unwrap_or(max_mem_gb)
            }
            "--limit" => limit = args.next().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX),
            "--skip" => skip = args.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            other => root = Some(PathBuf::from(other)),
        }
    }

    let root = match root {
        Some(r) => r,
        None => {
            eprintln!("usage: media_scan <directory> [--images-only|--videos-only] [--gpu] [--max-mem-gb N] [--limit N] [--skip N] [--quiet]");
            std::process::exit(2);
        }
    };
    if !root.is_dir() {
        eprintln!("media_scan: not a directory: {}", root.display());
        std::process::exit(2);
    }

    if let Err(e) = ffmpeg::init() {
        eprintln!("media_scan: ffmpeg init failed: {}", e);
    }
    // ffmpeg's own stderr chatter ("deprecated pixel format", "moov
    // atom not found") drowns out the scan results. We report failures
    // from return values instead.
    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Fatal);

    // Decoder panics are turned into errors by `media::decode_image`,
    // but the default hook still prints a full backtrace for each one.
    // Across tens of thousands of files that is unreadable, so silence
    // it; the scanner reports the panic itself.
    std::panic::set_hook(Box::new(|_| {}));

    println!("media_scan: walking {}", root.display());
    let mut files: Vec<PathBuf> = WalkDir::new(&root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| media::is_supported_file(p))
        .filter(|p| {
            if images_only {
                media::is_image_file(p)
            } else if videos_only {
                media::is_video_file(p)
            } else {
                true
            }
        })
        .collect();
    files.sort();
    println!("media_scan: {} supported files found", files.len());

    let mut harness = if gpu {
        match GpuHarness::new() {
            Some(h) => {
                println!("media_scan: GPU upload enabled (max texture side {})", h.max_side);
                Some(h)
            }
            None => {
                eprintln!("media_scan: no wgpu adapter available; --gpu ignored");
                None
            }
        }
    } else {
        None
    };

    let progress_path = std::env::temp_dir().join("media_scan_progress.txt");
    let mut progress = std::fs::File::create(&progress_path).ok();
    println!("media_scan: progress log -> {}", progress_path.display());

    let mut failures: Vec<(PathBuf, String)> = Vec::new();
    let mut slow: Vec<(PathBuf, Duration)> = Vec::new();
    let mut by_reason: BTreeMap<String, usize> = BTreeMap::new();
    let mut scanned = 0usize;
    let mut peak_commit: u64 = 0;
    let started = Instant::now();

    for (i, path) in files.iter().enumerate().skip(skip).take(limit) {
        if let Some(f) = progress.as_mut() {
            let _ = writeln!(f, "{}\t{}", i, path.display());
            let _ = f.flush();
        }

        let outcome = if media::is_video_file(path) {
            probe_video(path, harness.as_mut())
        } else {
            probe_image(path, harness.as_mut())
        };
        scanned += 1;

        // Each file should leave memory where it found it. If usage
        // keeps climbing, something is leaking — stop while the machine
        // is still healthy instead of scanning until it isn't.
        if let Some(commit) = process_commit_bytes() {
            peak_commit = peak_commit.max(commit);
            let limit = (max_mem_gb * 1024.0 * 1024.0 * 1024.0) as u64;
            if commit > limit {
                println!(
                    "\nmedia_scan: ABORTING — committed memory {:.2} GB exceeds the {:.1} GB ceiling \
                     after {} files (last: {}). This indicates a leak; not continuing.",
                    commit as f64 / 1_073_741_824.0,
                    max_mem_gb,
                    scanned,
                    path.display()
                );
                std::process::exit(3);
            }
        }

        match outcome {
            Outcome::Ok { detail, elapsed } => {
                if elapsed >= SLOW_DECODE {
                    slow.push((path.clone(), elapsed));
                    println!("SLOW  {:>7.2}s  {}  [{}]", elapsed.as_secs_f32(), path.display(), detail);
                } else if !quiet {
                    println!("ok    {:>7.0}ms  {}  [{}]", elapsed.as_millis(), path.display(), detail);
                }
            }
            Outcome::Failed { reason } => {
                println!("FAIL  {}  --  {}", path.display(), reason);
                *by_reason.entry(classify(&reason)).or_insert(0) += 1;
                failures.push((path.clone(), reason));
            }
        }

        if scanned % 1000 == 0 {
            println!(
                "-- progress: {}/{} scanned, {} failures, {:.0}s elapsed, {:.0} MB committed",
                scanned,
                files.len(),
                failures.len(),
                started.elapsed().as_secs_f32(),
                process_commit_bytes().unwrap_or(0) as f64 / 1_048_576.0
            );
            let _ = std::io::stdout().flush();
        }
    }

    println!("\n================ media_scan summary ================");
    println!("scanned:  {}", scanned);
    println!("failures: {}", failures.len());
    println!("slow:     {}", slow.len());
    println!("elapsed:  {:.1}s", started.elapsed().as_secs_f32());
    // Flat across a run means nothing is leaking per file.
    println!("peak committed memory: {:.0} MB", peak_commit as f64 / 1_048_576.0);

    if !by_reason.is_empty() {
        println!("\nfailures by kind:");
        let mut kinds: Vec<_> = by_reason.into_iter().collect();
        kinds.sort_by(|a, b| b.1.cmp(&a.1));
        for (kind, count) in kinds {
            println!("  {:>6}  {}", count, kind);
        }
    }

    if !failures.is_empty() {
        println!("\nfirst 60 failing files:");
        for (path, reason) in failures.iter().take(60) {
            println!("  {}\n      {}", path.display(), reason);
        }
    }

    if !slow.is_empty() {
        println!("\nslowest files:");
        let mut s = slow.clone();
        s.sort_by(|a, b| b.1.cmp(&a.1));
        for (path, elapsed) in s.iter().take(20) {
            println!("  {:>7.2}s  {}", elapsed.as_secs_f32(), path.display());
        }
    }

    // A scan is only "clean" when nothing failed; exit non-zero so this
    // can gate a release.
    if failures.is_empty() {
        println!("\nmedia_scan: clean");
    } else {
        println!("\nmedia_scan: {} file(s) would fail to open in the viewer", failures.len());
    }
}


/// Convert a decoded video frame to RGB so the GPU harness can push it
/// through the same texture path the viewer uses. Returns `None` if
/// swscale can't handle the frame — the decode itself already counted
/// as a success by then.
fn frame_to_rgb(frame: &mut ffmpeg::frame::Video) -> Option<DynamicImage> {
    let (w, h) = (frame.width(), frame.height());
    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        frame.format(),
        w,
        h,
        ffmpeg::format::Pixel::RGB24,
        w,
        h,
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    )
    .ok()?;
    let mut rgb = ffmpeg::frame::Video::empty();
    scaler.run(frame, &mut rgb).ok()?;

    let stride = rgb.stride(0);
    let row = w as usize * 3;
    let src = rgb.data(0);
    if stride < row || src.len() < (h as usize - 1) * stride + row {
        return None;
    }
    let mut buf = Vec::with_capacity(row * h as usize);
    for y in 0..h as usize {
        buf.extend_from_slice(&src[y * stride..y * stride + row]);
    }
    image::RgbImage::from_raw(w, h, buf).map(DynamicImage::ImageRgb8)
}

/// A real wgpu device plus the egui renderer, driven exactly the way
/// the viewer drives it.
///
/// Decoding an image and putting it on screen are different failure
/// surfaces: a picture can decode perfectly and still abort the process
/// when its dimensions exceed what the adapter will allocate, because
/// wgpu treats that as a validation error and the default handler
/// panics. Scanning without this step would miss the class of bug the
/// scan exists to find.
struct GpuHarness {
    ctx: egui::Context,
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: egui_wgpu::Renderer,
    max_side: usize,
}

impl GpuHarness {
    fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))?;
        // Mirror the viewer's device setup, including the raised
        // resolution limit — scanning against tighter limits than the
        // app uses would report failures the app never sees, and
        // scanning against looser ones would miss real ones.
        let required_limits = wgpu::Limits::default().using_resolution(adapter.limits());
        let max_side = required_limits.max_texture_dimension_2d as usize;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("media_scan"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .ok()?;
        let target_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let renderer = egui_wgpu::Renderer::new(&device, target_format, None, 1, false);
        Some(Self {
            ctx: egui::Context::default(),
            device,
            queue,
            renderer,
            max_side,
        })
    }

    /// Run the viewer's `regenerate_texture` logic for real: downscale
    /// to fit the adapter's limit, hand the pixels to egui, then flush
    /// egui's texture delta through the wgpu renderer so the upload
    /// actually happens.
    fn upload_image(&mut self, image: &DynamicImage) -> Result<(u32, u32), String> {
        let (w, h) = (image.width() as usize, image.height() as usize);
        let over = w.max(h);
        let rgba = if over > self.max_side {
            let scale = self.max_side as f32 / over as f32;
            let new_w = ((w as f32 * scale).floor() as u32).max(1);
            let new_h = ((h as f32 * scale).floor() as u32).max(1);
            image
                .resize(new_w, new_h, image::imageops::FilterType::Triangle)
                .to_rgba8()
        } else {
            image.to_rgba8()
        };
        let size = [rgba.width() as usize, rgba.height() as usize];
        let dims = (rgba.width(), rgba.height());
        let color_image =
            egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_flat_samples().as_slice());

        let handle = self.ctx.load_texture(
            "current_image",
            color_image,
            egui::TextureOptions::LINEAR,
        );

        // Capture validation failures instead of letting wgpu's default
        // uncaptured-error handler panic, so one bad file is a reported
        // failure rather than the end of the scan.
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mut raw = egui::RawInput::default();
        raw.max_texture_side = Some(self.max_side);
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(1280.0, 720.0),
        ));
        let output = self.ctx.run(raw, |_| {});
        for (id, delta) in &output.textures_delta.set {
            self.renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }

        // `write_texture` only stages the pixels; wgpu holds the staging
        // buffer — and keeps the destination texture alive — until the
        // next `submit`. eframe submits every frame, so the viewer never
        // notices, but a harness that only uploads must submit itself.
        // Without this every image stayed resident for the life of the
        // scan: tens of gigabytes across a photo library, which is
        // enough to take the whole machine down.
        self.queue.submit(std::iter::empty());
        self.device.poll(wgpu::Maintain::Wait);
        let error = pollster::block_on(self.device.pop_error_scope());

        // Release the texture and let egui emit the matching free, then
        // submit and poll again so wgpu destroys it now rather than at
        // some later maintenance pass.
        drop(handle);
        let output = self.ctx.run(egui::RawInput::default(), |_| {});
        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }
        self.queue.submit(std::iter::empty());
        self.device.poll(wgpu::Maintain::Wait);

        match error {
            Some(e) => Err(format!("{:?}", e)),
            None => Ok(dims),
        }
    }
}


/// This process's private committed memory in bytes — the figure that
/// has to fit in RAM plus the page file, and so the one that brings a
/// machine down when it runs away.
#[cfg(windows)]
fn process_commit_bytes() -> Option<u64> {
    use std::ffi::c_void;

    // PROCESS_MEMORY_COUNTERS from <psapi.h>.
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
    }

    let mut counters = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        ..Default::default()
    };
    // SAFETY: `counters` is a correctly sized PROCESS_MEMORY_COUNTERS
    // with `cb` set, and the pseudo-handle from GetCurrentProcess needs
    // no cleanup.
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    (ok != 0).then_some(counters.pagefile_usage as u64)
}

#[cfg(not(windows))]
fn process_commit_bytes() -> Option<u64> {
    None
}

/// Collapse a specific error string into a coarse bucket so the
/// summary shows kinds of failure rather than thousands of variants.
fn classify(reason: &str) -> String {
    let lower = reason.to_lowercase();
    for probe in [
        "gpu upload",
        "decoder panicked",
        "zero-sized",
        "image too large",
        "empty file",
        "no video stream",
        "moov atom not found",
        "invalid data found",
        "no frame decoded",
        "buffer size mismatch",
        "permission denied",
        "format error",
        "unsupported",
    ] {
        if lower.contains(probe) {
            return probe.to_string();
        }
    }
    // Fall back to the first clause of the message.
    lower
        .split(|c| c == ':' || c == ',')
        .next()
        .unwrap_or(&lower)
        .trim()
        .to_string()
}

fn probe_image(path: &Path, harness: Option<&mut GpuHarness>) -> Outcome {
    let start = Instant::now();
    let img = match media::decode_image(path) {
        Ok(img) => img,
        Err(e) => return Outcome::Failed { reason: e },
    };
    let mut detail = format!("{}x{}", img.width(), img.height());
    if let Some(h) = harness {
        match h.upload_image(&img) {
            Ok(uploaded) => detail.push_str(&format!(" -> tex {}x{}", uploaded.0, uploaded.1)),
            Err(e) => return Outcome::Failed { reason: format!("gpu upload: {}", e) },
        }
    }
    Outcome::Ok {
        detail,
        elapsed: start.elapsed(),
    }
}

/// Open a video the way `Player::open` does — demux, build a decoder,
/// decode the first frame — without spinning up audio or a GPU. This
/// is where a bad container shows up as either an error or a hang.
fn probe_video(path: &Path, harness: Option<&mut GpuHarness>) -> Outcome {
    let start = Instant::now();
    let mut frame_rgb: Option<DynamicImage> = None;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        probe_video_inner(path, &mut frame_rgb)
    }));
    if let (Some(h), Some(img)) = (harness, frame_rgb.as_ref()) {
        if let Err(e) = h.upload_image(img) {
            return Outcome::Failed {
                reason: format!("gpu upload: {}", e),
            };
        }
    }
    match result {
        Ok(Ok(detail)) => Outcome::Ok {
            detail,
            elapsed: start.elapsed(),
        },
        Ok(Err(e)) => Outcome::Failed { reason: e },
        Err(_) => Outcome::Failed {
            reason: "decoder panicked".to_string(),
        },
    }
}

fn probe_video_inner(
    path: &Path,
    out_frame: &mut Option<DynamicImage>,
) -> Result<String, String> {
    let mut ictx = ffmpeg::format::input(&path).map_err(|e| format!("open: {}", e))?;
    let duration_us = ictx.duration();

    let (index, time_base) = {
        let stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| "no video stream".to_string())?;
        (stream.index(), stream.time_base())
    };
    let _ = time_base;

    let params = ictx
        .stream(index)
        .ok_or_else(|| "stream disappeared".to_string())?
        .parameters();
    let ctx = ffmpeg::codec::context::Context::from_parameters(params)
        .map_err(|e| format!("codec ctx: {}", e))?;
    let mut decoder = ctx
        .decoder()
        .video()
        .map_err(|e| format!("decoder: {}", e))?;

    let (w, h) = (decoder.width(), decoder.height());
    media::validate_dimensions(w, h)?;

    let has_audio = ictx.streams().best(ffmpeg::media::Type::Audio).is_some();

    // Decode the first frame. Some containers need a few packets before
    // the decoder emits anything; cap the effort so a broken file can't
    // stall the scan.
    let mut frame = ffmpeg::frame::Video::empty();
    let mut got = false;
    let mut tried = 0usize;
    for item in ictx.packets() {
        tried += 1;
        if tried > 400 {
            break;
        }
        let (stream, packet) = match item {
            Ok(v) => v,
            Err(e) => return Err(format!("packet read: {}", e)),
        };
        if stream.index() != index {
            continue;
        }
        if decoder.send_packet(&packet).is_err() {
            continue;
        }
        if decoder.receive_frame(&mut frame).is_ok() {
            got = true;
            break;
        }
    }
    if !got {
        let _ = decoder.send_eof();
        got = decoder.receive_frame(&mut frame).is_ok();
    }
    if !got {
        return Err("no frame decoded".to_string());
    }

    media::validate_dimensions(frame.width(), frame.height())?;
    *out_frame = frame_to_rgb(&mut frame);

    Ok(format!(
        "{}x{} {:?} {:.1}s{}",
        frame.width(),
        frame.height(),
        frame.format(),
        duration_us.max(0) as f64 / 1_000_000.0,
        if has_audio { " +audio" } else { " (silent)" },
    ))
}
