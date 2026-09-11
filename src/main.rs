#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod media;
mod recycle;
mod video_player;

use eframe::egui;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::process::Command;
use std::collections::HashSet;
use walkdir::WalkDir;
use rand::seq::SliceRandom;
use image::DynamicImage;
use crate::media::{
    decode_image, is_image_file, is_supported_file, is_video_file, IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
};
use crate::video_player::{Player, PlayerState};
use ffmpeg_the_third as ffmpeg;

#[derive(Clone, Copy, PartialEq)]
enum MediaFilter {
    All,
    ImagesOnly,
    VideosOnly,
}

impl MediaFilter {
    fn label(self) -> &'static str {
        match self {
            MediaFilter::All => "All Media",
            MediaFilter::ImagesOnly => "Images Only",
            MediaFilter::VideosOnly => "Videos Only",
        }
    }

    fn cycle(self) -> Self {
        match self {
            MediaFilter::All => MediaFilter::ImagesOnly,
            MediaFilter::ImagesOnly => MediaFilter::VideosOnly,
            MediaFilter::VideosOnly => MediaFilter::All,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ViewOrder {
    Random,
    Ordered,
}

impl ViewOrder {
    fn label(self) -> &'static str {
        match self {
            ViewOrder::Random => "Random",
            ViewOrder::Ordered => "Ordered",
        }
    }

    fn toggle(self) -> Self {
        match self {
            ViewOrder::Random => ViewOrder::Ordered,
            ViewOrder::Ordered => ViewOrder::Random,
        }
    }
}

/// Which end of the A-B clip loop a seek-bar drag is moving.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ClipMarker {
    Start,
    End,
}

/// A short-lived status message shown over the media, used for actions
/// whose result isn't otherwise visible on screen — chiefly deletes,
/// where the user needs to know whether the file actually made it to
/// the Recycle Bin.
struct Toast {
    text: String,
    is_error: bool,
    shown_at: std::time::Instant,
}

impl Toast {
    const LIFETIME: std::time::Duration = std::time::Duration::from_millis(2600);

    fn new(text: impl Into<String>, is_error: bool) -> Self {
        Self {
            text: text.into(),
            is_error,
            shown_at: std::time::Instant::now(),
        }
    }

    fn expired(&self) -> bool {
        self.shown_at.elapsed() > Self::LIFETIME
    }
}

fn matches_filter(path: &Path, filter: MediaFilter) -> bool {
    match filter {
        MediaFilter::All => true,
        MediaFilter::ImagesOnly => is_image_file(path),
        MediaFilter::VideosOnly => is_video_file(path),
    }
}

// `file_has_audio_stream` was used by the old egui-video path; the in-
// house player now owns audio probing internally.

/// Extract `count` evenly-spaced thumbnail frames from a video for use
/// as a seek preview strip. Runs on a background thread; writes each
/// thumbnail into `out[i]` as an egui::ColorImage and requests a repaint
/// so the UI can lazy-upload new textures.
fn extract_seek_thumbnails(
    path: PathBuf,
    count: usize,
    out: Arc<Mutex<Vec<Option<egui::ColorImage>>>>,
    ctx: egui::Context,
) {
    let mut ictx = match ffmpeg::format::input(&path) {
        Ok(c) => c,
        Err(_) => return,
    };
    let (stream_index, src_w, src_h, src_format, params) = {
        let Some(stream) = ictx.streams().best(ffmpeg::media::Type::Video) else {
            return;
        };
        let params = stream.parameters();
        let decoder_ctx = match ffmpeg::codec::context::Context::from_parameters(params.clone()) {
            Ok(c) => c,
            Err(_) => return,
        };
        let decoder = match decoder_ctx.decoder().video() {
            Ok(d) => d,
            Err(_) => return,
        };
        (stream.index(), decoder.width(), decoder.height(), decoder.format(), params)
    };

    let decoder_ctx = match ffmpeg::codec::context::Context::from_parameters(params) {
        Ok(c) => c,
        Err(_) => return,
    };
    let mut decoder = match decoder_ctx.decoder().video() {
        Ok(d) => d,
        Err(_) => return,
    };

    // swscale asserts internally on zero-sized input rather than
    // returning an error, so bail before constructing the context.
    if crate::media::validate_dimensions(src_w, src_h).is_err() {
        return;
    }

    let thumb_w: u32 = 192;
    let thumb_h: u32 = ((thumb_w as u64 * src_h as u64) / src_w as u64).max(1) as u32;

    let mut scaler = match ffmpeg::software::scaling::context::Context::get(
        src_format,
        src_w,
        src_h,
        ffmpeg::format::Pixel::RGBA,
        thumb_w,
        thumb_h,
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    ) {
        Ok(s) => s,
        Err(_) => return,
    };

    let duration = ictx.duration(); // AV_TIME_BASE units (microseconds)
    if duration <= 0 {
        return;
    }

    let mut frame = ffmpeg::frame::Video::empty();
    let mut rgb = ffmpeg::frame::Video::empty();

    for i in 0..count {
        // Aim slightly past the start of each segment so we don't land
        // exactly on a non-keyframe boundary.
        let target = (duration as i128 * i as i128 / count as i128) as i64
            + duration / (count as i64 * 4);
        let _ = ictx.seek(target, ..target + duration / count as i64);
        let _ = decoder.flush();

        let mut got_frame = false;
        let mut tries = 0usize;
        for item in ictx.packets() {
            tries += 1;
            if tries > 200 {
                break;
            }
            let (s, packet) = match item {
                Ok(v) => v,
                Err(_) => break,
            };
            if s.index() != stream_index {
                continue;
            }
            if decoder.send_packet(&packet).is_err() {
                continue;
            }
            if decoder.receive_frame(&mut frame).is_ok() {
                got_frame = true;
                break;
            }
        }
        if !got_frame {
            continue;
        }

        if scaler.run(&frame, &mut rgb).is_err() {
            continue;
        }

        let w = rgb.width() as usize;
        let h = rgb.height() as usize;
        let stride = rgb.stride(0);
        let src = rgb.data(0);
        let row_bytes = w * 4;
        // Guard the row slicing: a frame whose plane is shorter than
        // its reported geometry would panic the thumbnail thread, and
        // an unwind here kills the whole process.
        if w == 0 || h == 0 || stride < row_bytes || src.len() < (h - 1) * stride + row_bytes {
            continue;
        }
        let mut buf = Vec::with_capacity(row_bytes * h);
        for y in 0..h {
            let start = y * stride;
            buf.extend_from_slice(&src[start..start + row_bytes]);
        }
        let color_image = egui::ColorImage::from_rgba_unmultiplied([w, h], &buf);

        {
            let mut o = out.lock().unwrap();
            if i < o.len() {
                o[i] = Some(color_image);
            }
        }
        ctx.request_repaint();
    }
}

fn main() -> eframe::Result<()> {
    let initial_file: Option<PathBuf> = std::env::args().nth(1).map(PathBuf::from);

    let icon_bytes = include_bytes!("../assets/icon.ico");
    let icon_image = image::load_from_memory(icon_bytes).expect("Failed to load app icon");
    let icon_rgba = icon_image.into_rgba8();
    let icon = egui::IconData {
        width: icon_rgba.width(),
        height: icon_rgba.height(),
        rgba: icon_rgba.into_raw(),
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("Fast Photo Viewer")
            .with_icon(std::sync::Arc::new(icon)),
        // Request the wgpu renderer so we can obtain a wgpu::Device in
        // CreationContext and manage our own video textures / custom
        // render callbacks downstream.
        renderer: eframe::Renderer::Wgpu,
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            // Override eframe's default device descriptor so we can
            // request optional features. `TEXTURE_FORMAT_16BIT_NORM`
            // is needed for the 10-bit HEVC playback path to upload
            // directly to an `Rgba16Unorm` texture.
            device_descriptor: std::sync::Arc::new(|adapter| {
                let wanted = eframe::wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
                let required_features = if adapter.features().contains(wanted) {
                    wanted
                } else {
                    eframe::wgpu::Features::empty()
                };
                // `Limits::default()` caps `max_texture_dimension_2d`
                // at 8192. Modern 45+MP cameras exceed that (Nikon
                // Z8/Z9 8256×5504, Sony A7R V 9504×6336, Fujifilm
                // GFX100 11648×8736), so uploading their full-res
                // frames blew up wgpu validation and crashed the app.
                // Raise the resolution limits to whatever the adapter
                // actually supports (typically 16384 on desktop).
                let required_limits = eframe::wgpu::Limits::default()
                    .using_resolution(adapter.limits());
                eframe::wgpu::DeviceDescriptor {
                    label: Some("fast-photo-viewer"),
                    required_features,
                    required_limits,
                    memory_hints: eframe::wgpu::MemoryHints::default(),
                }
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "Fast Photo Viewer",
        options,
        Box::new(move |cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Ok(Box::new(PhotoViewer::new(cc, initial_file)) as Box<dyn eframe::App>)
        }),
    )
}

struct PhotoViewer {
    media_paths: Arc<Mutex<Vec<PathBuf>>>,
    current_media_path: Option<PathBuf>,
    current_image: Option<DynamicImage>,

    // Video
    video_player: Option<Player>,
    is_video: bool,
    video_looping: bool,
    video_volume: f32,
    /// Remembers the volume level before the user muted so clicking
    /// the speaker icon restores it instead of snapping back to 0.
    video_pre_mute_volume: f32,
    video_has_audio: bool,
    seek_frac: f32,
    video_rotation: u16,

    // Scrubbing: while the user is dragging on the seek bar, we hold
    // the target position here and only commit it to the player on
    // release so the decoder isn't hammered every frame.
    scrubbing: Option<f32>,

    // A-B clip loop, in milliseconds from the start of the current
    // video. Both are per-video and cleared whenever new media loads.
    clip_start_ms: Option<i64>,
    clip_end_ms: Option<i64>,
    /// Set while a seek-bar drag is moving a clip marker rather than
    /// scrubbing playback.
    clip_drag: Option<ClipMarker>,

    // YouTube-style preview thumbnails for the current video: the bg
    // thread fills `seek_thumbs`, and the main thread lazily uploads
    // each one into `seek_thumb_textures` the first time it's needed.
    seek_thumbs: Arc<Mutex<Vec<Option<egui::ColorImage>>>>,
    seek_thumb_textures: Vec<Option<egui::TextureHandle>>,

    // UI state
    last_esc_press: Option<std::time::Instant>,
    toast: Option<Toast>,

    // History
    history: Vec<PathBuf>,
    history_index: Option<usize>,

    // View State
    zoom: f32,
    pan: egui::Vec2,

    // Filter
    media_filter: MediaFilter,
    view_order: ViewOrder,

    is_scanning: Arc<Mutex<bool>>,
    scan_count: Arc<Mutex<usize>>,
    texture: Option<egui::TextureHandle>,
    error_msg: Option<String>,
    pending_initial_file: Option<PathBuf>,
    wgpu_backend: Option<video_player::WgpuBackend>,
    /// Smoothed time between UI frames while video plays — the display
    /// refresh interval, near enough. A frame chosen in `update()` is on
    /// screen about one interval later, so the player picks for then.
    frame_interval_secs: f32,
}

impl PhotoViewer {
    fn new(cc: &eframe::CreationContext<'_>, initial_file: Option<PathBuf>) -> Self {
        let wgpu_backend = cc.wgpu_render_state.as_ref().map(|rs| {
            let supports_16bit_norm = rs
                .device
                .features()
                .contains(eframe::wgpu::Features::TEXTURE_FORMAT_16BIT_NORM);
            video_player::WgpuBackend {
                device: rs.device.clone(),
                queue: rs.queue.clone(),
                renderer: rs.renderer.clone(),
                target_format: rs.target_format,
                supports_16bit_norm,
            }
        });
        Self {
            media_paths: Arc::new(Mutex::new(Vec::new())),
            current_media_path: None,
            current_image: None,
            video_player: None,
            is_video: false,
            video_looping: true,
            video_volume: 0.10,
            video_pre_mute_volume: 0.10,
            video_has_audio: true,
            seek_frac: 0.0,
            video_rotation: 0,
            scrubbing: None,
            clip_start_ms: None,
            clip_end_ms: None,
            clip_drag: None,
            seek_thumbs: Arc::new(Mutex::new(Vec::new())),
            seek_thumb_textures: Vec::new(),
            last_esc_press: None,
            toast: None,
            history: Vec::new(),
            history_index: None,
            zoom: 1.0,
            pan: egui::Vec2::ZERO,
            media_filter: MediaFilter::All,
            view_order: ViewOrder::Random,
            wgpu_backend,
            frame_interval_secs: 1.0 / 60.0,
            is_scanning: Arc::new(Mutex::new(false)),
            scan_count: Arc::new(Mutex::new(0)),
            texture: None,
            error_msg: None,
            pending_initial_file: initial_file,
        }
    }

    fn open_directory(&mut self) {
        if let Some(path) = rfd::FileDialog::new().pick_folder() {
            self.start_scan(path);
        }
    }

    fn open_file_dialog(&mut self, ctx: &egui::Context) {
        let all_extensions: Vec<&str> = IMAGE_EXTENSIONS.iter()
            .chain(VIDEO_EXTENSIONS.iter())
            .copied()
            .collect();

        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Media", &all_extensions)
            .add_filter("Images", IMAGE_EXTENSIONS)
            .add_filter("Videos", VIDEO_EXTENSIONS)
            .pick_file()
        {
            if let Some(parent) = path.parent() {
                self.start_scan(parent.to_path_buf());
            }

            self.current_media_path = Some(path.clone());
            self.history.push(path.clone());
            self.history_index = Some(0);
            self.load_media(path, ctx);
        }
    }

    fn start_scan(&mut self, directory: PathBuf) {
        {
            let mut paths = self.media_paths.lock().unwrap();
            paths.clear();
        }
        self.history.clear();
        self.history_index = None;
        self.current_media_path = None;
        self.texture = None;
        self.video_player = None;
        self.is_video = false;
        self.reset_view();

        *self.scan_count.lock().unwrap() = 0;
        *self.is_scanning.lock().unwrap() = true;

        let paths_clone = self.media_paths.clone();
        let scanning_clone = self.is_scanning.clone();
        let count_clone = self.scan_count.clone();

        thread::spawn(move || {
            for entry in WalkDir::new(directory).into_iter().filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.is_file() && is_supported_file(path) {
                    let mut p = paths_clone.lock().unwrap();
                    p.push(path.to_path_buf());
                    *count_clone.lock().unwrap() += 1;
                }
            }
            // Sort for deterministic ordered navigation
            paths_clone.lock().unwrap().sort();
            *scanning_clone.lock().unwrap() = false;
        });
    }

    fn reset_view(&mut self) {
        self.zoom = 1.0;
        self.pan = egui::Vec2::ZERO;
        self.video_rotation = 0;
    }

    fn go_next(&mut self, ctx: &egui::Context) {
        if self.view_order == ViewOrder::Ordered {
            self.ordered_step(ctx, 1);
            return;
        }
        if let Some(idx) = self.history_index {
            // Look forward through history for a matching entry
            let mut next_idx = idx + 1;
            while next_idx < self.history.len() {
                if matches_filter(&self.history[next_idx], self.media_filter) {
                    self.history_index = Some(next_idx);
                    let path = self.history[next_idx].clone();
                    self.current_media_path = Some(path.clone());
                    self.reset_view();
                    self.load_media(path, ctx);
                    return;
                }
                next_idx += 1;
            }
        }
        self.next_random_media(ctx);
    }

    fn go_prev(&mut self, ctx: &egui::Context) {
        if self.view_order == ViewOrder::Ordered {
            self.ordered_step(ctx, -1);
            return;
        }
        if let Some(idx) = self.history_index {
            let mut prev_idx = idx;
            while prev_idx > 0 {
                prev_idx -= 1;
                if matches_filter(&self.history[prev_idx], self.media_filter) {
                    self.history_index = Some(prev_idx);
                    let path = self.history[prev_idx].clone();
                    self.current_media_path = Some(path.clone());
                    self.reset_view();
                    self.load_media(path, ctx);
                    return;
                }
            }
        }
    }

    /// Step through `media_paths` in sorted order by `delta` (+1 / -1),
    /// skipping files that don't match the current media filter. Wraps
    /// around at both ends.
    fn ordered_step(&mut self, ctx: &egui::Context, delta: isize) {
        let next_path = {
            let paths = self.media_paths.lock().unwrap();
            if paths.is_empty() {
                return;
            }
            let n = paths.len();
            let start = self
                .current_media_path
                .as_ref()
                .and_then(|p| paths.iter().position(|mp| mp == p))
                .map(|i| i as isize)
                .unwrap_or(if delta > 0 { -1 } else { n as isize });

            let mut found = None;
            for step in 1..=n {
                let idx = ((start + delta * step as isize).rem_euclid(n as isize)) as usize;
                if matches_filter(&paths[idx], self.media_filter) {
                    found = Some(paths[idx].clone());
                    break;
                }
            }
            found
        };

        if let Some(p) = next_path {
            self.current_media_path = Some(p.clone());
            self.reset_view();
            self.load_media(p, ctx);
        }
    }

    fn next_random_media(&mut self, ctx: &egui::Context) {
        let path = {
            let paths = self.media_paths.lock().unwrap();
            if paths.is_empty() {
                return;
            }

            let history_set: HashSet<&PathBuf> = self.history.iter().collect();

            // Filter by media type AND exclude already-seen files
            let available: Vec<&PathBuf> = paths.iter()
                .filter(|p| matches_filter(p, self.media_filter) && !history_set.contains(p))
                .collect();

            let mut rng = rand::thread_rng();

            if !available.is_empty() {
                available.choose(&mut rng).map(|p| (*p).clone())
            } else {
                // All matching files seen — pick any matching file except the current one
                let matching: Vec<&PathBuf> = paths.iter()
                    .filter(|p| matches_filter(p, self.media_filter) && Some(*p) != self.current_media_path.as_ref())
                    .collect();
                if !matching.is_empty() {
                    matching.choose(&mut rng).map(|p| (*p).clone())
                } else {
                    // Only one matching file (or none)
                    paths.iter()
                        .find(|p| matches_filter(p, self.media_filter))
                        .cloned()
                }
            }
        };

        if let Some(p) = path {
            self.history.push(p.clone());
            self.history_index = Some(self.history.len() - 1);

            self.current_media_path = Some(p.clone());
            self.reset_view();
            self.load_media(p, ctx);
        }
    }

    /// Position playback should return to when a finished video is
    /// restarted: the clip in-point when one is set, otherwise zero.
    fn replay_origin_us(&self) -> i64 {
        self.clip_start_ms.unwrap_or(0).max(0) * 1000
    }

    /// Play/pause toggle shared by the spacebar, the click-on-video
    /// gesture, and the transport button.
    fn toggle_playback(&mut self) {
        let origin_us = self.replay_origin_us();
        let Some(player) = &mut self.video_player else {
            return;
        };
        match player.state() {
            PlayerState::Playing => player.pause(),
            PlayerState::Paused => player.play(),
            PlayerState::EndOfFile => {
                player.seek_us(origin_us);
                player.play();
            }
            _ => player.play(),
        }
    }

    /// Clear the A-B clip loop and tell the player to stop wrapping.
    fn clear_clip_range(&mut self) {
        self.clip_start_ms = None;
        self.clip_end_ms = None;
        self.clip_drag = None;
        if let Some(player) = &mut self.video_player {
            player.set_loop_range(None, None);
        }
    }

    /// Push the current markers down to the player, normalising them
    /// first: markers are kept inside the video, ordered, and never
    /// closer together than `MIN_CLIP_MS` — a zero-length clip would
    /// make the demuxer seek on every frame.
    fn commit_clip_range(&mut self) {
        const MIN_CLIP_MS: i64 = 100;
        let Some(player) = &mut self.video_player else {
            return;
        };
        let duration_ms = player.duration_ms();
        if duration_ms <= 0 {
            return;
        }

        let clamp = |v: i64| v.clamp(0, duration_ms);
        let mut start = self.clip_start_ms.map(clamp);
        let mut end = self.clip_end_ms.map(clamp);

        // Markers set out of order are a normal way to work (mark the
        // end of an interesting moment, then its start), so swap
        // rather than reject.
        if let (Some(a), Some(b)) = (start, end) {
            if a > b {
                std::mem::swap(&mut start, &mut end);
            }
        }
        if let (Some(a), Some(b)) = (start, end) {
            if b - a < MIN_CLIP_MS {
                end = Some((a + MIN_CLIP_MS).min(duration_ms));
                if end == Some(a) {
                    start = Some((a - MIN_CLIP_MS).max(0));
                }
            }
        }

        self.clip_start_ms = start;
        self.clip_end_ms = end;
        player.set_loop_range(start.map(|ms| ms * 1000), end.map(|ms| ms * 1000));

        // Jump back into the clip if the playhead is already past it,
        // so setting an out-point behind the current position doesn't
        // leave playback stranded outside the loop.
        if let (Some(a), Some(b)) = (start, end) {
            if player.elapsed_ms() > b {
                player.seek_us(a * 1000);
            }
        }
    }

    /// Set one end of the clip loop to the current playback position.
    fn set_clip_marker_at_playhead(&mut self, marker: ClipMarker) {
        let Some(player) = &self.video_player else {
            return;
        };
        if player.duration_ms() <= 0 {
            return;
        }
        let now_ms = player.elapsed_ms();
        match marker {
            ClipMarker::Start => self.clip_start_ms = Some(now_ms),
            ClipMarker::End => self.clip_end_ms = Some(now_ms),
        }
        self.commit_clip_range();
        let (a, b) = (self.clip_start_ms, self.clip_end_ms);
        self.toast = Some(Toast::new(
            match (a, b) {
                (Some(a), Some(b)) => format!(
                    "Clip {} – {}  ({})",
                    Self::format_time(a),
                    Self::format_time(b),
                    Self::format_time(b - a)
                ),
                (Some(a), None) => format!("Clip start {}  (] sets the end)", Self::format_time(a)),
                (None, Some(b)) => format!("Clip end {}  ([ sets the start)", Self::format_time(b)),
                (None, None) => "Clip loop cleared".to_string(),
            },
            false,
        ));
    }

    /// Move the current file to the Recycle Bin and advance to the
    /// next one.
    fn delete_current_media(&mut self, ctx: &egui::Context) {
        let Some(path) = self.current_media_path.clone() else {
            return;
        };

        // The decoder thread holds the file open, and Windows refuses
        // to move a file with an open handle. Tear the player down
        // first — `Player::drop` waits for the decode thread to exit.
        let was_video = self.is_video;
        if was_video {
            self.video_player = None;
            self.is_video = false;
        }

        match recycle::move_to_recycle_bin(&path) {
            Ok(()) => {
                let name = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();
                self.forget_media(&path);
                self.toast = Some(Toast::new(format!("Moved to Recycle Bin: {}", name), false));
                self.advance_after_delete(&path, ctx);
            }
            Err(e) => {
                let msg = format!("Delete failed: {}", e);
                println!("{}", msg);
                self.toast = Some(Toast::new(msg, true));
                // The delete didn't happen, so put the viewer back the
                // way it was rather than leaving a blank screen.
                if was_video {
                    self.load_media(path, ctx);
                }
            }
        }
    }

    /// Drop every reference to a path that no longer exists.
    fn forget_media(&mut self, path: &Path) {
        // Take the two locks one at a time. The scan thread acquires
        // `media_paths` before `scan_count`, so holding `scan_count`
        // across a `media_paths` lock here would invert that order.
        let remaining = {
            let mut paths = self.media_paths.lock().unwrap();
            paths.retain(|p| p != path);
            paths.len()
        };
        *self.scan_count.lock().unwrap() = remaining;

        self.history.retain(|p| p != path);
        if self.history.is_empty() {
            self.history_index = None;
        } else if let Some(idx) = self.history_index {
            self.history_index = Some(idx.min(self.history.len() - 1));
        }
    }

    /// Show something after a delete: the next file in the current
    /// order, or an empty viewer when nothing is left.
    fn advance_after_delete(&mut self, deleted: &Path, ctx: &egui::Context) {
        // `ordered_step` navigates relative to `current_media_path`,
        // which has just been removed from the list. Anchor on the
        // nearest surviving neighbour instead.
        let replacement = {
            let paths = self.media_paths.lock().unwrap();
            paths
                .iter()
                .find(|p| p.as_path() > deleted && matches_filter(p, self.media_filter))
                .or_else(|| paths.iter().rev().find(|p| matches_filter(p, self.media_filter)))
                .cloned()
        };

        self.current_media_path = None;
        self.texture = None;
        self.current_image = None;
        self.clear_clip_range();
        self.reset_view();

        match (self.view_order, replacement) {
            (ViewOrder::Random, _) => self.next_random_media(ctx),
            (ViewOrder::Ordered, Some(next)) => {
                self.current_media_path = Some(next.clone());
                self.load_media(next, ctx);
            }
            (ViewOrder::Ordered, None) => {
                self.is_video = false;
            }
        }
    }

    fn open_in_explorer(&self) {
        if let Some(path) = &self.current_media_path {
            #[cfg(target_os = "windows")]
            {
                Command::new("explorer")
                    .args(["/select,", &path.to_string_lossy()])
                    .spawn()
                    .ok();
            }
            #[cfg(not(target_os = "windows"))]
            {
                let _ = path;
            }
        }
    }

    fn load_media(&mut self, path: PathBuf, ctx: &egui::Context) {
        // Clip markers refer to positions inside one specific video,
        // so they never carry over to the next file.
        self.clip_start_ms = None;
        self.clip_end_ms = None;
        self.clip_drag = None;
        if is_video_file(&path) {
            self.load_video(path, ctx);
        } else {
            self.load_image(path, ctx);
        }
    }

    fn load_video(&mut self, path: PathBuf, ctx: &egui::Context) {
        // Clear image state
        self.current_image = None;
        self.texture = None;

        // Reset any in-flight scrub and preview thumbnails.
        self.scrubbing = None;
        self.seek_thumb_textures.clear();
        const THUMB_COUNT: usize = 20;
        {
            let mut thumbs = self.seek_thumbs.lock().unwrap();
            *thumbs = vec![None; THUMB_COUNT];
        }
        self.seek_thumb_textures.resize_with(THUMB_COUNT, || None);
        {
            let path_clone = path.clone();
            let thumbs_clone = self.seek_thumbs.clone();
            let ctx_clone = ctx.clone();
            thread::spawn(move || {
                extract_seek_thumbnails(path_clone, THUMB_COUNT, thumbs_clone, ctx_clone);
            });
        }

        match Player::open_with_backend(ctx, &path, self.wgpu_backend.clone()) {
            Ok(mut player) => {
                player.set_looping(self.video_looping);
                player.set_volume(self.video_volume);
                player.play();
                self.video_has_audio = player.has_audio();
                self.video_player = Some(player);
                self.is_video = true;
                self.seek_frac = 0.0;
                self.error_msg = None;
            }
            Err(e) => {
                let msg = format!("Error loading video {}: {}", path.display(), e);
                println!("{}", msg);
                self.error_msg = Some(msg);
                self.video_player = None;
                self.is_video = false;
            }
        }
    }

    fn load_image(&mut self, path: PathBuf, ctx: &egui::Context) {
        // Clear video state
        self.video_player = None;
        self.is_video = false;

        let result = decode_image(&path);

        match result {
            Ok(image) => {
                self.current_image = Some(image);
                self.regenerate_texture(ctx);
                self.error_msg = None;
            }
            Err(e) => {
                let msg = format!("Error loading {}: {}", path.display(), e);
                println!("{}", msg);
                self.error_msg = Some(msg);
                self.current_image = None;
                self.texture = None;
            }
        }
    }

    fn regenerate_texture(&mut self, ctx: &egui::Context) {
        if let Some(image) = &self.current_image {
            // Clamp to the backend's max texture side so an oversized
            // image (e.g. 60+MP panorama on a GPU with a 16384 cap)
            // degrades to a scaled-down view instead of crashing wgpu
            // validation when the ColorImage is uploaded.
            let max_side = ctx.input(|i| i.max_texture_side).max(1);
            let (w, h) = (image.width() as usize, image.height() as usize);
            let over = w.max(h);
            let rgba = if over > max_side {
                let scale = max_side as f32 / over as f32;
                let new_w = ((w as f32 * scale).floor() as u32).max(1);
                let new_h = ((h as f32 * scale).floor() as u32).max(1);
                image
                    .resize(new_w, new_h, image::imageops::FilterType::Triangle)
                    .to_rgba8()
            } else {
                image.to_rgba8()
            };
            let size = [rgba.width() as usize, rgba.height() as usize];
            let pixels = rgba.as_flat_samples();

            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                size,
                pixels.as_slice(),
            );

            self.texture = Some(ctx.load_texture(
                "current_image",
                color_image,
                egui::TextureOptions::LINEAR,
            ));
        }
    }

    fn rotate_cw(&mut self, ctx: &egui::Context) {
        if let Some(img) = &self.current_image {
            self.current_image = Some(img.rotate90());
            self.regenerate_texture(ctx);
        }
    }

    fn rotate_ccw(&mut self, ctx: &egui::Context) {
        if let Some(img) = &self.current_image {
            self.current_image = Some(img.rotate270());
            self.regenerate_texture(ctx);
        }
    }

    fn format_time(ms: i64) -> String {
        let total_secs = ms / 1000;
        let hours = total_secs / 3600;
        let mins = (total_secs % 3600) / 60;
        let secs = total_secs % 60;
        let millis = (ms.rem_euclid(1000)) as i32;
        if hours > 0 {
            format!("{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
        } else {
            format!("{:02}:{:02}.{:03}", mins, secs, millis)
        }
    }

    /// Render a video frame with rotation using a custom textured mesh.
    fn render_rotated_video(ui: &mut egui::Ui, texture_id: egui::TextureId, rect: egui::Rect, rotation: u16) {
        // UV coordinates for each rotation (top-left, top-right, bottom-right, bottom-left)
        let uvs: [(f32, f32); 4] = match rotation {
            90  => [(0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            180 => [(1.0, 1.0), (0.0, 1.0), (0.0, 0.0), (1.0, 0.0)],
            270 => [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)],
            _   => [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        };

        let white = egui::Color32::WHITE;
        let mut mesh = egui::Mesh::with_texture(texture_id);
        mesh.vertices.push(egui::epaint::Vertex { pos: rect.left_top(),     uv: egui::pos2(uvs[0].0, uvs[0].1), color: white });
        mesh.vertices.push(egui::epaint::Vertex { pos: rect.right_top(),    uv: egui::pos2(uvs[1].0, uvs[1].1), color: white });
        mesh.vertices.push(egui::epaint::Vertex { pos: rect.right_bottom(), uv: egui::pos2(uvs[2].0, uvs[2].1), color: white });
        mesh.vertices.push(egui::epaint::Vertex { pos: rect.left_bottom(),  uv: egui::pos2(uvs[3].0, uvs[3].1), color: white });
        mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
        ui.painter().add(egui::Shape::mesh(mesh));
    }
}

impl eframe::App for PhotoViewer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle deferred initial file loading (first frame only)
        if let Some(path) = self.pending_initial_file.take() {
            if path.exists() && path.is_file() {
                if let Some(parent) = path.parent() {
                    self.start_scan(parent.to_path_buf());
                }
                self.current_media_path = Some(path.clone());
                self.history.push(path.clone());
                self.history_index = Some(0);
                self.load_media(path, ctx);
            }
        }

        // Upload any new decoded video frames.
        if let Some(player) = &mut self.video_player {
            // Video repaints continuously, so the gap between updates
            // tracks the display's refresh interval. Ignore outliers from
            // idle periods and window drags.
            let dt = ctx.input(|i| i.unstable_dt);
            if (0.002..0.1).contains(&dt) {
                self.frame_interval_secs = self.frame_interval_secs * 0.9 + dt * 0.1;
            }
            player.set_display_lead(std::time::Duration::from_secs_f32(self.frame_interval_secs));
            player.tick();
        }

        // Handle input differently for video vs image mode
        let ctrl_held = ctx.input(|i| i.modifiers.ctrl);

        if self.is_video {
            // Video mode: arrows seek, ctrl+arrows navigate
            if ctrl_held {
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
                    self.go_next(ctx);
                }
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
                    self.go_prev(ctx);
                }
            } else {
                // Seek ±3 seconds
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
                    if let Some(player) = &mut self.video_player {
                        let duration = player.duration_ms();
                        if duration > 0 {
                            let step = 3000.0 / duration as f32;
                            let current = player.elapsed_ms() as f32 / duration as f32;
                            let target = (current + step).clamp(0.0, 1.0);
                            player.seek(target);
                        }
                    }
                }
                if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
                    if let Some(player) = &mut self.video_player {
                        let duration = player.duration_ms();
                        if duration > 0 {
                            let step = 3000.0 / duration as f32;
                            let current = player.elapsed_ms() as f32 / duration as f32;
                            let target = (current - step).clamp(0.0, 1.0);
                            player.seek(target);
                        }
                    }
                }
            }
            // Frame-step: comma = back 1 frame, period = forward 1 frame.
            // Auto-pauses on first use so stepping is predictable.
            if ctx.input(|i| i.key_pressed(egui::Key::Comma)) {
                if let Some(player) = &mut self.video_player {
                    if matches!(player.state(), PlayerState::Playing) {
                        player.pause();
                    }
                    player.step_frames(-1);
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Period)) {
                if let Some(player) = &mut self.video_player {
                    if matches!(player.state(), PlayerState::Playing) {
                        player.pause();
                    }
                    player.step_frames(1);
                }
            }
            // Volume: Up/Down arrows ±2%
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowUp)) {
                self.video_volume = (self.video_volume + 0.02).clamp(0.0, 1.0);
                if let Some(player) = &mut self.video_player {
                    player.set_volume(self.video_volume);
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowDown)) {
                self.video_volume = (self.video_volume - 0.02).clamp(0.0, 1.0);
                if let Some(player) = &mut self.video_player {
                    player.set_volume(self.video_volume);
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Space)) {
                // Space toggles play/pause for video
                self.toggle_playback();
            }
            // A-B clip loop: [ marks the in-point, ] the out-point,
            // \ clears both.
            if ctx.input(|i| i.key_pressed(egui::Key::OpenBracket)) {
                self.set_clip_marker_at_playhead(ClipMarker::Start);
            }
            if ctx.input(|i| i.key_pressed(egui::Key::CloseBracket)) {
                self.set_clip_marker_at_playhead(ClipMarker::End);
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Backslash)) {
                self.clear_clip_range();
                self.toast = Some(Toast::new("Clip loop cleared", false));
            }
        } else {
            // Image mode: original behavior
            if ctx.input(|i| i.key_pressed(egui::Key::Space) || i.key_pressed(egui::Key::ArrowRight)) {
                self.go_next(ctx);
            }
            if ctx.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
                self.go_prev(ctx);
            }
        }

        // Delete: move the current file to the Recycle Bin. Recoverable
        // from Explorer, so it acts immediately rather than prompting.
        if ctx.input(|i| i.key_pressed(egui::Key::Delete)) {
            self.delete_current_media(ctx);
        }

        if ctx.input(|i| i.key_pressed(egui::Key::O)) {
            self.open_directory();
        }
        if ctx.input(|i| i.key_pressed(egui::Key::F)) {
            self.open_file_dialog(ctx);
        }

        // F11: toggle fullscreen (both modes)
        if ctx.input(|i| i.key_pressed(egui::Key::F11)) {
            let is_fullscreen = ctx.input(|i| i.viewport().fullscreen.unwrap_or(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(!is_fullscreen));
        }

        // Double-tap Esc within 500ms to close the app
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            let now = std::time::Instant::now();
            if let Some(last) = self.last_esc_press {
                if now.duration_since(last) < std::time::Duration::from_millis(500) {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            self.last_esc_press = Some(now);
        }

        // M: cycle media filter
        if ctx.input(|i| i.key_pressed(egui::Key::M)) {
            self.media_filter = self.media_filter.cycle();
        }

        // R: toggle random vs ordered viewing
        if ctx.input(|i| i.key_pressed(egui::Key::R)) {
            self.view_order = self.view_order.toggle();
        }

        // 0: reset zoom + pan to defaults
        if ctx.input(|i| i.key_pressed(egui::Key::Num0)) {
            self.zoom = 1.0;
            self.pan = egui::Vec2::ZERO;
        }

        // Handle Zoom (Mouse Wheel + keyboard) - images and videos
        {
            let scroll = ctx.input(|i| i.raw_scroll_delta);
            if scroll.y != 0.0 {
                let zoom_factor = if scroll.y > 0.0 { 1.1 } else { 0.9 };
                self.zoom *= zoom_factor;
                self.zoom = self.zoom.clamp(0.1, 50.0);
            }
            // +/= to zoom in, - to zoom out
            if ctx.input(|i| i.key_pressed(egui::Key::Plus) || i.key_pressed(egui::Key::Equals)) {
                self.zoom *= 1.15;
                self.zoom = self.zoom.clamp(0.1, 50.0);
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Minus)) {
                self.zoom *= 0.87;
                self.zoom = self.zoom.clamp(0.1, 50.0);
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let rect = ui.available_rect_before_wrap();
            ui.painter().rect_filled(rect, 0.0, egui::Color32::BLACK);

            if self.is_video {
                // Set when a control inside the panel changes the clip
                // markers. `commit_clip_range` needs `&mut self`, which
                // can't be taken while `player` is borrowed below, so
                // the work is deferred until that borrow ends.
                let mut clip_dirty = false;
                // Video rendering
                if let Some(player) = &mut self.video_player {
                    let available = rect.size();
                    // Reserve space for the floating controls panel at the
                    // bottom so panning doesn't sit under them.
                    let controls_height = 120.0;
                    let video_area = egui::vec2(available.x, available.y - controls_height);

                    // Click-to-pause and drag-to-pan on the video area (above controls)
                    let interact_rect = egui::Rect::from_min_size(rect.min, video_area);
                    let video_interact = ui.interact(interact_rect, ui.id().with("video_interact"), egui::Sense::click_and_drag());
                    if video_interact.dragged() {
                        self.pan += video_interact.drag_delta();
                    }
                    if video_interact.clicked() {
                        let origin_us = self.clip_start_ms.unwrap_or(0).max(0) * 1000;
                        match player.state() {
                            PlayerState::Playing => player.pause(),
                            PlayerState::Paused => player.play(),
                            PlayerState::EndOfFile => {
                                player.seek_us(origin_us);
                                player.play();
                            }
                            _ => {}
                        }
                    }

                    // Scale video to fit while maintaining aspect ratio
                    // For 90/270 rotation, swap the video dimensions for aspect ratio calc
                    let (decl_w, decl_h) = player.size();
                    let (intr_w, intr_h) = player.intrinsic_size();
                    let (src_w, src_h) = if intr_w > 0 && intr_h > 0 {
                        (intr_w, intr_h)
                    } else {
                        (decl_w, decl_h)
                    };
                    let video_size = egui::vec2(src_w as f32, src_h as f32);
                    let (effective_w, effective_h) = if self.video_rotation == 90 || self.video_rotation == 270 {
                        (video_size.y, video_size.x)
                    } else {
                        (video_size.x, video_size.y)
                    };
                    let scale = if effective_w > 0.0 && effective_h > 0.0 {
                        (video_area.x / effective_w).min(video_area.y / effective_h)
                    } else {
                        1.0
                    };
                    // Apply zoom to display size
                    let display_size = egui::vec2(effective_w * scale * self.zoom, effective_h * scale * self.zoom);

                    // Center the video in the available area, with pan offset
                    let video_rect = egui::Rect::from_center_size(
                        egui::pos2(
                            rect.min.x + available.x / 2.0,
                            rect.min.y + video_area.y / 2.0,
                        ) + self.pan,
                        display_size,
                    );

                    // Tell the NV12 renderer the *unclamped* video
                    // rect so its vertex shader can compute the right
                    // NDC slice. Without this the shader draws a
                    // full-NDC quad into egui's clamped viewport and
                    // panning past the edge looks like a resize.
                    if player.is_nv12_path() {
                        let screen_rect = ctx.screen_rect();
                        player.set_nv12_render_rect(
                            video_rect,
                            screen_rect.size(),
                            ctx.pixels_per_point(),
                        );
                    }

                    // Render via painter directly (not render_frame_at) so our
                    // click interaction isn't consumed by an internal widget.
                    if player.is_nv12_path() {
                        // Partial Phase E: custom wgpu pipeline that
                        // samples Y + UV and does YUV→RGB in the
                        // fragment shader. Rotation is not yet
                        // supported on this path.
                        if let Some(cb) = player.nv12_paint_callback(video_rect) {
                            ui.painter().add(egui::Shape::Callback(cb));
                        }
                    } else if let Some(texture_id) = player.texture_id() {
                        if self.video_rotation == 0 {
                            ui.painter().image(
                                texture_id,
                                video_rect,
                                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                egui::Color32::WHITE,
                            );
                        } else {
                            Self::render_rotated_video(ui, texture_id, video_rect, self.video_rotation);
                        }
                    }

                    // Floating controls panel: centered, rounded, not full
                    // width. Two rows — a wide seek slider for precise
                    // scrubbing on top, buttons/volume/fullscreen below.
                    let panel_w = (available.x * 0.80).clamp(420.0, 1100.0);
                    let panel_h = 88.0;
                    let panel_bottom_margin = 0.0;
                    let panel_rect = egui::Rect::from_min_size(
                        egui::pos2(
                            rect.min.x + (available.x - panel_w) / 2.0,
                            rect.max.y - panel_h - panel_bottom_margin,
                        ),
                        egui::vec2(panel_w, panel_h),
                    );

                    egui::Area::new(egui::Id::new("VideoControls"))
                        .fixed_pos(panel_rect.min)
                        .order(egui::Order::Foreground)
                        .show(ctx, |ui| {
                            ui.set_width(panel_rect.width());
                            ui.set_height(panel_rect.height());
                            egui::Frame::none()
                                .fill(egui::Color32::from_black_alpha(200))
                                .rounding(14.0)
                                .inner_margin(egui::Margin::symmetric(14.0, 10.0))
                                .shadow(egui::epaint::Shadow {
                                    offset: egui::vec2(0.0, 3.0),
                                    blur: 12.0,
                                    spread: 0.0,
                                    color: egui::Color32::from_black_alpha(140),
                                })
                                .show(ui, |ui| {
                                    ui.set_width(panel_rect.width() - 28.0);
                            let elapsed = player.elapsed_ms();
                            let duration = player.duration_ms();

                            ui.vertical(|ui| {
                                // ---- Row 1: custom-drawn seek bar ----
                                let bar_height = 12.0;
                                let (bar_rect, bar_response) = ui.allocate_exact_size(
                                    egui::vec2(ui.available_width(), bar_height),
                                    egui::Sense::click_and_drag(),
                                );

                                let pointer_frac = bar_response
                                    .interact_pointer_pos()
                                    .or_else(|| ui.input(|i| i.pointer.hover_pos()))
                                    .map(|pos| {
                                        ((pos.x - bar_rect.left()) / bar_rect.width())
                                            .clamp(0.0, 1.0)
                                    });

                                // Clip markers, as fractions of the bar,
                                // so hit-testing and painting work off the
                                // same numbers.
                                let to_frac = |ms: i64| {
                                    if duration > 0 {
                                        (ms as f32 / duration as f32).clamp(0.0, 1.0)
                                    } else {
                                        0.0
                                    }
                                };
                                let clip_a = self.clip_start_ms.map(to_frac);
                                let clip_b = self.clip_end_ms.map(to_frac);

                                // A drag that starts on a marker moves it;
                                // anywhere else scrubs. Hit-test in pixels so
                                // the grab area stays the same size no matter
                                // how long the video is.
                                let grab_px = 7.0;
                                let marker_under = |frac: f32| -> Option<ClipMarker> {
                                    let x = bar_rect.left() + bar_rect.width() * frac;
                                    let near = |m: Option<f32>| {
                                        m.map(|mf| {
                                            (bar_rect.left() + bar_rect.width() * mf - x)
                                                .abs()
                                                <= grab_px
                                        })
                                        .unwrap_or(false)
                                    };
                                    if near(clip_a) {
                                        Some(ClipMarker::Start)
                                    } else if near(clip_b) {
                                        Some(ClipMarker::End)
                                    } else {
                                        None
                                    }
                                };

                                // While dragging, buffer the target position and
                                // commit it on release — calling player.seek() on
                                // every drag frame causes noticeable lag.
                                if bar_response.drag_started() {
                                    if let Some(f) = pointer_frac {
                                        match marker_under(f) {
                                            Some(marker) => self.clip_drag = Some(marker),
                                            None => self.scrubbing = Some(f),
                                        }
                                    }
                                }
                                if bar_response.dragged() {
                                    if let Some(f) = pointer_frac {
                                        match self.clip_drag {
                                            Some(ClipMarker::Start) => {
                                                self.clip_start_ms =
                                                    Some((f * duration as f32) as i64);
                                            }
                                            Some(ClipMarker::End) => {
                                                self.clip_end_ms =
                                                    Some((f * duration as f32) as i64);
                                            }
                                            None => self.scrubbing = Some(f),
                                        }
                                    }
                                }
                                if bar_response.drag_stopped() {
                                    if self.clip_drag.take().is_some() {
                                        // Normalising can swap the markers, so
                                        // only push the result down on release.
                                        clip_dirty = true;
                                    } else if let Some(f) = self.scrubbing.take() {
                                        if duration > 0 {
                                            player.seek(f);
                                        }
                                    }
                                }
                                if bar_response.clicked() && duration > 0 {
                                    if let Some(f) = pointer_frac {
                                        player.seek(f);
                                    }
                                }

                                let played_frac = if duration > 0 {
                                    (elapsed as f32 / duration as f32).clamp(0.0, 1.0)
                                } else {
                                    0.0
                                };
                                let display_frac = self.scrubbing.unwrap_or(played_frac);

                                // Recompute after the drag so the paint below
                                // reflects this frame's marker positions.
                                let clip_a = self.clip_start_ms.map(to_frac);
                                let clip_b = self.clip_end_ms.map(to_frac);

                                let painter = ui.painter();
                                let rounding = egui::Rounding::same(bar_height * 0.5);
                                painter.rect_filled(
                                    bar_rect,
                                    rounding,
                                    egui::Color32::from_rgb(55, 55, 60),
                                );

                                // Selected clip region, drawn under the
                                // progress fill so the playhead stays legible.
                                const CLIP_COLOR: egui::Color32 =
                                    egui::Color32::from_rgb(255, 186, 72);
                                if let (Some(a), Some(b)) = (clip_a, clip_b) {
                                    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
                                    let mut band = bar_rect;
                                    band.min.x = bar_rect.left() + bar_rect.width() * lo;
                                    band.max.x = bar_rect.left() + bar_rect.width() * hi;
                                    painter.rect_filled(
                                        band,
                                        egui::Rounding::ZERO,
                                        CLIP_COLOR.gamma_multiply(0.35),
                                    );
                                }

                                if display_frac > 0.0 {
                                    let mut filled = bar_rect;
                                    filled.max.x =
                                        filled.min.x + bar_rect.width() * display_frac;
                                    painter.rect_filled(
                                        filled,
                                        rounding,
                                        egui::Color32::from_rgb(100, 200, 255),
                                    );
                                }

                                // Marker posts, tall enough to grab and drawn
                                // above the fill so they read as handles
                                // rather than as progress.
                                for marker in [clip_a, clip_b].into_iter().flatten() {
                                    let x = bar_rect.left() + bar_rect.width() * marker;
                                    let post = egui::Rect::from_center_size(
                                        egui::pos2(x, bar_rect.center().y),
                                        egui::vec2(4.0, bar_height + 8.0),
                                    );
                                    painter.rect_filled(
                                        post,
                                        egui::Rounding::same(2.0),
                                        CLIP_COLOR,
                                    );
                                }

                                let handle_x =
                                    bar_rect.left() + bar_rect.width() * display_frac;
                                painter.circle_filled(
                                    egui::pos2(handle_x, bar_rect.center().y),
                                    bar_height * 0.8,
                                    egui::Color32::WHITE,
                                );

                                // ---- Preview thumbnail (YouTube-style) ----
                                let show_preview = (bar_response.hovered()
                                    || bar_response.dragged())
                                    && duration > 0;
                                if show_preview {
                                    if let Some(frac) = pointer_frac {
                                        // Lazily upload any newly-decoded
                                        // thumbnails into GPU textures.
                                        {
                                            let thumbs = self.seek_thumbs.lock().unwrap();
                                            for (i, slot) in thumbs.iter().enumerate() {
                                                if let Some(img) = slot {
                                                    if self
                                                        .seek_thumb_textures
                                                        .get(i)
                                                        .and_then(|t| t.as_ref())
                                                        .is_none()
                                                    {
                                                        let handle = ctx.load_texture(
                                                            format!("seek_thumb_{}", i),
                                                            img.clone(),
                                                            egui::TextureOptions::LINEAR,
                                                        );
                                                        if i < self.seek_thumb_textures.len() {
                                                            self.seek_thumb_textures[i] =
                                                                Some(handle);
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        let count = self.seek_thumb_textures.len().max(1);
                                        let idx = ((frac * count as f32) as usize)
                                            .min(count - 1);
                                        // Find the nearest available thumb
                                        // (search outward from idx).
                                        let mut found: Option<&egui::TextureHandle> = None;
                                        for step in 0..count {
                                            let candidates = [
                                                idx.saturating_sub(step),
                                                (idx + step).min(count - 1),
                                            ];
                                            for c in candidates {
                                                if let Some(Some(tex)) =
                                                    self.seek_thumb_textures.get(c)
                                                {
                                                    found = Some(tex);
                                                    break;
                                                }
                                            }
                                            if found.is_some() {
                                                break;
                                            }
                                        }

                                        if let Some(tex) = found {
                                            let tex_size = tex.size_vec2();
                                            let preview_w = 192.0_f32;
                                            let preview_h = if tex_size.x > 0.0 {
                                                preview_w * tex_size.y / tex_size.x
                                            } else {
                                                108.0
                                            };
                                            let caption_h = 18.0;
                                            let total_h = preview_h + caption_h;
                                            let padding = 4.0;

                                            let cursor_x = bar_rect.left()
                                                + bar_rect.width() * frac;
                                            let preview_bottom = bar_rect.top() - 10.0;
                                            let mut preview_left = cursor_x - preview_w / 2.0;
                                            // Clamp within the panel so it doesn't
                                            // slip off either edge.
                                            let clamp_left = panel_rect.left() + 6.0;
                                            let clamp_right = panel_rect.right() - 6.0;
                                            if preview_left < clamp_left {
                                                preview_left = clamp_left;
                                            }
                                            if preview_left + preview_w > clamp_right {
                                                preview_left = clamp_right - preview_w;
                                            }
                                            let preview_rect = egui::Rect::from_min_size(
                                                egui::pos2(
                                                    preview_left,
                                                    preview_bottom - total_h,
                                                ),
                                                egui::vec2(preview_w, total_h),
                                            );

                                            // Draw on a top-layer painter so
                                            // the preview sits above the panel.
                                            let layer = egui::LayerId::new(
                                                egui::Order::Foreground,
                                                egui::Id::new("seek_preview"),
                                            );
                                            let top = ctx.layer_painter(layer);
                                            let bg_rect = preview_rect.expand(padding);
                                            top.rect_filled(
                                                bg_rect,
                                                egui::Rounding::same(6.0),
                                                egui::Color32::from_black_alpha(220),
                                            );
                                            let img_rect = egui::Rect::from_min_size(
                                                preview_rect.min,
                                                egui::vec2(preview_w, preview_h),
                                            );
                                            top.image(
                                                tex.id(),
                                                img_rect,
                                                egui::Rect::from_min_max(
                                                    egui::pos2(0.0, 0.0),
                                                    egui::pos2(1.0, 1.0),
                                                ),
                                                egui::Color32::WHITE,
                                            );
                                            let hover_ms = (frac * duration as f32) as i64;
                                            top.text(
                                                egui::pos2(
                                                    preview_rect.center().x,
                                                    preview_rect.min.y + preview_h + 2.0,
                                                ),
                                                egui::Align2::CENTER_TOP,
                                                Self::format_time(hover_ms),
                                                egui::FontId::monospace(12.0),
                                                egui::Color32::WHITE,
                                            );
                                        } else {
                                            // Thumbs not ready yet — at least
                                            // show a time tooltip.
                                            let hover_ms = (frac * duration as f32) as i64;
                                            bar_response.clone().on_hover_text(
                                                Self::format_time(hover_ms),
                                            );
                                        }
                                    }
                                }

                                ui.add_space(6.0);

                                // ---- Row 2: buttons + time + volume + fullscreen ----
                                ui.horizontal(|ui| {
                                    let state = player.state();
                                    let btn_text = match state {
                                        PlayerState::Playing => "⏸",
                                        _ => "▶",
                                    };
                                    if ui.button(egui::RichText::new(btn_text).size(16.0)).clicked() {
                                        let origin_us =
                                            self.clip_start_ms.unwrap_or(0).max(0) * 1000;
                                        match state {
                                            PlayerState::Playing => player.pause(),
                                            PlayerState::Paused => player.play(),
                                            PlayerState::EndOfFile => {
                                                player.seek_us(origin_us);
                                                player.play();
                                            }
                                            _ => player.play(),
                                        }
                                    }

                                    ui.label(
                                        egui::RichText::new(format!(
                                            "{} / {}",
                                            Self::format_time(elapsed),
                                            Self::format_time(duration),
                                        ))
                                        .color(egui::Color32::WHITE)
                                        .monospace(),
                                    );

                                    // Loop toggle
                                    let loop_btn = ui.button(
                                        egui::RichText::new("🔁").color(
                                            if self.video_looping {
                                                egui::Color32::from_rgb(100, 200, 255)
                                            } else {
                                                egui::Color32::GRAY
                                            },
                                        ),
                                    );
                                    if loop_btn
                                        .on_hover_text(
                                            "Toggle loop — also controls whether an \
                                             A-B clip repeats",
                                        )
                                        .clicked()
                                    {
                                        self.video_looping = !self.video_looping;
                                        player.set_looping(self.video_looping);
                                    }

                                    // ---- A-B clip loop ----
                                    let clip_set = self.clip_start_ms.is_some()
                                        || self.clip_end_ms.is_some();
                                    let marker_color = |set: bool| {
                                        if set {
                                            egui::Color32::from_rgb(255, 186, 72)
                                        } else {
                                            egui::Color32::GRAY
                                        }
                                    };
                                    if ui
                                        .button(
                                            egui::RichText::new("[")
                                                .monospace()
                                                .size(15.0)
                                                .color(marker_color(
                                                    self.clip_start_ms.is_some(),
                                                )),
                                        )
                                        .on_hover_text("Set clip start at playhead ( [ )")
                                        .clicked()
                                    {
                                        self.clip_start_ms = Some(elapsed);
                                        clip_dirty = true;
                                    }
                                    if ui
                                        .button(
                                            egui::RichText::new("]")
                                                .monospace()
                                                .size(15.0)
                                                .color(marker_color(
                                                    self.clip_end_ms.is_some(),
                                                )),
                                        )
                                        .on_hover_text("Set clip end at playhead ( ] )")
                                        .clicked()
                                    {
                                        self.clip_end_ms = Some(elapsed);
                                        clip_dirty = true;
                                    }
                                    if clip_set {
                                        if ui
                                            .button(
                                                egui::RichText::new("✖")
                                                    .color(egui::Color32::from_rgb(
                                                        255, 186, 72,
                                                    )),
                                            )
                                            .on_hover_text("Clear clip loop ( \\ )")
                                            .clicked()
                                        {
                                            self.clip_start_ms = None;
                                            self.clip_end_ms = None;
                                            clip_dirty = true;
                                        }
                                        // Spell the selection out: the
                                        // markers on the bar show where it
                                        // is, not how long it runs.
                                        let label = match (self.clip_start_ms, self.clip_end_ms) {
                                            (Some(a), Some(b)) => format!(
                                                "{} – {}  ({})",
                                                Self::format_time(a),
                                                Self::format_time(b),
                                                Self::format_time((b - a).abs()),
                                            ),
                                            (Some(a), None) => {
                                                format!("{} – end?", Self::format_time(a))
                                            }
                                            (None, Some(b)) => {
                                                format!("start? – {}", Self::format_time(b))
                                            }
                                            (None, None) => String::new(),
                                        };
                                        ui.label(
                                            egui::RichText::new(label)
                                                .color(egui::Color32::from_rgb(255, 186, 72))
                                                .monospace()
                                                .size(11.0),
                                        );
                                    }

                                    // Right-side cluster: fullscreen on the far right,
                                    // volume immediately left of it. `with_layout`
                                    // lays children out right-to-left so the items
                                    // read left-to-right in the final result.
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            let is_fullscreen = ctx.input(|i| {
                                                i.viewport().fullscreen.unwrap_or(false)
                                            });
                                            let fs_icon = if is_fullscreen { "⊡" } else { "⊞" };
                                            if ui
                                                .button(
                                                    egui::RichText::new(fs_icon)
                                                        .color(egui::Color32::WHITE),
                                                )
                                                .on_hover_text("Toggle fullscreen (F11)")
                                                .clicked()
                                            {
                                                ctx.send_viewport_cmd(
                                                    egui::ViewportCommand::Fullscreen(!is_fullscreen),
                                                );
                                            }

                                            if self.video_has_audio {
                                                let vol_pct = (self.video_volume * 100.0)
                                                    .round() as i32;
                                                ui.label(
                                                    egui::RichText::new(format!("{}%", vol_pct))
                                                        .color(egui::Color32::WHITE)
                                                        .monospace(),
                                                );
                                                let vol_slider = egui::Slider::new(
                                                    &mut self.video_volume,
                                                    0.0..=1.0,
                                                )
                                                .show_value(false)
                                                .trailing_fill(true);
                                                let vol_response =
                                                    ui.add_sized([110.0, 20.0], vol_slider);
                                                if vol_response.changed() {
                                                    player.set_volume(self.video_volume);
                                                }
                                                vol_response.on_hover_text(format!("{}%", vol_pct));

                                                let vol_icon = if self.video_volume == 0.0 {
                                                    "🔇"
                                                } else {
                                                    "🔊"
                                                };
                                                if ui
                                                    .add(egui::Button::new(
                                                        egui::RichText::new(vol_icon)
                                                            .color(egui::Color32::WHITE),
                                                    ))
                                                    .on_hover_text("Mute / unmute")
                                                    .clicked()
                                                {
                                                    if self.video_volume > 0.0 {
                                                        self.video_pre_mute_volume =
                                                            self.video_volume;
                                                        self.video_volume = 0.0;
                                                    } else {
                                                        self.video_volume = if self
                                                            .video_pre_mute_volume
                                                            > 0.0
                                                        {
                                                            self.video_pre_mute_volume
                                                        } else {
                                                            0.10
                                                        };
                                                    }
                                                    player.set_volume(self.video_volume);
                                                }
                                            } else {
                                                ui.label(
                                                    egui::RichText::new("🔇 No Audio").color(
                                                        egui::Color32::from_rgb(180, 180, 180),
                                                    ),
                                                );
                                            }
                                        },
                                    );
                                });
                            });
                                });
                        });

                    // Request continuous repaint for video playback
                    ctx.request_repaint();
                }
                if clip_dirty {
                    self.commit_clip_range();
                }
            } else {
                // Image rendering (original logic)
                let response = ui.interact(rect, ui.id().with("pan_drag"), egui::Sense::drag());
                if response.dragged() {
                    self.pan += response.drag_delta();
                }

                if let Some(texture) = &self.texture {
                    let available_size = rect.size();
                    let original_size = texture.size_vec2();

                    if original_size.x > 0.0 && original_size.y > 0.0 {
                        let width_ratio = available_size.x / original_size.x;
                        let height_ratio = available_size.y / original_size.y;
                        let base_scale = width_ratio.min(height_ratio);

                        let display_size = original_size * base_scale * self.zoom;

                        let center_x = rect.min.x + available_size.x / 2.0;
                        let center_y = rect.min.y + available_size.y / 2.0;

                        let image_rect = egui::Rect::from_center_size(
                            egui::pos2(center_x, center_y) + self.pan,
                            display_size,
                        );

                        ui.painter().image(
                            texture.id(),
                            image_rect,
                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                            egui::Color32::WHITE,
                        );
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label(
                            egui::RichText::new(
                                "Press 'O' to Open Folder  |  'F' to Open File\n\n\
                                 Images: Space / Right Arrow = Next  |  Left Arrow = Prev  |  +/- Zoom\n\
                                 Videos: Left/Right Arrow = Seek 3s  |  Ctrl + Left/Right = Prev/Next\n\
                                 Videos: , / . = Step back / forward one frame (auto-pauses)\n\
                                 Videos: [ / ] = Set clip start / end  |  \\ = Clear clip loop\n\
                                 Space: Play/Pause (video)  |  Next (image)\n\
                                 Up/Down: Volume ±2%  |  Click 🔊 to mute / unmute\n\
                                 Delete: Move current file to the Recycle Bin\n\
                                 Scroll to Zoom  |  Drag to Pan  |  0: Reset View\n\
                                 F11: Fullscreen  |  M: Filter  |  R: Random / Ordered\n\
                                 Double-tap Esc: Close",
                            )
                            .color(egui::Color32::WHITE)
                            .size(18.0),
                        );
                    });
                }
            }

            // Overlays (Status)
            let scanning = *self.is_scanning.lock().unwrap();
            let count = *self.scan_count.lock().unwrap();

            if scanning || count > 0 {
                egui::Window::new("Status")
                    .anchor(egui::Align2::LEFT_TOP, [10.0, 10.0])
                    .title_bar(false)
                    .resizable(false)
                    .auto_sized()
                    .frame(egui::Frame::popup(ui.style()).multiply_with_opacity(0.8))
                    .show(ctx, |ui| {
                        if scanning {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label(format!("Scanning... Found {} media files", count));
                            });
                            ctx.request_repaint();
                        } else {
                            ui.label(format!("Total: {} media files", count));
                            if let Some(path) = &self.current_media_path {
                                ui.label(
                                    path.file_name()
                                        .unwrap_or_default()
                                        .to_string_lossy(),
                                );
                                if self.is_video {
                                    ui.label("(Video)");
                                }
                                if let Some(idx) = self.history_index {
                                    ui.label(format!(
                                        "History: {}/{}",
                                        idx + 1,
                                        self.history.len()
                                    ));
                                }
                            }
                        }
                        // Show active filter
                        if self.media_filter != MediaFilter::All {
                            ui.colored_label(
                                egui::Color32::from_rgb(100, 200, 255),
                                format!("Filter: {}", self.media_filter.label()),
                            );
                        }
                        ui.colored_label(
                            egui::Color32::from_rgb(100, 200, 255),
                            format!("Order: {}", self.view_order.label()),
                        );
                        if let Some(err) = &self.error_msg {
                            ui.colored_label(egui::Color32::RED, err);
                        }
                    });
            }

            // Controls overlay — rotation for both images and videos.
            // When a video is loaded we lift the overlay above the
            // floating playback panel so the rotate / explorer
            // buttons don't get hidden underneath it.
            if self.current_media_path.is_some() {
                // `delete_current_media` needs `&mut self`, which the
                // window closure below is already holding.
                let mut delete_requested = false;
                let controls_offset_y = if self.is_video { -128.0 } else { -10.0 };
                egui::Window::new("Controls")
                    .anchor(
                        egui::Align2::RIGHT_BOTTOM,
                        [-10.0, controls_offset_y],
                    )
                    .title_bar(false)
                    .resizable(false)
                    .auto_sized()
                    .frame(egui::Frame::popup(ui.style()).multiply_with_opacity(0.8))
                    .show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            if ui
                                .button("⟲")
                                .on_hover_text("Rotate Left")
                                .clicked()
                            {
                                if self.is_video {
                                    self.video_rotation = (self.video_rotation + 270) % 360;
                                } else {
                                    self.rotate_ccw(ctx);
                                }
                            }
                            if ui
                                .button("⟳")
                                .on_hover_text("Rotate Right")
                                .clicked()
                            {
                                if self.is_video {
                                    self.video_rotation = (self.video_rotation + 90) % 360;
                                } else {
                                    self.rotate_cw(ctx);
                                }
                            }
                        });
                        if ui.button("📂 Show in Explorer").clicked() {
                            self.open_in_explorer();
                        }
                        if ui
                            .button(
                                egui::RichText::new("🗑 Delete")
                                    .color(egui::Color32::from_rgb(255, 140, 140)),
                            )
                            .on_hover_text("Move to Recycle Bin (Delete)")
                            .clicked()
                        {
                            delete_requested = true;
                        }
                    });
                if delete_requested {
                    self.delete_current_media(ctx);
                }
            }

            // Transient status message (deletes, clip markers). Drawn
            // last so it sits above the video and both control panels.
            if let Some(toast) = &self.toast {
                if toast.expired() {
                    self.toast = None;
                } else {
                    let (fill, text_color) = if toast.is_error {
                        (
                            egui::Color32::from_rgb(120, 30, 30),
                            egui::Color32::from_rgb(255, 225, 225),
                        )
                    } else {
                        (egui::Color32::from_black_alpha(225), egui::Color32::WHITE)
                    };
                    egui::Area::new(egui::Id::new("StatusToast"))
                        .anchor(egui::Align2::CENTER_TOP, [0.0, 24.0])
                        .order(egui::Order::Tooltip)
                        .interactable(false)
                        .show(ctx, |ui| {
                            egui::Frame::none()
                                .fill(fill)
                                .rounding(10.0)
                                .inner_margin(egui::Margin::symmetric(16.0, 9.0))
                                .show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new(&toast.text)
                                            .color(text_color)
                                            .size(15.0),
                                    );
                                });
                        });
                    // Toasts time out on their own, so keep frames
                    // coming until this one is gone.
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
            }
        });

        // Must run AFTER all input handling in this frame: a
        // `player.seek(...)` called from a keypress handler earlier
        // in `update()` needs the post-keypress `is_seeking()`
        // state to trigger the keep-alive repaint burst, otherwise
        // the scheduled repaint for the NEXT tick cycle is missed
        // and back-to-back seeks leave the displayed frame stale.
        if let Some(player) = &self.video_player {
            if player.is_seeking() {
                ctx.request_repaint_after(std::time::Duration::from_millis(8));
            }
        }
    }
}
