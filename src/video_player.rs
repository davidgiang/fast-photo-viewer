//! In-house video player built directly on ffmpeg-the-third and cpal.
//!
//! Replaces the egui-video dependency so we own the decode pipeline, AV
//! sync, and can later plug in hardware acceleration.
//!
//! Architecture:
//!   - `open()` spawns a decoder thread that opens the file, sets up
//!     video+audio decoders, and loops on `Packet::read`. Each decoded
//!     video frame is converted to RGBA and pushed into a bounded
//!     PTS-ordered queue. Each decoded audio frame is resampled to
//!     interleaved stereo f32 at the output device's rate and pushed
//!     into a ring buffer, behind an `AudioSpan` recording its seek
//!     generation and presentation time.
//!   - A cpal output stream consumes the audio ring buffer and sets the
//!     shared `audio_clock_us` to the presentation time of the audio it
//!     is playing — the master clock against which video frames are
//!     scheduled. Videos without audio (or without a usable output
//!     device) advance the clock via wall time.
//!   - `tick()` runs on the egui thread every frame, picks the newest
//!     queued video frame whose PTS ≤ audio_clock, drops older ones,
//!     and uploads to an egui texture.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use ffmpeg_the_third as ffmpeg;
use ffmpeg::ffi::{
    av_buffer_ref, av_buffer_unref, av_frame_unref, av_hwdevice_ctx_create,
    av_hwframe_transfer_data, av_seek_frame, avformat_flush, avio_seek, AVBufferRef,
    AVCodecContext, AVHWDeviceType, AVPixelFormat, AVSEEK_FLAG_ANY, AVSEEK_FLAG_BACKWARD,
};
use ringbuf::{
    traits::{Consumer, Observer, Producer, Split},
    HeapRb,
};

#[cfg(windows)]
#[path = "wasapi_output.rs"]
mod wasapi_output;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlayerState {
    Loading,
    Playing,
    Paused,
    EndOfFile,
    Error,
}

enum VideoFramePayload {
    /// 8-bit sRGB RGBA, stored as a pre-built ColorImage so the main
    /// thread can hand it directly to egui in the fallback path.
    Rgba8Srgb {
        width: u32,
        height: u32,
        image: egui::ColorImage,
    },
    /// 16-bit linear RGBA (u16 per channel little-endian), matching
    /// wgpu's `Rgba16Unorm` byte layout on x86. Used only on the
    /// direct-wgpu path and only when the source is 10-bit.
    Rgba16Unorm {
        width: u32,
        height: u32,
        bytes: Vec<u8>,
    },
    /// NV12: planar Y (full resolution) + interleaved UV (half
    /// resolution). Produced when HW-decoded frames arrive as NV12
    /// after `av_hwframe_transfer_data`. Skipping swscale saves a
    /// CPU YUV→RGB pass; GPU does the color conversion in a shader.
    Nv12 {
        width: u32,
        height: u32,
        y_plane: Vec<u8>,
        uv_plane: Vec<u8>,
    },
}

struct VideoFrame {
    pts_us: i64,
    payload: VideoFramePayload,
}

enum Command {
    Play,
    Pause,
    /// Absolute seek target in microseconds. Microseconds rather than a
    /// fraction so the A-B loop wrap and an explicit seek can share one
    /// code path without round-tripping through the duration.
    Seek(i64),
    SetLooping(bool),
    Stop,
}

/// Message type routed from the demuxer into the video or audio
/// decoder threads. `Flush` marks a discontinuity (seek or loop wrap)
/// — the receiving thread should flush its decoder and discard any
/// queued output.
enum DecodeMsg {
    /// Encoded packet, tagged with the demuxer's `flush_seq` at the
    /// moment it was routed. Workers compare the tag against the
    /// current `Shared::flush_seq` and discard messages whose tag
    /// is stale — this lets a mid-seek worker skip decoding the
    /// entire backlog of pre-seek packets almost instantly instead
    /// of paying one full decode per stale packet.
    Packet(ffmpeg::Packet, u64),
    /// End of stream: flush the codec so frames it is still holding
    /// internally are emitted. Frame-threaded decoders buffer several
    /// frames, and without this drain every video loses its tail.
    /// Tagged like `Packet` so a drain queued before a seek is ignored.
    Drain(u64),
}

struct SharedState {
    player_state: PlayerState,
    duration_us: i64,
    width: u32,
    height: u32,
    has_audio: bool,
    /// Average frame interval in microseconds, computed from the
    /// video stream's reported frame rate. Used by the GUI for
    /// frame-step keybinds (comma / period) and as a pacing hint.
    frame_interval_us: i64,
    error: Option<String>,
}

struct Shared {
    state: Mutex<SharedState>,
    video_queue: Mutex<VecDeque<VideoFrame>>,
    /// Master playback clock in microseconds. Driven by the audio output
    /// callback when audio is present, otherwise advanced by the decoder
    /// thread from wall time.
    audio_clock_us: AtomicI64,
    /// Volume as f32 bits (0.0..=1.0) applied by the cpal callback.
    volume_bits: AtomicU32,
    /// Set to true briefly after a seek so the audio callback can decide
    /// not to advance the clock past what's been flushed.
    clock_frozen: std::sync::atomic::AtomicBool,
    /// Timing for the audio in the output ring: one `AudioSpan` per run
    /// of pushed samples, published by the audio worker *before* the
    /// samples themselves. The output callback uses them to discard
    /// audio from superseded seek generations exactly, and to set the
    /// clock from the timestamp of the audio it is actually playing.
    audio_spans: Mutex<VecDeque<AudioSpan>>,
    /// Output-side pause. The audio stream keeps running while paused
    /// and plays silence, so stale audio from seeks made while paused
    /// is discarded promptly instead of piling up in the ring.
    audio_paused: std::sync::atomic::AtomicBool,
    /// Seek target the audio worker trims to. Seeking lands on the
    /// keyframe *before* the target; the video worker hides frames
    /// ahead of the target, and this does the same for audio so both
    /// start from the same instant.
    audio_trim_before_us: AtomicI64,
    /// No more audio is coming for the current generation: the track
    /// has ended, or it can't be decoded. The output callback then
    /// keeps the clock moving through the silence so the video plays
    /// out instead of freezing where the audio stopped.
    audio_exhausted: std::sync::atomic::AtomicBool,
    /// Raised by the demuxer on seek and cleared by the video worker
    /// when it processes the subsequent `DecodeMsg::Flush`. While
    /// raised, the worker drops any frames it decodes instead of
    /// pushing them into the shared queue — this prevents frames
    /// that were buffered in the ffmpeg decoder from before the seek
    /// from flickering through as "fast-forward" before the new
    /// position takes effect.
    flush_pending: std::sync::atomic::AtomicBool,
    /// Minimum pts (microseconds) that the video worker is allowed
    /// to push into the display queue. Set by the demuxer to the
    /// seek target so the codec's pre-target warmup frames (from the
    /// keyframe backward-seek) get decoded but not displayed.
    min_display_pts_us: AtomicI64,
    /// Incremented every time the video worker successfully pushes a
    /// post-seek frame (i.e., a frame that passed the
    /// `min_display_pts_us` gate). The demuxer uses this to know
    /// when it can stop priming packets after a seek and return to
    /// its normal paused state.
    post_seek_frames_pushed: std::sync::atomic::AtomicU32,
    /// Set on Player drop. Worker threads check this in their
    /// back-pressure wait so `decode_thread.join()` can return
    /// even when the shared video queue is full.
    stopping: std::sync::atomic::AtomicBool,
    /// Monotonic flush counter. The demuxer increments this when it
    /// initiates a seek; the workers compare it against the last
    /// value they saw and, on mismatch, call `decoder.flush()` plus
    /// clear the display queue. Replaces the in-channel `Flush`
    /// message so seek handling never blocks on a full channel.
    flush_seq: std::sync::atomic::AtomicU64,
    /// PTS of the most recent frame the video worker pushed into the
    /// display queue. At end of file this is the true last frame of
    /// the video, which the demuxer waits for the clock to reach
    /// before looping — container duration is not a reliable stand-in
    /// (it is frequently long, and can be absent entirely).
    last_pushed_pts_us: AtomicI64,
    /// Raised by the video worker once it has finished draining the
    /// codec in response to `DecodeMsg::Drain`. Until then the
    /// demuxer must not treat the display queue emptying as "the
    /// video ended" — more frames are still on their way.
    drain_complete: std::sync::atomic::AtomicBool,
    /// A-B clip loop, in microseconds. `NO_LOOP_POINT` means unset.
    /// When an out-point is set, playback wraps back to the in-point
    /// instead of running to the end of the file.
    loop_start_us: AtomicI64,
    loop_end_us: AtomicI64,
    /// The first raw (pre-volume) sample of the latest output buffer that
    /// carried audio, and when it reaches the listener. Test
    /// instrumentation: with a fixture whose samples encode their own
    /// timestamp, this says exactly what is audible at any instant,
    /// independently of the spans and anchors the player computes.
    probe_first_sample: Mutex<Option<(f32, Instant)>>,
    /// How long until the first sample of the most recent output buffer
    /// is heard, as the output backend measured it.
    output_latency_us: AtomicI64,
    /// When the audio most recently written will be heard. See
    /// `AudioAnchor`.
    audio_anchor: Mutex<Option<AudioAnchor>>,
    /// The audio output stopped for good (device gone and nothing to
    /// reopen). The player falls back to the wall clock.
    audio_output_failed: std::sync::atomic::AtomicBool,
}

/// When the audio being written will actually be heard. Published by the
/// output after each buffer, so the player can work out which moment of
/// the video is audible at any instant, not only at the last callback.
#[derive(Clone, Copy, Debug)]
struct AudioAnchor {
    /// Seek generation of the audio this describes.
    seq: u64,
    /// Presentation time of the first sample in the latest buffer.
    pts_us: i64,
    /// When that sample reaches the listener.
    heard_at: Instant,
    /// Presentation time just past the last sample handed to the device.
    /// Nothing later has been written, so the audible position stops
    /// here until more is.
    end_pts_us: i64,
}

/// `to - from` in microseconds, negative when `to` is earlier.
fn signed_micros_between(from: Instant, to: Instant) -> i64 {
    if to >= from {
        (to - from).as_micros() as i64
    } else {
        -((from - to).as_micros() as i64)
    }
}

impl Shared {
    /// The presentation time audible at `at`, from the latest anchor of
    /// the current seek generation. `None` before the first audio of a
    /// generation has been written, or once the output has failed.
    fn audible_pts_at(&self, at: Instant) -> Option<i64> {
        let anchor = {
            let guard = match self.audio_anchor.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            (*guard)?
        };
        if anchor.seq != self.flush_seq.load(Ordering::Acquire) {
            return None;
        }
        let pts = anchor.pts_us + signed_micros_between(anchor.heard_at, at);
        Some(pts.min(anchor.end_pts_us))
    }

    /// The master clock now: the audible position when audio drives
    /// playback, otherwise the stored clock.
    fn clock_now_us(&self) -> i64 {
        if !self.clock_frozen.load(Ordering::Relaxed) {
            if let Some(pts) = self.audible_pts_at(Instant::now()) {
                return pts;
            }
        }
        self.audio_clock_us.load(Ordering::Relaxed)
    }
}

/// Sentinel for "this A-B loop marker is not set". `i64::MIN` can't
/// collide with a real PTS.
pub const NO_LOOP_POINT: i64 = i64::MIN;

/// Render backend passed to `Player::open`. When `Wgpu` is supplied,
/// the player manages its own `wgpu::Texture` and uploads decoded
/// frames directly via `queue.write_texture`, bypassing egui's
/// `ColorImage` / `TextureHandle` path. When `None`, the player falls
/// back to egui's texture manager (used by the headless test binary
/// which doesn't run a real wgpu backend).
#[derive(Clone)]
pub struct WgpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub renderer: Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,
    pub target_format: wgpu::TextureFormat,
    /// True when the wgpu device was created with the
    /// `TEXTURE_FORMAT_16BIT_NORM` feature, which gates use of
    /// `Rgba16Unorm` textures for our 10-bit source path.
    pub supports_16bit_norm: bool,
}

pub struct Player {
    shared: Arc<Shared>,
    cmd_tx: Sender<Command>,
    decode_thread: Option<thread::JoinHandle<()>>,
    _audio_output: AudioOutput,
    // Egui-managed fallback path (test binary uses this).
    texture: Option<egui::TextureHandle>,
    // Direct wgpu path — preferred when available.
    wgpu: Option<WgpuBackend>,
    wgpu_texture: Option<wgpu::Texture>,
    wgpu_texture_id: Option<egui::TextureId>,
    wgpu_texture_size: (u32, u32),
    wgpu_texture_format: Option<wgpu::TextureFormat>,
    // NV12 direct-upload pipeline (partial Phase E)
    yuv_renderer: Option<YuvRenderer>,
    nv12_state: Option<Nv12GpuState>,
    // Caches the intrinsic video frame size so the GUI can lay out
    // the video area correctly even when the current GPU texture is
    // a planar YUV pair without an associated `TextureId`.
    intrinsic_size: (u32, u32),
    /// When set, the most recent uploaded frame went through the
    /// NV12 path — the main thread should render via
    /// `draw_into_painter` instead of `painter.image(texture_id())`.
    last_upload_was_nv12: bool,
    egui_ctx: egui::Context,
    last_uploaded_pts_us: i64,
    /// Wall-clock fallback: when the file has no audio, we advance the
    /// shared clock ourselves based on elapsed real time since playback
    /// started.
    no_audio_clock_origin: Option<Instant>,
    no_audio_clock_base_us: i64,
    playing_cached: bool,
    /// True when an audio output stream is running and drives the
    /// clock. False for silent videos, and also when the file has audio
    /// but no output device could be opened — timing then falls back
    /// to the wall clock instead of waiting on audio that never plays.
    clock_from_audio: bool,
    /// Set when a frame step moved the picture without moving the
    /// audio. The next `play()` re-seeks to the frame on screen so the
    /// two resume from the same point.
    resync_on_play: bool,
    /// How far ahead of now the frame picked in `tick()` will actually be
    /// on screen. Frames are chosen for that moment, not for now.
    display_lead: Duration,
}

/// Keeps the audio output running for the player's lifetime.
enum AudioOutput {
    None,
    Cpal(cpal::Stream),
    // Held only for its Drop, which stops the output thread.
    #[cfg(windows)]
    #[allow(dead_code)]
    Wasapi(wasapi_output::WasapiOutput),
}

#[allow(dead_code)]
impl Player {
    pub fn open(ctx: &egui::Context, path: &Path) -> Result<Self, String> {
        Self::open_with_backend(ctx, path, None)
    }

    pub fn open_with_backend(
        ctx: &egui::Context,
        path: &Path,
        wgpu: Option<WgpuBackend>,
    ) -> Result<Self, String> {
        // Query cpal for the output device's native rate/channels up
        // front. We resample the decoded audio directly to that rate so
        // the playback stream doesn't need any further conversion.
        let host = cpal::default_host();
        let (target_rate, target_channels) = match host.default_output_device() {
            Some(dev) => match dev.default_output_config() {
                Ok(cfg) => (cfg.sample_rate().0, cfg.channels() as usize),
                Err(_) => (48_000u32, 2usize),
            },
            None => (48_000u32, 2usize),
        };

        let shared = Arc::new(Shared {
            state: Mutex::new(SharedState {
                player_state: PlayerState::Loading,
                duration_us: 0,
                width: 0,
                height: 0,
                has_audio: false,
                frame_interval_us: 33_333,
                error: None,
            }),
            video_queue: Mutex::new(VecDeque::new()),
            audio_clock_us: AtomicI64::new(0),
            volume_bits: AtomicU32::new(1.0f32.to_bits()),
            clock_frozen: std::sync::atomic::AtomicBool::new(false),
            audio_spans: Mutex::new(VecDeque::with_capacity(256)),
            audio_paused: std::sync::atomic::AtomicBool::new(true),
            audio_trim_before_us: AtomicI64::new(i64::MIN),
            audio_exhausted: std::sync::atomic::AtomicBool::new(false),
            flush_pending: std::sync::atomic::AtomicBool::new(false),
            min_display_pts_us: AtomicI64::new(i64::MIN),
            post_seek_frames_pushed: std::sync::atomic::AtomicU32::new(0),
            stopping: std::sync::atomic::AtomicBool::new(false),
            flush_seq: std::sync::atomic::AtomicU64::new(0),
            last_pushed_pts_us: AtomicI64::new(0),
            drain_complete: std::sync::atomic::AtomicBool::new(false),
            loop_start_us: AtomicI64::new(NO_LOOP_POINT),
            loop_end_us: AtomicI64::new(NO_LOOP_POINT),
            probe_first_sample: Mutex::new(None),
            output_latency_us: AtomicI64::new(0),
            audio_anchor: Mutex::new(None),
            audio_output_failed: std::sync::atomic::AtomicBool::new(false),
        });

        // Audio ring buffer: 4 seconds worth of stereo f32 at the
        // output device's native rate. Generous headroom so a brief
        // scheduling hiccup doesn't drain it.
        let rb = HeapRb::<f32>::new(target_rate as usize * 2 * 4);
        let (audio_producer, audio_consumer) = rb.split();

        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>();

        let high_bit_depth_enabled = wgpu
            .as_ref()
            .map(|b| b.supports_16bit_norm)
            .unwrap_or(false);
        let shared_for_thread = shared.clone();
        let ctx_for_thread = ctx.clone();
        let path_for_thread = path.to_path_buf();
        let decode_thread = thread::spawn(move || {
            if let Err(e) = run_decode_pipeline(
                path_for_thread,
                cmd_rx,
                shared_for_thread.clone(),
                audio_producer,
                ctx_for_thread,
                target_rate,
                high_bit_depth_enabled,
            ) {
                let mut state = shared_for_thread.state.lock().unwrap();
                state.player_state = PlayerState::Error;
                state.error = Some(e);
            }
        });

        // Wait up to 5 s for the decoder to finish init (duration/size
        // populated and state transitioned out of Loading).
        let start = Instant::now();
        loop {
            {
                let state = shared.state.lock().unwrap();
                if state.player_state != PlayerState::Loading {
                    if let Some(e) = &state.error {
                        return Err(e.clone());
                    }
                    break;
                }
            }
            if start.elapsed() > Duration::from_secs(5) {
                return Err("timeout waiting for decoder initialization".into());
            }
            thread::sleep(Duration::from_millis(5));
        }

        let has_audio = shared.state.lock().unwrap().has_audio;
        let _ = target_channels;
        let audio_output = if has_audio {
            start_audio_output(audio_consumer, &shared, target_rate)
        } else {
            AudioOutput::None
        };
        let clock_from_audio = !matches!(audio_output, AudioOutput::None);

        Ok(Self {
            shared,
            cmd_tx,
            decode_thread: Some(decode_thread),
            _audio_output: audio_output,
            texture: None,
            yuv_renderer: wgpu
                .as_ref()
                .map(|b| YuvRenderer::new(&b.device, b.target_format)),
            nv12_state: None,
            intrinsic_size: (0, 0),
            last_upload_was_nv12: false,
            wgpu,
            wgpu_texture: None,
            wgpu_texture_id: None,
            wgpu_texture_size: (0, 0),
            wgpu_texture_format: None,
            egui_ctx: ctx.clone(),
            last_uploaded_pts_us: i64::MIN,
            no_audio_clock_origin: None,
            no_audio_clock_base_us: 0,
            playing_cached: false,
            clock_from_audio,
            resync_on_play: false,
            display_lead: Duration::ZERO,
        })
    }

    pub fn play(&mut self) {
        // Frame steps move the picture but leave the audio where it was
        // paused. Resuming from there would hold the picture still until
        // the audio caught up (or, stepping back, jump it forward), so
        // re-seek to the frame on screen and let both start from it.
        if self.clock_from_audio && self.resync_on_play && self.last_uploaded_pts_us != i64::MIN {
            let pts_us = self.last_uploaded_pts_us;
            self.seek_us(pts_us);
        }
        self.resync_on_play = false;
        let _ = self.cmd_tx.send(Command::Play);
        self.playing_cached = true;
        self.shared.audio_paused.store(false, Ordering::Relaxed);
        if !self.clock_from_audio {
            self.no_audio_clock_origin = Some(Instant::now());
            self.no_audio_clock_base_us = self.shared.audio_clock_us.load(Ordering::Relaxed);
        }
    }

    pub fn pause(&mut self) {
        let _ = self.cmd_tx.send(Command::Pause);
        self.playing_cached = false;
        self.shared.audio_paused.store(true, Ordering::Relaxed);
        if !self.clock_from_audio {
            // Freeze the clock at its current value.
            if let Some(origin) = self.no_audio_clock_origin.take() {
                let now_us = self.no_audio_clock_base_us
                    + origin.elapsed().as_micros() as i64;
                self.shared.audio_clock_us.store(now_us, Ordering::Relaxed);
            }
        }
    }

    /// Seek to `fraction` of the total duration (0.0 ..= 1.0).
    pub fn seek(&mut self, fraction: f32) {
        let duration_us = self.shared.state.lock().unwrap().duration_us.max(0);
        let target_us = (duration_us as f64 * fraction.clamp(0.0, 1.0) as f64) as i64;
        self.seek_us(target_us);
    }

    /// Seek to an absolute position in microseconds.
    pub fn seek_us(&mut self, target_us: i64) {
        let duration_us = self.shared.state.lock().unwrap().duration_us.max(0);
        let target_us = if duration_us > 0 {
            target_us.clamp(0, duration_us)
        } else {
            target_us.max(0)
        };
        // Lock the display pipeline synchronously before the demuxer
        // even sees the command: clear the queue so any pre-seek
        // frames still buffered there vanish, set `min_display_pts_us`
        // to MAX so tick() can't accidentally display a stale frame
        // while the demuxer is still processing, and raise
        // `clock_frozen` so the priming path is the only one that
        // can surface the next rendered frame. The demuxer's seek
        // handler will overwrite `min_display_pts_us` with the real
        // target shortly afterwards.
        self.shared
            .min_display_pts_us
            .store(i64::MAX, Ordering::Relaxed);
        self.shared.video_queue.lock().unwrap().clear();
        self.shared
            .clock_frozen
            .store(true, Ordering::Relaxed);
        let _ = self.cmd_tx.send(Command::Seek(target_us));
        // Briefly wait for the demuxer to actually pick up the new
        // seek target (it overwrites `min_display_pts_us` from MAX
        // to the real target inside its seek handler). Without this
        // short handoff, subsequent tick() calls on the main thread
        // can race with the demuxer and see stale state, which has
        // been observed as the GUI not updating after back-to-back
        // seeks / frame-step keypresses.
        let deadline = Instant::now() + Duration::from_millis(30);
        while Instant::now() < deadline {
            if self
                .shared
                .min_display_pts_us
                .load(Ordering::Relaxed)
                != i64::MAX
            {
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        // Make sure egui schedules another update cycle so that
        // when the decoder worker pushes the post-seek frame shortly
        // afterwards, tick() actually runs and surfaces it.
        self.egui_ctx.request_repaint();
        // A seek realigns audio and video by itself.
        self.resync_on_play = false;
        if !self.clock_from_audio {
            self.shared
                .audio_clock_us
                .store(target_us, Ordering::Relaxed);
            // Leave the wall-clock origin unset: while `clock_frozen`
            // is raised, `tick()` must not advance the clock at all,
            // and it re-anchors both origin and base to the first
            // post-seek frame's PTS once one arrives.
            self.no_audio_clock_origin = None;
            self.no_audio_clock_base_us = target_us;
        }
        // clock_frozen is cleared by the decoder thread after the seek.
    }

    /// Set the A-B clip loop. Passing `None` for either end clears
    /// that marker. When an out-point is set, playback wraps back to
    /// the in-point (or the start of the file if none is set) instead
    /// of running through to the end.
    pub fn set_loop_range(&mut self, start_us: Option<i64>, end_us: Option<i64>) {
        self.shared
            .loop_start_us
            .store(start_us.unwrap_or(NO_LOOP_POINT), Ordering::Relaxed);
        self.shared
            .loop_end_us
            .store(end_us.unwrap_or(NO_LOOP_POINT), Ordering::Relaxed);
    }

    /// Current A-B clip loop markers in microseconds, if set.
    pub fn loop_range(&self) -> (Option<i64>, Option<i64>) {
        let to_opt = |v: i64| if v == NO_LOOP_POINT { None } else { Some(v) };
        (
            to_opt(self.shared.loop_start_us.load(Ordering::Relaxed)),
            to_opt(self.shared.loop_end_us.load(Ordering::Relaxed)),
        )
    }

    pub fn set_volume(&mut self, v: f32) {
        self.shared
            .volume_bits
            .store(v.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    pub fn set_looping(&mut self, looping: bool) {
        let _ = self.cmd_tx.send(Command::SetLooping(looping));
    }

    pub fn state(&self) -> PlayerState {
        self.shared.state.lock().unwrap().player_state
    }

    pub fn duration_ms(&self) -> i64 {
        self.shared.state.lock().unwrap().duration_us / 1000
    }

    /// Tell the player how long a frame chosen now takes to reach the
    /// screen (about one display refresh), so it picks frames for the
    /// moment they'll actually be seen.
    pub fn set_display_lead(&mut self, lead: Duration) {
        self.display_lead = lead.min(Duration::from_millis(100));
    }

    /// Where playback is: the frame on screen while paused (audio is
    /// buffered ahead of it), otherwise the clock. In milliseconds,
    /// clamped to the duration — the no-audio wall clock keeps running
    /// once the last frame has been shown, and would otherwise report a
    /// position past the end.
    pub fn elapsed_ms(&self) -> i64 {
        let state = self.shared.state.lock().unwrap();
        let duration_us = state.duration_us;
        drop(state);
        let clock_us = self.position_us().max(0);
        if duration_us > 0 {
            clock_us.min(duration_us) / 1000
        } else {
            clock_us / 1000
        }
    }

    /// Average frame interval in microseconds, derived from the
    /// source video's `avg_frame_rate`. Used by the GUI to implement
    /// frame-step seeking (comma / period keys).
    pub fn frame_interval_us(&self) -> i64 {
        self.shared.state.lock().unwrap().frame_interval_us.max(1_000)
    }

    /// Step by `delta` frames. Forward steps preferentially consume
    /// frames the worker has already decoded and buffered past the
    /// current display position — this avoids the cost of calling
    /// `av_seek_frame` back to the enclosing keyframe and
    /// re-decoding ~a GOP's worth of content, which for long-GOP
    /// HEVC can take hundreds of ms per step and feel frozen.
    /// Backward steps always fall back to a real seek.
    pub fn step_frames(&mut self, delta: i32) {
        let duration_us = self.shared.state.lock().unwrap().duration_us;
        if duration_us <= 0 {
            return;
        }

        if delta > 0 {
            // Try to advance through the already-decoded queue
            // first. If there's a buffered frame immediately after
            // the currently displayed one, just show it. No seek,
            // no decode, no stall.
            let mut queue = self.shared.video_queue.lock().unwrap();
            let cutoff = self.last_uploaded_pts_us;
            // Drop anything at/before the current display pts so
            // the next pop yields the following frame.
            while let Some(front) = queue.front() {
                if front.pts_us <= cutoff {
                    queue.pop_front();
                } else {
                    break;
                }
            }
            if let Some(frame) = queue.pop_front() {
                drop(queue);
                self.display_frame(frame);
                return;
            }
            // Queue didn't have anything buffered — fall through
            // to a real seek below.
            drop(queue);
        }

        let interval = self.frame_interval_us();
        let current_us = self.position_us();
        if delta < 0 {
            // Backward step: "first frame ≥ target" semantics can
            // round back to `current` when the `target_us =
            // current - interval` value falls between actual
            // frames (variable frame rate / avg_frame_rate
            // mismatch), producing a visible oscillation. Seek to
            // a point safely several frames back, then pick the
            // LATEST decoded frame strictly before `current_us`.
            let back = (interval.max(16_000)) * (-delta as i64) * 4;
            let target_us = (current_us - back).max(0);
            let frac = (target_us as f32 / duration_us as f32).clamp(0.0, 1.0);
            self.seek(frac);
            // Wait briefly for the worker to fill the queue with
            // post-seek frames, then pop the latest frame whose
            // pts is strictly less than `current_us`. The 800 ms
            // deadline covers long-GOP HW decode from a keyframe
            // to the seek target.
            let cutoff = current_us;
            let deadline = Instant::now() + Duration::from_millis(800);
            while Instant::now() < deadline {
                let mut queue = self.shared.video_queue.lock().unwrap();
                let mut latest_back: Option<VideoFrame> = None;
                while let Some(front) = queue.front() {
                    if front.pts_us < cutoff {
                        latest_back = queue.pop_front();
                    } else {
                        break;
                    }
                }
                if let Some(frame) = latest_back {
                    drop(queue);
                    self.display_frame(frame);
                    return;
                }
                drop(queue);
                thread::sleep(Duration::from_millis(8));
            }
            return;
        }

        // `clamp` panics when min > max, which a video shorter than one
        // frame interval (a single-frame clip, a stub file) would hit.
        let target_us =
            (current_us + interval * delta as i64).clamp(0, (duration_us - interval).max(0));
        let frac = (target_us as f32 / duration_us as f32).clamp(0.0, 1.0);
        self.seek(frac);
    }

    /// Upload `frame` to the active texture, update the bookkeeping
    /// atoms so subsequent ticks/seeks see a consistent post-step
    /// state, and request a repaint. Shared between the forward- and
    /// backward-step paths.
    fn display_frame(&mut self, frame: VideoFrame) {
        let (w, h) = match &frame.payload {
            VideoFramePayload::Rgba8Srgb { width, height, .. }
            | VideoFramePayload::Rgba16Unorm { width, height, .. }
            | VideoFramePayload::Nv12 { width, height, .. } => (*width, *height),
        };
        self.intrinsic_size = (w, h);
        let new_pts = frame.pts_us;
        if self.wgpu.is_some() {
            self.upload_wgpu(frame.payload);
        } else if let VideoFramePayload::Rgba8Srgb { image, .. } = frame.payload {
            self.upload_egui(image);
            self.last_upload_was_nv12 = false;
        }
        self.last_uploaded_pts_us = new_pts;
        self.shared
            .audio_clock_us
            .store(new_pts, Ordering::Relaxed);
        self.shared
            .clock_frozen
            .store(false, Ordering::Relaxed);
        self.resync_on_play = true;
        self.egui_ctx.request_repaint();
    }

    pub fn size(&self) -> (u32, u32) {
        let s = self.shared.state.lock().unwrap();
        (s.width, s.height)
    }

    pub fn has_audio(&self) -> bool {
        self.shared.state.lock().unwrap().has_audio
    }

    /// True while the player is in post-seek priming, i.e. the
    /// display clock is frozen waiting for the first post-seek frame
    /// to land. The GUI uses this to keep requesting repaints while
    /// paused so the target frame actually renders.
    pub fn is_seeking(&self) -> bool {
        self.shared
            .clock_frozen
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn texture(&self) -> Option<&egui::TextureHandle> {
        self.texture.as_ref()
    }

    /// Returns the egui `TextureId` that should be drawn for the
    /// current video frame, regardless of whether it's backed by an
    /// egui `TextureHandle` or a directly-managed `wgpu::Texture`.
    pub fn texture_id(&self) -> Option<egui::TextureId> {
        if let Some(id) = self.wgpu_texture_id {
            return Some(id);
        }
        self.texture.as_ref().map(|t| t.id())
    }

    pub fn error(&self) -> Option<String> {
        self.shared.state.lock().unwrap().error.clone()
    }

    /// Test hook: last video frame PTS uploaded to the texture. Used
    /// by the `video_test` binary to detect progression without having
    /// to inspect the texture handle.
    pub fn uploaded_pts_us_for_test(&self) -> i64 {
        self.last_uploaded_pts_us
    }

    /// Which audio output is driving playback: "wasapi", "cpal", or
    /// "none" (silent video, or no usable device).
    pub fn audio_backend(&self) -> &'static str {
        match self._audio_output {
            AudioOutput::None => "none",
            AudioOutput::Cpal(_) => "cpal",
            #[cfg(windows)]
            AudioOutput::Wasapi(_) => "wasapi",
        }
    }

    /// Test hook: behave as if the audio device had just disappeared
    /// for good.
    pub fn simulate_audio_output_loss_for_test(&self) {
        self.shared.audio_output_failed.store(true, Ordering::Relaxed);
        if let Ok(mut anchor) = self.shared.audio_anchor.lock() {
            *anchor = None;
        }
    }

    /// Output latency the audio backend reports, in microseconds: how
    /// long until the first sample of the latest buffer is heard.
    pub fn audio_output_latency_us(&self) -> i64 {
        self.shared.output_latency_us.load(Ordering::Relaxed)
    }

    /// Test hook: the first raw sample of the latest output buffer and
    /// when it reaches the listener, or `None` before any audio has
    /// played. Paired with a fixture whose audio encodes its own
    /// timestamp, this gives the audible position at any instant.
    pub fn first_played_sample_for_test(&self) -> Option<(f32, Instant)> {
        match self.shared.probe_first_sample.lock() {
            Ok(g) => *g,
            Err(poisoned) => *poisoned.into_inner(),
        }
    }

    /// Called from the egui update loop every frame. Advances the
    /// no-audio clock, picks the newest video frame whose PTS ≤ the
    /// current clock, and uploads it to a texture.
    pub fn tick(&mut self) {
        let clock_frozen = self
            .shared
            .clock_frozen
            .load(Ordering::Relaxed);

        // No-audio wall-clock advance.
        //
        // Skipped while the clock is frozen. A frozen clock means the
        // demuxer has repositioned (a seek, or an end-of-file loop
        // wrap) and pinned the clock at the new target; advancing it
        // from a wall-clock origin captured before the reposition
        // would immediately drag it back to the old position. That is
        // what made looping silent videos run away at the end: the
        // wrap reset the clock to 0, the very next tick overwrote it
        // with "time since play started" — still out at the end of
        // the file — and every freshly decoded frame was then treated
        // as already overdue.
        if !clock_frozen && !self.clock_from_audio && self.playing_cached {
            if let Some(origin) = self.no_audio_clock_origin {
                let now_us =
                    self.no_audio_clock_base_us + origin.elapsed().as_micros() as i64;
                self.shared.audio_clock_us.store(now_us, Ordering::Relaxed);
            }
        }

        // The audio output stopped for good: carry on from the wall clock
        // rather than waiting forever on audio that will never play.
        if self.clock_from_audio && self.shared.audio_output_failed.load(Ordering::Relaxed) {
            eprintln!("video_player: audio output lost; timing playback from the wall clock");
            self.clock_from_audio = false;
            self.no_audio_clock_base_us = self.current_clock_us();
            self.no_audio_clock_origin = self.playing_cached.then(Instant::now);
        }

        // Pick the frame for the moment it will be on screen. With audio
        // driving playback, that is the audio audible at that moment —
        // interpolated from the output's device-clock anchor, so it is
        // both smooth between callbacks and compensated for output
        // latency. Without an anchor yet (just after a seek) or without
        // audio, fall back to the stored clock.
        let lead_us = self.display_lead.as_micros() as i64;
        let clock_us = if clock_frozen {
            self.current_clock_us()
        } else if self.clock_from_audio {
            self.shared
                .audible_pts_at(Instant::now() + self.display_lead)
                .unwrap_or_else(|| self.current_clock_us())
        } else {
            self.current_clock_us() + lead_us
        };

        let min_pts = self
            .shared
            .min_display_pts_us
            .load(Ordering::Relaxed);

        // While paused (and not in post-seek warmup), the display
        // has already been chosen by the last seek / step_frames
        // call. Don't drain the demuxer-filled lookahead queue
        // here: that lookahead exists so the next step-forward
        // can reuse it, and popping it eagerly would "scroll
        // forward" through buffered frames — which on high-fps
        // content makes step_frames(-1) appear to jump forward
        // because tick()'s ±15 ms tolerance sweeps up the next
        // frame every time the clock unfreezes.
        if !self.playing_cached && !clock_frozen {
            return;
        }
        let best = {
            let mut queue = self.shared.video_queue.lock().unwrap();
            if clock_frozen {
                // Post-seek warmup: the clock is intentionally frozen
                // at the seek target until the first decoded frame
                // arrives. Pop the first frame at or past the seek
                // target, discarding any stale frames that leaked
                // through the flush-race window.
                let mut chosen: Option<VideoFrame> = None;
                while let Some(front) = queue.front() {
                    if front.pts_us >= min_pts {
                        chosen = queue.pop_front();
                        break;
                    } else {
                        let _ = queue.pop_front();
                    }
                }
                chosen
            } else {
                let loop_end = self.shared.loop_end_us.load(Ordering::Relaxed);
                let mut best: Option<VideoFrame> = None;
                while let Some(front) = queue.front() {
                    // Also honour min_pts here in case a stale frame
                    // leaked through after a seek but before the
                    // worker cleared the queue on its Flush handler.
                    if front.pts_us < min_pts {
                        let _ = queue.pop_front();
                        continue;
                    }
                    // With an A-B out-point set, stop at the last
                    // in-range frame and hold it. The demuxer notices
                    // the clock passing the out-point within a few ms
                    // and wraps; without this gate the 15 ms display
                    // tolerance would flash a frame past the marker
                    // first.
                    if loop_end != NO_LOOP_POINT && front.pts_us > loop_end {
                        break;
                    }
                    // A millisecond of slack absorbs container timestamp
                    // rounding (Matroska stores whole milliseconds).
                    if front.pts_us <= clock_us + 1_000 {
                        best = queue.pop_front();
                    } else {
                        break;
                    }
                }
                best
            }
        };

        if let Some(frame) = best {
            if frame.pts_us != self.last_uploaded_pts_us {
                // If the clock is frozen (post-seek warmup), unfreeze
                // it and re-anchor to this frame's pts so playback
                // resumes from exactly this frame instead of jumping
                // forward by the decoder-warmup latency.
                if self
                    .shared
                    .clock_frozen
                    .load(Ordering::Relaxed)
                {
                    self.shared
                        .audio_clock_us
                        .store(frame.pts_us, Ordering::Relaxed);
                    self.shared
                        .clock_frozen
                        .store(false, Ordering::Relaxed);
                    // For the no-audio wall-clock path, also reset
                    // the wall-clock origin so it advances from here.
                    if !self.clock_from_audio && self.playing_cached {
                        self.no_audio_clock_origin = Some(Instant::now());
                        self.no_audio_clock_base_us = frame.pts_us;
                    }
                }
                let (w, h) = match &frame.payload {
                    VideoFramePayload::Rgba8Srgb { width, height, .. }
                    | VideoFramePayload::Rgba16Unorm { width, height, .. }
                    | VideoFramePayload::Nv12 { width, height, .. } => (*width, *height),
                };
                self.intrinsic_size = (w, h);
                if self.wgpu.is_some() {
                    self.upload_wgpu(frame.payload);
                } else {
                    match frame.payload {
                        VideoFramePayload::Rgba8Srgb { image, .. } => {
                            self.upload_egui(image);
                            self.last_upload_was_nv12 = false;
                        }
                        VideoFramePayload::Rgba16Unorm { .. }
                        | VideoFramePayload::Nv12 { .. } => {
                            // These variants never reach us without a
                            // wgpu backend — the decode thread gates
                            // on features that require one.
                        }
                    }
                }
                self.last_uploaded_pts_us = frame.pts_us;
            }
        }
    }

    /// True when the most recent frame was uploaded via the NV12
    /// direct path; the renderer must draw it with a custom
    /// PaintCallback because egui's image path can't sample planar
    /// YUV textures.
    pub fn is_nv12_path(&self) -> bool {
        self.last_upload_was_nv12
    }

    pub fn intrinsic_size(&self) -> (u32, u32) {
        self.intrinsic_size
    }

    fn upload_egui(&mut self, image: egui::ColorImage) {
        let same_size = self
            .texture
            .as_ref()
            .map(|t| t.size() == image.size)
            .unwrap_or(false);
        if same_size {
            if let Some(t) = self.texture.as_mut() {
                t.set(image, egui::TextureOptions::LINEAR);
            }
        } else {
            self.texture = Some(self.egui_ctx.load_texture(
                "video_frame",
                image,
                egui::TextureOptions::LINEAR,
            ));
        }
    }

    fn upload_wgpu(&mut self, payload: VideoFramePayload) {
        let backend = match self.wgpu.as_ref() {
            Some(b) => b.clone(),
            None => return,
        };

        // NV12 path is entirely separate: two textures, bind group,
        // custom render pipeline.
        if let VideoFramePayload::Nv12 {
            width,
            height,
            y_plane,
            uv_plane,
        } = payload
        {
            self.upload_wgpu_nv12(&backend, width, height, &y_plane, &uv_plane);
            self.last_upload_was_nv12 = true;
            return;
        }
        self.last_upload_was_nv12 = false;
        self.nv12_state = None;

        let (width, height, wgpu_format, bytes_per_pixel): (u32, u32, wgpu::TextureFormat, u32) =
            match &payload {
                VideoFramePayload::Rgba8Srgb { width, height, .. } => (
                    *width,
                    *height,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    4,
                ),
                VideoFramePayload::Rgba16Unorm { width, height, .. } => (
                    *width,
                    *height,
                    wgpu::TextureFormat::Rgba16Unorm,
                    8,
                ),
                VideoFramePayload::Nv12 { .. } => unreachable!("handled above"),
            };

        // (Re)create the wgpu texture on the first frame, when the
        // aspect ratio changes, or when the pixel format changes
        // (e.g. switching from an 8-bit to a 10-bit video).
        let needs_new_texture = self.wgpu_texture.is_none()
            || self.wgpu_texture_size != (width, height)
            || self.wgpu_texture_format != Some(wgpu_format);
        if needs_new_texture {
            let tex = backend.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("video_player_frame"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu_format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

            let mut renderer = backend.renderer.write();
            let new_id = if let Some(existing_id) = self.wgpu_texture_id {
                renderer.update_egui_texture_from_wgpu_texture(
                    &backend.device,
                    &view,
                    wgpu::FilterMode::Linear,
                    existing_id,
                );
                existing_id
            } else {
                renderer.register_native_texture(
                    &backend.device,
                    &view,
                    wgpu::FilterMode::Linear,
                )
            };
            drop(renderer);
            self.wgpu_texture = Some(tex);
            self.wgpu_texture_id = Some(new_id);
            self.wgpu_texture_size = (width, height);
            self.wgpu_texture_format = Some(wgpu_format);
        }

        // Reinterpret the frame payload as a flat byte slice and
        // write it straight into the wgpu texture.
        let bytes: &[u8] = match &payload {
            VideoFramePayload::Rgba8Srgb { image, .. } => unsafe {
                std::slice::from_raw_parts(
                    image.pixels.as_ptr() as *const u8,
                    image.pixels.len() * 4,
                )
            },
            VideoFramePayload::Rgba16Unorm { bytes, .. } => bytes.as_slice(),
            VideoFramePayload::Nv12 { .. } => unreachable!("handled above"),
        };
        if let Some(tex) = self.wgpu_texture.as_ref() {
            backend.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytes,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(width * bytes_per_pixel),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    fn current_clock_us(&self) -> i64 {
        self.shared.audio_clock_us.load(Ordering::Relaxed)
    }

    /// Current playback position: the frame on screen while paused,
    /// otherwise the clock.
    fn position_us(&self) -> i64 {
        if !self.playing_cached && self.last_uploaded_pts_us != i64::MIN {
            self.last_uploaded_pts_us
        } else {
            self.shared.clock_now_us()
        }
    }

    fn upload_wgpu_nv12(
        &mut self,
        backend: &WgpuBackend,
        width: u32,
        height: u32,
        y_plane: &[u8],
        uv_plane: &[u8],
    ) {
        let renderer = match self.yuv_renderer.as_ref() {
            Some(r) => r.clone(),
            None => return,
        };
        let uv_h = (height + 1) / 2;

        // (Re)create state when the size changes.
        let needs_new = match &self.nv12_state {
            Some(s) => s.width != width || s.height != height,
            None => true,
        };
        if needs_new {
            let y_texture = backend.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("nv12_y_texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let uv_texture = backend.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("nv12_uv_texture"),
                size: wgpu::Extent3d {
                    width: width / 2,
                    height: uv_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let y_view = y_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let uv_view = uv_texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Uniform buffer holding the NDC rect (left, top, right,
            // bottom) for the vertex shader. Initialised to a
            // full-NDC quad so the first frame draws even before
            // `set_nv12_ndc_rect` is called.
            let ndc_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("nv12_ndc_uniform"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let initial_ndc: [f32; 4] = [-1.0, 1.0, 1.0, -1.0];
            let initial_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    initial_ndc.as_ptr() as *const u8,
                    std::mem::size_of::<[f32; 4]>(),
                )
            };
            backend.queue.write_buffer(&ndc_buffer, 0, initial_bytes);

            let bind_group = Arc::new(backend.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("nv12_bind_group"),
                layout: &renderer.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&y_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&uv_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&renderer.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ndc_buffer.as_entire_binding(),
                    },
                ],
            }));

            self.nv12_state = Some(Nv12GpuState {
                y_texture,
                uv_texture,
                ndc_buffer,
                bind_group,
                width,
                height,
            });
        }
        let state = match self.nv12_state.as_ref() {
            Some(s) => s,
            None => return,
        };

        // Upload Y plane.
        backend.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &state.y_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            y_plane,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        // Upload UV plane.
        backend.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &state.uv_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            uv_plane,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(width), // Rg8 × (width/2) = width bytes
                rows_per_image: Some(uv_h),
            },
            wgpu::Extent3d {
                width: width / 2,
                height: uv_h,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Emit an `egui_wgpu::Callback` that draws the current NV12
    /// video into `video_rect`. egui_wgpu sets the wgpu viewport to
    /// this rect (in pixels) before the callback runs, so our shader
    /// just emits a full NDC quad.
    pub fn nv12_paint_callback(
        &self,
        video_rect: egui::Rect,
    ) -> Option<egui::epaint::PaintCallback> {
        let renderer = self.yuv_renderer.as_ref()?;
        let state = self.nv12_state.as_ref()?;
        let cb = Nv12PaintCallback {
            pipeline: renderer.pipeline.clone(),
            bind_group: state.bind_group.clone(),
        };
        Some(egui_wgpu::Callback::new_paint_callback(video_rect, cb))
    }

    /// Update the NV12 vertex shader's quad rect for the next frame.
    /// Caller passes the unclamped video rect (in egui points) and the
    /// caller's own viewport metrics. We compute the equivalent
    /// quad-in-NDC inside the egui_wgpu-clamped viewport so the video
    /// translates correctly when panning pushes the rect off-screen
    /// (the rasterizer clips the over-extended quad via the scissor).
    pub fn set_nv12_render_rect(
        &self,
        video_rect: egui::Rect,
        screen_size_points: egui::Vec2,
        pixels_per_point: f32,
    ) {
        let backend = match self.wgpu.as_ref() {
            Some(b) => b,
            None => return,
        };
        let state = match self.nv12_state.as_ref() {
            Some(s) => s,
            None => return,
        };
        let scr_w = screen_size_points.x * pixels_per_point;
        let scr_h = screen_size_points.y * pixels_per_point;
        let rect_l = video_rect.min.x * pixels_per_point;
        let rect_t = video_rect.min.y * pixels_per_point;
        let rect_r = video_rect.max.x * pixels_per_point;
        let rect_b = video_rect.max.y * pixels_per_point;
        // Replicate epaint::ViewportInPixels::from_points clamping.
        let vp_l = rect_l.max(0.0).min(scr_w);
        let vp_t = rect_t.max(0.0).min(scr_h);
        let vp_r = rect_r.max(vp_l).min(scr_w);
        let vp_b = rect_b.max(vp_t).min(scr_h);
        let vp_w = (vp_r - vp_l).max(1.0);
        let vp_h = (vp_b - vp_t).max(1.0);
        // NDC X: -1 at viewport left, +1 at viewport right.
        let ndc_l = (rect_l - vp_l) / vp_w * 2.0 - 1.0;
        let ndc_r = (rect_r - vp_l) / vp_w * 2.0 - 1.0;
        // NDC Y: +1 at top, -1 at bottom (wgpu).
        let ndc_t = 1.0 - (rect_t - vp_t) / vp_h * 2.0;
        let ndc_b = 1.0 - (rect_b - vp_t) / vp_h * 2.0;
        let data: [f32; 4] = [ndc_l, ndc_t, ndc_r, ndc_b];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                std::mem::size_of::<[f32; 4]>(),
            )
        };
        backend.queue.write_buffer(&state.ndc_buffer, 0, bytes);
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.shared.stopping.store(true, Ordering::Relaxed);
        let _ = self.cmd_tx.send(Command::Stop);
        if let AudioOutput::Cpal(stream) = &self._audio_output {
            let _ = stream.pause();
        }
        // Wait briefly for the decode thread to exit cleanly, but
        // don't block forever: if it's wedged (e.g. in ffmpeg I/O),
        // we'd rather leak the thread than freeze the GUI shutdown.
        // The OS will clean up when the process exits.
        if let Some(h) = self.decode_thread.take() {
            let deadline = Instant::now() + Duration::from_millis(300);
            while Instant::now() < deadline {
                if h.is_finished() {
                    let _ = h.join();
                    return;
                }
                thread::sleep(Duration::from_millis(5));
            }
            // Timed out — detach. Handle drop leaves the thread
            // running, but since Player::drop is only called when
            // the owning window is closing, that's fine.
            std::mem::drop(h);
        }
    }
}

// =========================================================================
// Audio output (cpal)
// =========================================================================

type AudioConsumer = <HeapRb<f32> as Split>::Cons;

/// Copy samples from `pop` into `data` as interleaved PCM for
/// `channels` output channels, scaled by `volume`. Returns the number
/// of audio frames whose samples were actually popped — NOT the total
/// number of output frames written.
///
/// The distinction is load-bearing for A/V sync. The audio callback
/// only moves the master `audio_clock_us` when real samples were
/// played; if a ring underrun occurs mid-buffer, the remainder of
/// `data` is zero-filled and the returned count stays at the
/// pre-underrun value. Treating silence-filled frames as played would
/// let the clock run ahead of the audio actually heard.
fn mix_audio_output<F: FnMut() -> Option<f32>>(
    data: &mut [f32],
    channels: usize,
    volume: f32,
    mut pop: F,
) -> usize {
    let frames = data.len() / channels.max(1);
    let mut played = 0usize;
    for frame in 0..frames {
        let Some(l) = pop() else {
            for f in frame..frames {
                for c in 0..channels {
                    data[f * channels + c] = 0.0;
                }
            }
            break;
        };
        // Resampler outputs stereo; if cpal wants more channels,
        // duplicate; if mono, take L and still pop R to keep the
        // ring aligned on stereo pairs.
        let r = pop().unwrap_or(l);
        for c in 0..channels {
            let sample = if c == 0 {
                l
            } else if c == 1 {
                r
            } else {
                (l + r) * 0.5
            };
            data[frame * channels + c] = sample * volume;
        }
        played += 1;
    }
    played
}

/// Where one run of pushed audio sits on the timeline: the `len` f32
/// samples starting at cumulative ring position `pos` belong to flush
/// generation `seq` and begin at presentation time `pts_us`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AudioSpan {
    seq: u64,
    pos: u64,
    len: u64,
    pts_us: i64,
}

/// The output callback's view of the audio ring: which generation and
/// presentation time every sample at the head belongs to.
///
/// This is what keeps audio locked to the picture. The old callback
/// counted samples as they played and added them to whatever the clock
/// was set to at the last seek, with no idea what those samples
/// actually were. Audio from before the seek, audio wiped by a drain
/// that fired at the wrong moment, and audio from the keyframe ahead of
/// the seek target all shifted what was heard against the clock, by a
/// different amount on every seek. Here the clock is the timestamp of
/// the sample being played, so those errors can't accumulate.
struct AudioTimeline {
    spans: VecDeque<AudioSpan>,
    /// Cumulative samples taken from the ring, played or discarded.
    /// The ring is FIFO, so this lines up with `AudioSpan::pos`.
    popped: u64,
    us_per_pair: f64,
    /// Presentation time just past the last sample played.
    head_pts_us: Option<i64>,
}

impl AudioTimeline {
    fn new(ring_rate: u32) -> Self {
        Self {
            spans: VecDeque::with_capacity(1024),
            popped: 0,
            us_per_pair: 1_000_000.0 / ring_rate.max(1) as f64,
            head_pts_us: None,
        }
    }

    /// Pop and throw away every sample belonging to a generation older
    /// than `seq`. Stops early — leaving the stale span at the front —
    /// if its samples haven't all reached the ring yet; nothing after
    /// it can play until they have.
    fn discard_stale(&mut self, seq: u64, mut pop: impl FnMut() -> Option<f32>) {
        while let Some(front) = self.spans.front().copied() {
            if front.seq >= seq {
                break;
            }
            let end = front.pos + front.len;
            while self.popped < end {
                if pop().is_none() {
                    return;
                }
                self.popped += 1;
            }
            self.spans.pop_front();
            self.head_pts_us = None;
        }
    }

    /// Ring position up to which samples are known to be `seq` audio.
    /// Samples beyond it either belong to another generation or haven't
    /// had their span published yet, and must not be played.
    fn playable_end(&self, seq: u64) -> u64 {
        let mut end = self.popped;
        for span in &self.spans {
            if span.seq != seq {
                break;
            }
            end = end.max(span.pos + span.len);
        }
        end
    }

    /// Record that `n` more samples were played.
    fn advance(&mut self, n: u64) {
        if n == 0 {
            return;
        }
        self.popped += n;
        if let Some(span) = self
            .spans
            .iter()
            .find(|s| s.pos < self.popped && self.popped <= s.pos + s.len)
        {
            let pairs = (self.popped - span.pos) / 2;
            self.head_pts_us =
                Some(span.pts_us + (pairs as f64 * self.us_per_pair).round() as i64);
        }
        while let Some(front) = self.spans.front() {
            if front.pos + front.len <= self.popped {
                self.spans.pop_front();
            } else {
                break;
            }
        }
    }

    fn head_pts_us(&self) -> Option<i64> {
        self.head_pts_us
    }

    /// Presentation time of the sample at ring position `pos`.
    fn pts_at(&self, pos: u64) -> Option<i64> {
        let span = self
            .spans
            .iter()
            .find(|s| s.pos <= pos && pos < s.pos + s.len)?;
        Some(span.pts_us + (((pos - span.pos) / 2) as f64 * self.us_per_pair).round() as i64)
    }

    /// Ring position of the first `seq` sample at or after `pts_us`, if
    /// audio reaching that far has been published.
    fn position_at_pts(&self, seq: u64, pts_us: i64) -> Option<u64> {
        for span in self.spans.iter().filter(|s| s.seq == seq) {
            let pairs = span.len / 2;
            let span_end = span.pts_us + (pairs as f64 * self.us_per_pair).round() as i64;
            if pts_us <= span.pts_us {
                return Some(span.pos);
            }
            if pts_us < span_end {
                let into = ((pts_us - span.pts_us) as f64 / self.us_per_pair).ceil() as u64;
                return Some(span.pos + into.min(pairs) * 2);
            }
        }
        None
    }

    fn is_idle(&self) -> bool {
        self.spans.is_empty()
    }
}

/// How long the output holds audio back after a seek while it waits for
/// the first frame at the new position, so sound and picture start
/// together. Capped so a video that is slow to produce that frame can't
/// hold the audio hostage indefinitely.
const SEEK_AUDIO_HOLD: Duration = Duration::from_millis(300);

/// Everything the audio output does for each buffer, whichever backend
/// asks for it: drop audio from superseded seeks, hold after a seek, mix,
/// and publish when the audio just written will be heard.
struct AudioRenderer {
    consumer: AudioConsumer,
    shared: Arc<Shared>,
    timeline: AudioTimeline,
    out_rate: u32,
    hold_frames: usize,
    hold_left: usize,
    /// Generation last reported by `take_generation_change`.
    seen_seq: u64,
    /// The anchor most recently published.
    last_anchor: Option<AudioAnchor>,
}

impl AudioRenderer {
    fn new(consumer: AudioConsumer, shared: Arc<Shared>, rate: u32) -> Self {
        let hold_frames = (rate as f64 * SEEK_AUDIO_HOLD.as_secs_f64()) as usize;
        let seen_seq = shared.flush_seq.load(Ordering::Acquire);
        Self {
            consumer,
            shared,
            timeline: AudioTimeline::new(rate),
            out_rate: rate,
            hold_frames,
            hold_left: hold_frames,
            seen_seq,
            last_anchor: None,
        }
    }

    /// True, once, after a seek has superseded everything written so far.
    fn take_generation_change(&mut self) -> bool {
        let seq = self.shared.flush_seq.load(Ordering::Acquire);
        if seq != self.seen_seq {
            self.seen_seq = seq;
            true
        } else {
            false
        }
    }

    /// The output is gone for good.
    fn fail(&mut self, reason: &str) {
        eprintln!("video_player: audio output stopped: {}", reason);
        self.shared.audio_output_failed.store(true, Ordering::Relaxed);
        let mut guard = match self.shared.audio_anchor.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = None;
    }

    fn publish(&mut self, anchor: AudioAnchor) {
        self.last_anchor = Some(anchor);
        let mut guard = match self.shared.audio_anchor.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        *guard = Some(anchor);
    }

    /// Fill one device buffer of interleaved `channels`-channel f32.
    /// `heard_at` is when the buffer's first frame reaches the listener.
    fn render(&mut self, data: &mut [f32], channels: usize, heard_at: Instant) {
        let shared = self.shared.clone();
        let now = Instant::now();
        shared
            .output_latency_us
            .store(signed_micros_between(now, heard_at), Ordering::Relaxed);

        let volume = f32::from_bits(shared.volume_bits.load(Ordering::Relaxed));
        let seq = shared.flush_seq.load(Ordering::Acquire);
        let frozen = shared.clock_frozen.load(Ordering::Relaxed);
        let paused = shared.audio_paused.load(Ordering::Relaxed);
        let frames = data.len() / channels.max(1);

        // Pick up spans published since the last buffer. The worker only
        // holds this lock for a push_back.
        {
            let mut incoming = match shared.audio_spans.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            self.timeline.spans.extend(incoming.drain(..));
        }

        // Throw away audio from superseded seeks. This runs even while
        // paused, so a string of paused seeks or frame steps can't fill
        // the ring with dead audio.
        let consumer = &mut self.consumer;
        self.timeline.discard_stale(seq, || consumer.try_pop());

        // Paused, or just seeked and still waiting for the first frame
        // at the new position: play silence without consuming anything.
        let holding = frozen && self.hold_left > 0;
        if paused || holding {
            data.fill(0.0);
            if holding && !paused {
                self.hold_left = self.hold_left.saturating_sub(frames);
            }
            return;
        }
        if !frozen {
            self.hold_left = self.hold_frames;
        }

        // Only play samples that are both in the ring and covered by a
        // published span for this generation. Rounding availability down
        // to whole stereo pairs keeps left and right from ever swapping.
        let base = self.timeline.popped;
        let first_pts = self.timeline.pts_at(base);
        let available = self.consumer.occupied_len() as u64 & !1;
        let mut limit = self.timeline.playable_end(seq).min(base + available);
        // Stop exactly at an A-B out-point, so no audio past it ever
        // reaches the device; the demuxer wraps once it has been heard.
        let loop_end = shared.loop_end_us.load(Ordering::Relaxed);
        if loop_end != NO_LOOP_POINT {
            if let Some(stop_at) = self.timeline.position_at_pts(seq, loop_end) {
                limit = limit.min(stop_at.max(base));
            }
        }

        let consumer = &mut self.consumer;
        let mut taken = 0u64;
        let mut first_sample: Option<f32> = None;
        let played_frames = mix_audio_output(data, channels, volume, || {
            if base + taken >= limit {
                return None;
            }
            let s = consumer.try_pop();
            if s.is_some() {
                taken += 1;
                if first_sample.is_none() {
                    first_sample = s;
                }
            }
            s
        });
        self.timeline.advance(taken);
        if let Some(v) = first_sample {
            if let Ok(mut probe) = shared.probe_first_sample.lock() {
                *probe = Some((v, heard_at));
            }
        }

        // A seek that landed mid-buffer makes this timing stale.
        if frozen || shared.flush_seq.load(Ordering::Acquire) != seq {
            return;
        }

        if played_frames > 0 {
            if let (Some(pts_us), Some(end_pts_us)) = (first_pts, self.timeline.head_pts_us()) {
                self.publish(AudioAnchor {
                    seq,
                    pts_us,
                    heard_at,
                    end_pts_us,
                });
            }
        } else if shared.audio_exhausted.load(Ordering::Relaxed)
            && self.timeline.is_idle()
            && self.consumer.is_empty()
        {
            // The audio has run out but the video hasn't. An ordinary
            // underrun must not move the clock, but once no more audio is
            // coming, silence is simply elapsed time: publish it as if it
            // were audio, so the picture plays out to its end.
            let start = self
                .last_anchor
                .filter(|a| a.seq == seq)
                .map(|a| a.end_pts_us)
                .or(self.timeline.head_pts_us())
                .unwrap_or_else(|| shared.audio_clock_us.load(Ordering::Relaxed));
            let duration_us = (frames as f64 * 1_000_000.0 / self.out_rate as f64) as i64;
            self.publish(AudioAnchor {
                seq,
                pts_us: start,
                heard_at,
                end_pts_us: start + duration_us,
            });
        }

        // Keep the stored clock current for anything that reads it
        // directly rather than interpolating.
        if let Some(pts) = shared.audible_pts_at(now) {
            shared.audio_clock_us.store(pts, Ordering::Relaxed);
        }
    }
}

#[cfg(windows)]
impl wasapi_output::RenderSource for AudioRenderer {
    fn fill(&mut self, data: &mut [f32], channels: usize, heard_at: Instant) {
        self.render(data, channels, heard_at);
    }

    fn take_flush(&mut self) -> bool {
        self.take_generation_change()
    }

    fn on_failed(&mut self, reason: &str) {
        self.fail(reason);
    }
}

/// Start the audio output: WASAPI with device-clock timing on Windows,
/// cpal elsewhere or if WASAPI can't be used.
fn start_audio_output(consumer: AudioConsumer, shared: &Arc<Shared>, rate: u32) -> AudioOutput {
    let renderer = AudioRenderer::new(consumer, shared.clone(), rate);

    // FPV_AUDIO_BACKEND=cpal skips WASAPI, so the fallback path can be
    // exercised on a machine where WASAPI works.
    #[cfg(windows)]
    let prefer_cpal = std::env::var("FPV_AUDIO_BACKEND").map(|v| v == "cpal").unwrap_or(false);
    #[cfg(windows)]
    let renderer = if prefer_cpal {
        renderer
    } else {
        match wasapi_output::WasapiOutput::start(renderer, rate) {
            Ok(output) => return AudioOutput::Wasapi(output),
            Err(wasapi_output::StartError::Unavailable(reason, renderer)) => {
                eprintln!("video_player: WASAPI output unavailable ({}); using cpal", reason);
                renderer
            }
            Err(wasapi_output::StartError::Lost(reason)) => {
                eprintln!(
                    "video_player: audio output failed to start ({}); timing playback from the wall clock",
                    reason
                );
                return AudioOutput::None;
            }
        }
    };

    match start_cpal_output(renderer, shared.clone()) {
        Ok(stream) => AudioOutput::Cpal(stream),
        Err(e) => {
            eprintln!(
                "video_player: audio output unavailable ({}); timing playback from the wall clock",
                e
            );
            AudioOutput::None
        }
    }
}

/// cpal output. Its notion of when a buffer is heard is only an estimate
/// (on Windows, the free space in its buffer), which is why WASAPI is
/// preferred there.
fn start_cpal_output(mut renderer: AudioRenderer, shared: Arc<Shared>) -> Result<cpal::Stream, String> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| "no default audio output device".to_string())?;
    let config = device
        .default_output_config()
        .map_err(|e| format!("audio default config: {}", e))?;
    let channels = config.channels() as usize;

    let shared_err = shared.clone();
    let err_fn = move |err| {
        eprintln!("cpal stream error: {}", err);
        shared_err.audio_output_failed.store(true, Ordering::Relaxed);
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device
            .build_output_stream(
                &config.into(),
                move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                    let ts = info.timestamp();
                    let until_heard = ts.playback.duration_since(&ts.callback).unwrap_or_default();
                    renderer.render(data, channels, Instant::now() + until_heard);
                },
                err_fn,
                None,
            )
            .map_err(|e| format!("cpal build stream: {}", e))?,
        other => return Err(format!("unsupported cpal sample format: {:?}", other)),
    };

    // Start immediately. The renderer plays silence until
    // `Player::play()` clears `audio_paused`, and keeps running while
    // paused so stale audio from seeks is discarded as it arrives.
    stream
        .play()
        .map_err(|e| format!("cpal start stream: {}", e))?;
    Ok(stream)
}

// =========================================================================
// Decode pipeline: demuxer thread + video decode thread + audio decode thread
// =========================================================================
//
// The demuxer owns the format input and is responsible for reading
// packets and routing them to the appropriate decoder thread via
// unbounded channels. Each decoder thread runs independently, so a
// slow video frame cannot starve audio playback. On seek or loop
// wraparound, the demuxer calls ictx.seek, clears shared state, and
// sends a Flush marker into each channel so the decoder threads flush
// their internal state and resume cleanly.

fn run_decode_pipeline(
    path: PathBuf,
    cmd_rx: Receiver<Command>,
    shared: Arc<Shared>,
    audio_producer: <HeapRb<f32> as Split>::Prod,
    egui_ctx: egui::Context,
    target_rate: u32,
    high_bit_depth_enabled: bool,
) -> Result<(), String> {
    let mut ictx =
        ffmpeg::format::input(&path).map_err(|e| format!("open: {}", e))?;

    // --- Video stream ---
    let (video_stream_index, video_time_base, video_frame_interval_us) = {
        let stream = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| "no video stream".to_string())?;
        let rate = stream.avg_frame_rate();
        let frame_us = if rate.denominator() > 0 && rate.numerator() > 0 {
            1_000_000_i64 * rate.denominator() as i64 / rate.numerator() as i64
        } else {
            33_333
        };
        (stream.index(), stream.time_base(), frame_us)
    };

    // Re-enabled now that decode threads are split: HW transfer lives
    // on the video thread and can no longer starve the audio path.
    const HW_ACCEL: bool = true;

    let (video_decoder, hw_enabled) = {
        let stream = ictx
            .stream(video_stream_index)
            .ok_or_else(|| "stream disappeared".to_string())?;
        let mut ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("video codec ctx: {}", e))?;
        // Enable multi-threaded decoding — 0 means "use one thread per
        // available core". Big win for software H.264/HEVC decode on
        // multi-core machines. Harmless when HW decode is active.
        ctx.set_threading(ffmpeg::codec::threading::Config {
            kind: ffmpeg::codec::threading::Type::Frame,
            count: 0,
        });
        let hw = if HW_ACCEL {
            unsafe { try_enable_hw_accel(ctx.as_mut_ptr()) }
        } else {
            false
        };
        let dec = ctx
            .decoder()
            .video()
            .map_err(|e| format!("video decoder: {}", e))?;
        (dec, hw)
    };
    let _ = hw_enabled;
    let video_src_w = video_decoder.width();
    let video_src_h = video_decoder.height();
    // A zero-sized video reaches swscale and the wgpu texture
    // descriptor as an invalid extent, neither of which fails
    // gracefully — swscale asserts and wgpu raises a validation
    // error that aborts the process. Refuse the file instead.
    if video_src_w == 0 || video_src_h == 0 {
        return Err(format!(
            "video reports zero dimensions ({}x{})",
            video_src_w, video_src_h
        ));
    }

    // Container duration is missing on some files (AV_NOPTS_VALUE
    // arrives as a large negative number) and wrong on others. Fall
    // back to the video stream's own duration, and treat anything
    // still non-positive as unknown so downstream `duration > 0`
    // guards take over rather than computing seek targets from junk.
    let duration_us = {
        let container = ictx.duration();
        if container > 0 {
            container
        } else {
            ictx.stream(video_stream_index)
                .map(|stream| {
                    let d = stream.duration();
                    if d > 0 {
                        ts_to_us(d, stream.time_base())
                    } else {
                        0
                    }
                })
                .unwrap_or(0)
        }
    };

    // --- Audio stream (optional) ---
    let audio_info = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .map(|s| s.index());

    // The resampler is built by the audio worker from the first decoded
    // frame: a frame's format is authoritative, whereas the stream
    // parameters are sometimes incomplete until decoding starts.
    let (audio_decoder, audio_time_base, audio_stream_index): (
        Option<ffmpeg::decoder::Audio>,
        ffmpeg::Rational,
        Option<usize>,
    ) = if let Some(idx) = audio_info {
        let stream = ictx.stream(idx).unwrap();
        let time_base = stream.time_base();
        let ctx = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("audio codec ctx: {}", e))?;
        let dec = ctx
            .decoder()
            .audio()
            .map_err(|e| format!("audio decoder: {}", e))?;
        (Some(dec), time_base, Some(idx))
    } else {
        (None, ffmpeg::Rational::new(1, 1), None)
    };

    // Populate initial shared state and hand off to Paused.
    {
        let mut state = shared.state.lock().unwrap();
        state.duration_us = duration_us;
        state.width = video_src_w;
        state.height = video_src_h;
        state.has_audio = audio_stream_index.is_some();
        state.frame_interval_us = video_frame_interval_us;
        state.player_state = PlayerState::Paused;
    }
    egui_ctx.request_repaint();

    // --- Spawn the two decode workers ---
    // Unbounded channels so the demuxer never blocks on `send` while
    // waiting for the workers to catch up. Blocking here would
    // prevent the demuxer from reaching its `cmd_rx.try_recv()` at
    // the top of the loop, which is what causes rapid-seek commands
    // to be queued but never processed until playback resumes.
    // Runtime back-pressure is still applied via the `video_queue`
    // length check above (demuxer sleeps when the decoded frame
    // queue is full), so in normal playback the channels don't
    // grow unbounded — they only accumulate during the brief
    // priming window after a seek, where we *want* the demuxer to
    // run ahead.
    // Bounded channels cap how far the demuxer can race ahead of
    // the workers, keeping memory bounded. Flushes are signalled
    // out-of-band via `shared.flush_seq`, so send-side blocking
    // here can't deadlock seeks even when the worker is paused.
    let (video_tx, video_rx) = mpsc::sync_channel::<DecodeMsg>(8);
    let (audio_tx_opt, audio_rx_opt) = if audio_stream_index.is_some() {
        let (tx, rx) = mpsc::sync_channel::<DecodeMsg>(16);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };

    let video_handle = {
        let shared_c = shared.clone();
        let ctx_c = egui_ctx.clone();
        thread::spawn(move || {
            video_decode_loop(
                video_decoder,
                video_time_base,
                video_rx,
                shared_c,
                ctx_c,
                high_bit_depth_enabled,
            );
        })
    };
    let audio_handle = match (audio_decoder, audio_rx_opt) {
        (Some(dec), Some(rx)) => {
            let shared_a = shared.clone();
            Some(thread::spawn(move || {
                audio_decode_loop(dec, audio_time_base, rx, shared_a, audio_producer, target_rate);
            }))
        }
        _ => None,
    };

    // --- Demuxer loop ---
    let mut paused = true;
    let mut looping = false;
    let mut eof = false;
    // When true, the demuxer reads packets regardless of `paused`
    // state until the video worker has successfully pushed at least
    // one post-seek frame. This ensures the user sees the new frame
    // at the seek target even while playback is paused.
    let mut priming_after_seek = false;
    // Safety cap so we don't spin forever on a broken file.
    let mut prime_packets_read: u32 = 0;
    const MAX_PRIME_PACKETS: u32 = 600;

    // Carried across iterations: the packet-routing back-pressure
    // helper can pull commands out of cmd_rx when it has to wait for
    // channel space, and it stashes them here so the next iteration
    // of the main loop processes them instead of losing them.
    let mut pending_seek: Option<i64> = None;
    let mut pending_play: Option<bool> = None;
    // End-of-file bookkeeping. `eof_drain_sent` makes sure the codec
    // flush goes out exactly once per EOF, and `tail` tracks whether
    // the clock is still making progress so a stalled clock can't
    // wedge the demuxer waiting for a tail that will never play.
    let mut eof_drain_sent = false;
    let mut tail = TailWatch::new();

    'main: loop {
        // Drain all pending commands in one go and coalesce them.
        // Multiple queued `Seek`s collapse into the latest — so a
        // rapid drag on the scrub bar never processes a backlog.
        let mut should_stop = false;
        loop {
            match cmd_rx.try_recv() {
                Ok(Command::Play) => pending_play = Some(true),
                Ok(Command::Pause) => pending_play = Some(false),
                Ok(Command::Seek(target_us)) => pending_seek = Some(target_us),
                Ok(Command::SetLooping(l)) => looping = l,
                Ok(Command::Stop) => {
                    should_stop = true;
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    should_stop = true;
                    break;
                }
            }
        }
        if should_stop {
            break 'main;
        }
        if let Some(play) = pending_play.take() {
            paused = !play;
            eof = false;
            let mut state = shared.state.lock().unwrap();
            if state.player_state != PlayerState::Error {
                state.player_state = if play {
                    PlayerState::Playing
                } else {
                    PlayerState::Paused
                };
            }
        }
        // --- A-B clip loop -------------------------------------
        // Checked every iteration, against the *display* clock rather
        // than how far the demuxer has read, so the wrap happens when
        // the viewer reaches the out-point. Back-pressure keeps the
        // demuxer parked a few frames past it in the meantime, so the
        // over-read is bounded even on a five-second clip cut from a
        // ten-minute file.
        if pending_seek.is_none() && !paused && clip_out_point_reached(&shared) {
            if looping {
                let start_us = shared.loop_start_us.load(Ordering::Relaxed);
                pending_seek = Some(if start_us == NO_LOOP_POINT {
                    0
                } else {
                    start_us.max(0)
                });
            } else {
                // Range set but looping off: stop at the out-point
                // instead of playing on past it. The ring can hold
                // seconds of audio beyond the marker; silence the output
                // so it doesn't keep playing over a held frame. Resuming
                // from here always seeks, which discards it.
                paused = true;
                shared.audio_paused.store(true, Ordering::Relaxed);
                let mut state = shared.state.lock().unwrap();
                if state.player_state == PlayerState::Playing {
                    state.player_state = PlayerState::EndOfFile;
                }
                drop(state);
                egui_ctx.request_repaint();
            }
        }

        if let Some(target_us) = pending_seek.take() {
            reposition(&mut ictx, &shared, target_us);
            eof = false;
            eof_drain_sent = false;
            tail.reset();
            priming_after_seek = true;
            prime_packets_read = 0;
        }

        if priming_after_seek {
            let pushed = shared
                .post_seek_frames_pushed
                .load(Ordering::Relaxed);
            if pushed > 0 || prime_packets_read >= MAX_PRIME_PACKETS {
                priming_after_seek = false;
            }
        }
        // While paused, keep the demuxer routing until the display
        // queue is well-populated. This lets `Player::step_frames`
        // and the seek bar consume many buffered frames without
        // falling back into the expensive full-seek-and-decode
        // path. `PAUSED_FILL_TARGET` is the depth we aim for; once
        // the queue has that many buffered frames we sleep until
        // tick() drains some.
        if paused && !priming_after_seek {
            const PAUSED_FILL_TARGET: usize = 12;
            let qlen = shared.video_queue.lock().unwrap().len();
            if qlen >= PAUSED_FILL_TARGET {
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        }
        // --- End of file ---------------------------------------
        if eof {
            // Flush the codecs once. A frame-threaded decoder holds
            // several finished frames internally and only releases
            // them when told the stream is over; skipping this drain
            // silently discarded the last fraction of a second of
            // every video, which is what made clips look cut short.
            if !eof_drain_sent {
                eof_drain_sent = true;
                shared.drain_complete.store(false, Ordering::Relaxed);
                let tag = shared.flush_seq.load(Ordering::Relaxed);
                send_drain(&video_tx, tag, &shared);
                if let Some(tx) = audio_tx_opt.as_ref() {
                    send_drain(tx, tag, &shared);
                }
            }

            // Nothing to do until the tail has actually been shown.
            // While paused the clock legitimately sits still, so the
            // stall detector is held off and we simply idle.
            let tail_done = tail.tail_played(&shared, paused);
            if !tail_done {
                thread::sleep(Duration::from_millis(5));
                continue;
            }

            if !looping || paused {
                let mut state = shared.state.lock().unwrap();
                if state.player_state == PlayerState::Playing {
                    state.player_state = PlayerState::EndOfFile;
                }
                drop(state);
                egui_ctx.request_repaint();
                thread::sleep(Duration::from_millis(30));
                continue;
            }

            // Wrap. Going through `reposition` rather than an ad-hoc
            // reset is what keeps looping in step with seeking: it
            // raises `clock_frozen`, so `tick()` re-anchors the
            // no-audio wall clock to the first frame of the new pass
            // instead of leaving it pinned at the end of the file.
            let start_us = shared.loop_start_us.load(Ordering::Relaxed);
            let wrap_to = if start_us == NO_LOOP_POINT {
                0
            } else {
                start_us.max(0)
            };
            if !reposition(&mut ictx, &shared, wrap_to) {
                eprintln!("demuxer: loop seek to {}us failed; stopping loop", wrap_to);
                looping = false;
                let mut state = shared.state.lock().unwrap();
                state.player_state = PlayerState::EndOfFile;
                continue;
            }
            eof = false;
            eof_drain_sent = false;
            tail.reset();
            priming_after_seek = true;
            prime_packets_read = 0;
            continue;
        }

        // Back-pressure: cap the decoded-video queue so it doesn't
        // grow without bound. CRITICAL: only stall once the clock has
        // actually started advancing. During startup the audio decoder
        // needs its first packets to arrive through the demuxer before
        // it can produce samples that move the clock. Stalling on
        // depth alone here deadlocks high-fps content (e.g. HEVC
        // 60fps) where ~16 decoded frames span less than one audio
        // decode cycle — the video queue hits the cap before any
        // audio packet has been routed, so audio never starts, the
        // clock never ticks, tick() never pops, and the demuxer
        // stays stalled forever.
        let clock_us = shared.clock_now_us();
        let video_queue_len = shared.video_queue.lock().unwrap().len();
        if video_queue_len > 48 && clock_us > 0 {
            thread::sleep(Duration::from_millis(3));
            continue;
        }
        if video_queue_len > 240 {
            // Hard safety cap regardless of clock state.
            thread::sleep(Duration::from_millis(3));
            continue;
        }

        // Read one packet.
        let mut packet = ffmpeg::Packet::empty();
        match packet.read(&mut ictx) {
            Ok(()) => {}
            Err(ffmpeg::Error::Eof) => {
                // All packets read. The tail-drain, wait, and loop
                // wrap are handled by the `if eof` block at the top
                // of the loop so they stay on the same command-
                // servicing path as everything else.
                eof = true;
                continue;
            }
            Err(e) => {
                eprintln!("packet read: {}", e);
                thread::sleep(Duration::from_millis(10));
                continue;
            }
        }

        let packet_idx = packet.stream();
        if packet_idx == video_stream_index {
            let tag = shared.flush_seq.load(Ordering::Relaxed);
            let mut msg = Some(DecodeMsg::Packet(packet, tag));
            loop {
                match video_tx.try_send(msg.take().unwrap()) {
                    Ok(()) => break,
                    Err(mpsc::TrySendError::Full(m)) => {
                        msg = Some(m);
                        if shared.stopping.load(Ordering::Relaxed) {
                            break 'main;
                        }
                        // At an A-B out-point `tick()` holds the last in-range
                        // frame, so the display queue stops draining and this send
                        // would wait until the worker's push times out (~2 s) before
                        // the wrap could happen. Hand control back to the top of the
                        // loop, which wraps or stops; every way out of that state
                        // seeks, so the dropped packet is never needed.
                        if !paused && clip_out_point_reached(&shared) {
                            break;
                        }
                        // Peek for a pending seek/stop so we don't
                        // stall here while the user is waiting.
                        if let Ok(cmd) = cmd_rx.try_recv() {
                            match cmd {
                                Command::Play => pending_play = Some(true),
                                Command::Pause => pending_play = Some(false),
                                Command::Seek(us) => {
                                    pending_seek = Some(us);
                                    // Drop the in-flight stale
                                    // packet; the seek path
                                    // will flush anyway.
                                    break;
                                }
                                Command::SetLooping(l) => looping = l,
                                Command::Stop => break 'main,
                            }
                        }
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(mpsc::TrySendError::Disconnected(_)) => break 'main,
                }
            }
        } else if Some(packet_idx) == audio_stream_index {
            if let Some(ref tx) = audio_tx_opt {
                let tag = shared.flush_seq.load(Ordering::Relaxed);
                let mut msg = Some(DecodeMsg::Packet(packet, tag));
                loop {
                    match tx.try_send(msg.take().unwrap()) {
                        Ok(()) => break,
                        Err(mpsc::TrySendError::Full(m)) => {
                            msg = Some(m);
                            if shared.stopping.load(Ordering::Relaxed) {
                                break 'main;
                            }
                            // At an A-B out-point `tick()` holds the last in-range
                            // frame, so the display queue stops draining and this send
                            // would wait until the worker's push times out (~2 s) before
                            // the wrap could happen. Hand control back to the top of the
                            // loop, which wraps or stops; every way out of that state
                            // seeks, so the dropped packet is never needed.
                            if !paused && clip_out_point_reached(&shared) {
                                break;
                            }
                            if let Ok(cmd) = cmd_rx.try_recv() {
                                match cmd {
                                    Command::Play => pending_play = Some(true),
                                    Command::Pause => pending_play = Some(false),
                                    Command::Seek(us) => {
                                        pending_seek = Some(us);
                                        break;
                                    }
                                    Command::SetLooping(l) => looping = l,
                                    Command::Stop => break 'main,
                                }
                            }
                            thread::sleep(Duration::from_millis(2));
                        }
                        Err(mpsc::TrySendError::Disconnected(_)) => break 'main,
                    }
                }
            }
        }
        if priming_after_seek {
            prime_packets_read += 1;
        }
    }

    // Drop the send halves so the workers exit their recv loops.
    drop(video_tx);
    drop(audio_tx_opt);
    let _ = video_handle.join();
    if let Some(h) = audio_handle {
        let _ = h.join();
    }

    Ok(())
}

/// Reposition the demuxer to `target_us` and reset every piece of
/// shared state the display pipeline keys off.
///
/// Both explicit seeks and end-of-file loop wraps go through here so
/// the two can't drift apart. They did: the old loop-wrap path reset
/// the clock but skipped `clock_frozen`, and for a video with no audio
/// track that left `tick()`'s wall clock still anchored to the moment
/// playback began — so the instant the wrap set the clock to zero, the
/// next tick dragged it back out to the end of the file and every
/// newly decoded frame looked overdue.
///
/// Returns false if the container refused to seek at all.
fn reposition(
    ictx: &mut ffmpeg::format::context::Input,
    shared: &Arc<Shared>,
    target_us: i64,
) -> bool {
    // av_seek_frame(..., BACKWARD) lands on a keyframe at or before
    // the target. Combined with the `min_display_pts_us` drop on the
    // worker, that gives frame-accurate seeking — the decoder gets
    // the keyframe, walks forward until the target, and the first
    // displayed frame is the first one at or past it.
    let mut ok = unsafe {
        av_seek_frame(
            ictx.as_mut_ptr(),
            -1,
            target_us,
            AVSEEK_FLAG_BACKWARD,
        ) >= 0
    };
    unsafe {
        let fctx = ictx.as_mut_ptr();
        if !fctx.is_null() {
            if !(*fctx).pb.is_null() {
                // Rewinding to the very start is the one case worth a
                // fallback: some containers reject the indexed seek
                // after EOF but rewind fine at the byte level, and
                // this path runs on every loop wrap.
                if !ok && target_us <= 0 {
                    ok = avio_seek((*fctx).pb, 0, 0 /* SEEK_SET */) >= 0
                        || av_seek_frame(
                            fctx,
                            -1,
                            0,
                            AVSEEK_FLAG_BACKWARD | AVSEEK_FLAG_ANY,
                        ) >= 0;
                }
                (*(*fctx).pb).eof_reached = 0;
                (*(*fctx).pb).error = 0;
            }
            avformat_flush(fctx);
        }
    }

    // Set every gate BEFORE bumping `flush_seq`, because the workers
    // may service the flush and decode subsequent packets before this
    // function returns. If the gates aren't all set first, post-flush
    // frames can slip through with a stale `min_display_pts_us` and
    // get displayed at completely wrong timestamps.
    shared
        .min_display_pts_us
        .store(target_us, Ordering::Relaxed);
    shared
        .post_seek_frames_pushed
        .store(0, Ordering::Relaxed);
    shared.flush_pending.store(true, Ordering::Relaxed);
    shared.last_pushed_pts_us.store(target_us, Ordering::Relaxed);
    shared.drain_complete.store(false, Ordering::Relaxed);
    shared.audio_clock_us.store(target_us, Ordering::Relaxed);
    shared.clock_frozen.store(true, Ordering::Relaxed);
    // Audio decoded from the keyframe up to the target belongs before
    // the first picture the viewer will see; the audio worker drops it.
    shared
        .audio_trim_before_us
        .store(target_us, Ordering::Relaxed);
    shared.audio_exhausted.store(false, Ordering::Relaxed);
    // Bump the flush counter so workers notice on their next decode
    // iteration. Sending a flush *message* through the bounded channel
    // can deadlock when the channel is full and the worker is parked
    // in back-pressure (which happens on pause-then-seek). Release
    // ordering publishes the gates above to anyone who observes the
    // new generation. Audio still in the ring from the old generation
    // is discarded by the output callback, sample-exact, by its tag.
    shared.flush_seq.fetch_add(1, Ordering::Release);
    shared.video_queue.lock().unwrap().clear();
    ok
}

/// True once playback has reached an A-B out-point: a marker is set,
/// the clock isn't frozen mid-reposition, and it has passed the marker.
fn clip_out_point_reached(shared: &Shared) -> bool {
    let end_us = shared.loop_end_us.load(Ordering::Relaxed);
    end_us != NO_LOOP_POINT
        && !shared.clock_frozen.load(Ordering::Relaxed)
        && shared.clock_now_us() >= end_us
}

/// Hand a decoder worker its end-of-stream marker, giving up rather
/// than blocking forever if the worker has stopped draining. The
/// channel is bounded, and at EOF the worker may still be parked in
/// `push_with_cap` waiting for the display queue to empty.
fn send_drain(tx: &mpsc::SyncSender<DecodeMsg>, tag: u64, shared: &Arc<Shared>) {
    let deadline = Instant::now() + Duration::from_millis(500);
    loop {
        match tx.try_send(DecodeMsg::Drain(tag)) {
            Ok(()) => return,
            Err(mpsc::TrySendError::Full(_)) => {
                if shared.stopping.load(Ordering::Relaxed)
                    || shared.flush_seq.load(Ordering::Relaxed) != tag
                    || Instant::now() > deadline
                {
                    return;
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(mpsc::TrySendError::Disconnected(_)) => return,
        }
    }
}

/// Decides when the viewer has actually seen the end of the file.
///
/// The demuxer reaches EOF long before playback does, so it can't wrap
/// a loop the moment packets run out. The old code waited until the
/// clock was within 250 ms of the *container* duration, which both
/// truncated the last quarter second and misfired on files whose
/// declared duration doesn't match their content. This waits for the
/// real thing: the codec drained, every decoded frame displayed, and
/// the clock past the final frame's PTS.
struct TailWatch {
    last_clock_us: i64,
    last_progress: Instant,
}

impl TailWatch {
    fn new() -> Self {
        Self {
            last_clock_us: i64::MIN,
            last_progress: Instant::now(),
        }
    }

    fn reset(&mut self) {
        self.last_clock_us = i64::MIN;
        self.last_progress = Instant::now();
    }

    /// True once the tail has played. `paused` suppresses the stall
    /// fallback — a paused clock is standing still on purpose.
    fn tail_played(&mut self, shared: &Arc<Shared>, paused: bool) -> bool {
        let clock_us = shared.clock_now_us();
        let queue_len = shared.video_queue.lock().unwrap().len();
        let drained = shared.drain_complete.load(Ordering::Relaxed);
        let tail_pts = shared.last_pushed_pts_us.load(Ordering::Relaxed);

        if drained && queue_len == 0 && clock_us >= tail_pts {
            return true;
        }

        if clock_us != self.last_clock_us {
            self.last_clock_us = clock_us;
            self.last_progress = Instant::now();
            return false;
        }
        if paused {
            self.last_progress = Instant::now();
            return false;
        }
        // The clock has stopped moving while playing. That is the
        // expected end state for a file whose audio ran out — the
        // mixer deliberately doesn't advance the clock over the
        // silence it fills on underrun — so treat a sustained stall
        // as the end rather than waiting for a tick that isn't
        // coming.
        self.last_progress.elapsed() > TAIL_STALL_TIMEOUT
    }
}

/// How long the clock may sit still at end of file before we accept
/// that playback has finished. Long enough not to trip on ordinary
/// scheduler jitter, short enough not to read as a hang.
const TAIL_STALL_TIMEOUT: Duration = Duration::from_millis(600);

const WORKER_QUEUE_CAP: usize = 16;

/// Push a decoded video frame into the shared display queue, then
/// (after pushing) park the worker if the queue is now at capacity.
/// This is post-push back-pressure: we always make at least one
/// frame available — important during post-seek priming when tick()
/// needs a frame to land before it can clear `clock_frozen` — and
/// only stall once that frame is buffered. Exits on the shared
/// `stopping` flag so `decode_thread.join()` can return promptly.
/// Returns true when the frame actually landed in the shared queue.
/// Callers must consult this result before bumping
/// `post_seek_frames_pushed`, otherwise the demuxer can incorrectly
/// conclude that seek priming is done and stop routing packets.
fn push_with_cap(shared: &Arc<Shared>, frame: VideoFrame, last_flush_seq: u64) -> bool {
    let mut slot = Some(frame);
    for _ in 0..400 {
        if shared.stopping.load(Ordering::Relaxed) {
            return false;
        }
        if shared.flush_seq.load(Ordering::Relaxed) != last_flush_seq {
            return false;
        }
        {
            let mut q = shared.video_queue.lock().unwrap();
            if q.len() < WORKER_QUEUE_CAP {
                let frame = slot.take().expect("frame present");
                // Record the furthest-along frame the pipeline has
                // produced. At EOF this is the file's true final PTS,
                // which is what the demuxer waits for the clock to
                // reach before looping.
                let pts_us = frame.pts_us;
                q.push_back(frame);
                drop(q);
                shared
                    .last_pushed_pts_us
                    .fetch_max(pts_us, Ordering::Relaxed);
                return true;
            }
        }
        thread::sleep(Duration::from_millis(5));
    }
    false
}

/// Video decode worker: pulls DecodeMsg from the channel, decodes,
/// scales to RGBA, pushes finished frames into shared.video_queue.
fn video_decode_loop(
    mut decoder: ffmpeg::decoder::Video,
    time_base: ffmpeg::Rational,
    rx: mpsc::Receiver<DecodeMsg>,
    shared: Arc<Shared>,
    egui_ctx: egui::Context,
    high_bit_depth_enabled: bool,
) {
    let mut video_frame = ffmpeg::frame::Video::empty();
    let mut rgba_frame = ffmpeg::frame::Video::empty();
    let mut sw_frame = ffmpeg::frame::Video::empty();
    let mut video_scaler: Option<ffmpeg::software::scaling::context::Context> = None;
    // cfg is (src_fmt, src_w, src_h, output_is_16bit) so a transition
    // between 8- and 16-bit output forces a scaler rebuild.
    let mut scaler_cfg: Option<(ffmpeg::format::Pixel, u32, u32, bool)> = None;
    let mut last_flush_seq: u64 = 0;
    loop {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => return,
        };
        let cur_seq = shared.flush_seq.load(Ordering::Relaxed);
        if cur_seq != last_flush_seq {
            last_flush_seq = cur_seq;
            decoder.flush();
            shared.video_queue.lock().unwrap().clear();
            shared.flush_pending.store(false, Ordering::Relaxed);
        }
        // A drain iteration pushes the codec's internal backlog out
        // instead of feeding it a new packet; everything downstream is
        // identical, so the two share the receive loop below.
        let draining = match msg {
            DecodeMsg::Drain(tag) => {
                if tag != last_flush_seq {
                    continue;
                }
                let _ = decoder.send_eof();
                true
            }
            DecodeMsg::Packet(p, tag) => {
                if tag != last_flush_seq {
                    continue;
                }
                if decoder.send_packet(&p).is_err() {
                    continue;
                }
                false
            }
        };

        while decoder.receive_frame(&mut video_frame).is_ok() {
            // If a seek arrived mid-decode, drop the frame in hand
            // and stop draining the decoder — the next outer loop
            // iteration will service the flush.
            let s = shared.flush_seq.load(Ordering::Relaxed);
            if s != last_flush_seq {
                break;
            }
            // Drop any frames the decoder is draining from pre-seek
            // state — they show as "fast-forward" blur otherwise.
            if shared.flush_pending.load(Ordering::Relaxed) {
                continue;
            }
            let raw_format = unsafe { (*video_frame.as_ptr()).format };
            let is_hw = raw_format == AVPixelFormat::AV_PIX_FMT_D3D11 as i32;

            let pts_raw = video_frame.pts().unwrap_or(0);
            let pts_us = ts_to_us(pts_raw, time_base);

            // After a seek, av_seek_frame lands on the keyframe
            // before the target. Frames between that keyframe and
            // the seek target must be decoded (they feed the codec
            // state) but should not be displayed, so we drop any
            // frame whose pts is strictly before the target.
            let min_pts = shared
                .min_display_pts_us
                .load(Ordering::Relaxed);
            if pts_us < min_pts {
                unsafe {
                    av_frame_unref(video_frame.as_mut_ptr());
                }
                continue;
            }


            let source_frame: &ffmpeg::frame::Video = if is_hw {
                let ret = unsafe {
                    av_hwframe_transfer_data(
                        sw_frame.as_mut_ptr(),
                        video_frame.as_ptr(),
                        0,
                    )
                };
                if ret < 0 {
                    eprintln!("hwframe transfer failed: {}", ret);
                    continue;
                }
                &sw_frame
            } else {
                &video_frame
            };

            let src_fmt = source_frame.format();
            let src_w = source_frame.width();
            let src_h = source_frame.height();

            // NV12 fast path: skip swscale and pass Y + UV planes
            // straight through. The GPU does YUV→RGB conversion in
            // the fragment shader. Typically hit after HW decode
            // because `av_hwframe_transfer_data` produces NV12 on
            // CPU.
            if src_fmt == ffmpeg::format::Pixel::NV12 {
                let y_stride = source_frame.stride(0);
                let uv_stride = source_frame.stride(1);
                let y_src = source_frame.data(0);
                let uv_src = source_frame.data(1);
                let y_row = src_w as usize;
                let uv_row = src_w as usize; // interleaved UV: 2 bytes per pixel at half width = src_w
                let uv_h = (src_h as usize + 1) / 2;
                let mut y_plane: Vec<u8> = Vec::with_capacity(y_row * src_h as usize);
                let mut uv_plane: Vec<u8> = Vec::with_capacity(uv_row * uv_h);
                if y_stride == y_row {
                    y_plane.extend_from_slice(&y_src[..y_row * src_h as usize]);
                } else {
                    for y in 0..src_h as usize {
                        let s = y * y_stride;
                        y_plane.extend_from_slice(&y_src[s..s + y_row]);
                    }
                }
                if uv_stride == uv_row {
                    uv_plane.extend_from_slice(&uv_src[..uv_row * uv_h]);
                } else {
                    for y in 0..uv_h {
                        let s = y * uv_stride;
                        uv_plane.extend_from_slice(&uv_src[s..s + uv_row]);
                    }
                }
                let pushed = push_with_cap(
                    &shared,
                    VideoFrame {
                        pts_us,
                        payload: VideoFramePayload::Nv12 {
                            width: src_w,
                            height: src_h,
                            y_plane,
                            uv_plane,
                        },
                    },
                    last_flush_seq,
                );
                if pushed {
                    shared
                        .post_seek_frames_pushed
                        .fetch_add(1, Ordering::Relaxed);
                    egui_ctx.request_repaint();
                }
                continue;
            }

            // Decide whether this frame should go through the 16-bit
            // path: feature enabled AND the source is actually a
            // 10-bit-or-higher pixel format worth preserving.
            let use_16bit = high_bit_depth_enabled && is_high_bit_depth_format(src_fmt);
            let dst_fmt = if use_16bit {
                ffmpeg::format::Pixel::RGBA64LE
            } else {
                ffmpeg::format::Pixel::RGBA
            };
            let bytes_per_pixel: usize = if use_16bit { 8 } else { 4 };

            // Cap output width to keep the RGBA buffer small and
            // texture uploads cheap. 1920-wide is enough for most
            // displays — scaling up to fit on the egui side uses GPU
            // bilinear at essentially zero cost, whereas CPU-side
            // upload of a 3440-wide frame chews bandwidth.
            const MAX_OUT_WIDTH: u32 = 1920;
            let (dst_w, dst_h) = if src_w > MAX_OUT_WIDTH {
                let ratio = MAX_OUT_WIDTH as f64 / src_w as f64;
                let h = ((src_h as f64 * ratio).round() as u32).max(1);
                (MAX_OUT_WIDTH, h)
            } else {
                (src_w, src_h)
            };

            let cfg = (src_fmt, src_w, src_h, use_16bit);
            if scaler_cfg != Some(cfg) {
                let flags = ffmpeg::software::scaling::flag::Flags::BILINEAR
                    | ffmpeg::software::scaling::flag::Flags::ACCURATE_RND
                    | ffmpeg::software::scaling::flag::Flags::FULL_CHR_H_INT
                    | ffmpeg::software::scaling::flag::Flags::FULL_CHR_H_INP;
                match ffmpeg::software::scaling::context::Context::get(
                    src_fmt,
                    src_w,
                    src_h,
                    dst_fmt,
                    dst_w,
                    dst_h,
                    flags,
                ) {
                    Ok(s) => {
                        video_scaler = Some(s);
                        scaler_cfg = Some(cfg);
                    }
                    Err(e) => {
                        eprintln!("scaler rebuild: {}", e);
                        continue;
                    }
                }
            }
            let scaler = match video_scaler.as_mut() {
                Some(s) => s,
                None => continue,
            };
            if scaler.run(source_frame, &mut rgba_frame).is_err() {
                continue;
            }

            let w = rgba_frame.width();
            let h = rgba_frame.height();
            let stride = rgba_frame.stride(0);
            let src = rgba_frame.data(0);
            let row = w as usize * bytes_per_pixel;
            let total = row * h as usize;
            let mut pixels: Vec<u8> = Vec::with_capacity(total);
            if stride == row {
                pixels.extend_from_slice(&src[..total]);
            } else {
                for y in 0..h as usize {
                    let s = y * stride;
                    pixels.extend_from_slice(&src[s..s + row]);
                }
            }

            let payload = if use_16bit {
                VideoFramePayload::Rgba16Unorm {
                    width: w,
                    height: h,
                    bytes: pixels,
                }
            } else {
                // Transmute Vec<u8> -> Vec<Color32> without touching
                // the pixel bytes. Safe: Color32 is #[repr(C)] [u8; 4]
                // with alignment 1, same as u8, byte count multiple
                // of 4.
                let color_pixels: Vec<egui::Color32> = unsafe {
                    debug_assert!(pixels.len() % 4 == 0);
                    debug_assert!(pixels.capacity() % 4 == 0);
                    let len = pixels.len() / 4;
                    let cap = pixels.capacity() / 4;
                    let ptr = pixels.as_mut_ptr() as *mut egui::Color32;
                    std::mem::forget(pixels);
                    Vec::from_raw_parts(ptr, len, cap)
                };
                let image = egui::ColorImage {
                    size: [w as usize, h as usize],
                    pixels: color_pixels,
                };
                VideoFramePayload::Rgba8Srgb {
                    width: w,
                    height: h,
                    image,
                }
            };

            let pushed = push_with_cap(
                &shared,
                VideoFrame {
                    pts_us,
                    payload,
                },
                last_flush_seq,
            );
            if pushed {
                shared
                    .post_seek_frames_pushed
                    .fetch_add(1, Ordering::Relaxed);
                egui_ctx.request_repaint();
            }
        }

        if draining {
            // Put the codec back into a state that accepts packets, so
            // a seek or loop wrap after EOF can resume decoding, and
            // tell the demuxer the backlog is out — until it sees this
            // it must not mistake an empty display queue for the end
            // of the video.
            decoder.flush();
            if shared.flush_seq.load(Ordering::Relaxed) == last_flush_seq {
                shared.drain_complete.store(true, Ordering::Relaxed);
                egui_ctx.request_repaint();
            }
        }
    }
}

// =========================================================================
// NV12 YUV → RGB custom render pipeline (partial Phase E)
// =========================================================================

/// WGSL shader for NV12 sampling: full-screen-quad vertex shader with
/// a user-specified NDC rect, fragment shader that samples Y + UV,
/// applies BT.709 limited-range YUV→R'G'B', decodes sRGB gamma, and
/// outputs linear RGB (the swapchain's sRGB-encoded target will
/// re-encode on store).
const NV12_SHADER_SRC: &str = r#"
@group(0) @binding(0) var y_tex: texture_2d<f32>;
@group(0) @binding(1) var uv_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

// NDC rect (left, top, right, bottom) in the current render-pass
// viewport's coordinate space. Set from the CPU side every frame
// based on the unclamped video_rect relative to the clamped
// egui_wgpu viewport, so the quad is positioned correctly even when
// panning pushes the video partially off-screen.
struct NdcRect {
    rect: vec4<f32>,
};
@group(0) @binding(3) var<uniform> ndc: NdcRect;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    let l = ndc.rect.x;
    let t = ndc.rect.y;
    let r = ndc.rect.z;
    let b = ndc.rect.w;
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(l, t),
        vec2<f32>(r, t),
        vec2<f32>(r, b),
        vec2<f32>(l, t),
        vec2<f32>(r, b),
        vec2<f32>(l, b),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

fn srgb_to_linear(c: f32) -> f32 {
    if (c <= 0.04045) {
        return c / 12.92;
    }
    return pow((c + 0.055) / 1.055, 2.4);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let y  = textureSample(y_tex,  samp, in.uv).r;
    let uv = textureSample(uv_tex, samp, in.uv).rg;

    // BT.709 limited-range YUV (all channels normalized to [0,1]).
    // Output is gamma-encoded R'G'B'.
    let y_lin = 1.164 * (y - 0.0627);
    let u_off = uv.r - 0.502;
    let v_off = uv.g - 0.502;
    let r_g = clamp(y_lin + 1.793 * v_off, 0.0, 1.0);
    let g_g = clamp(y_lin - 0.213 * u_off - 0.533 * v_off, 0.0, 1.0);
    let b_g = clamp(y_lin + 2.112 * u_off, 0.0, 1.0);

    // FRAGMENT_RETURN
}
"#;

struct YuvRenderer {
    pipeline: Arc<wgpu::RenderPipeline>,
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
}

impl Clone for YuvRenderer {
    fn clone(&self) -> Self {
        Self {
            pipeline: self.pipeline.clone(),
            bind_group_layout: self.bind_group_layout.clone(),
            sampler: self.sampler.clone(),
        }
    }
}

impl YuvRenderer {
    fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let is_srgb_target = matches!(
            target_format,
            wgpu::TextureFormat::Rgba8UnormSrgb
                | wgpu::TextureFormat::Bgra8UnormSrgb
                | wgpu::TextureFormat::Bc1RgbaUnormSrgb
                | wgpu::TextureFormat::Bc2RgbaUnormSrgb
                | wgpu::TextureFormat::Bc3RgbaUnormSrgb
                | wgpu::TextureFormat::Bc7RgbaUnormSrgb
        );
        // Build the fragment output expression based on target format.
        // - sRGB target: hardware encodes linear→sRGB on store, so the
        //   shader must output LINEAR. We apply the BT.709/sRGB inverse
        //   transfer function to the YUV→R'G'B' result.
        // - Non-sRGB target: hardware stores bytes as-is, so the shader
        //   must output gamma-encoded R'G'B' directly.
        let source = if is_srgb_target {
            NV12_SHADER_SRC.replace(
                "// FRAGMENT_RETURN",
                "return vec4<f32>(\
                    srgb_to_linear(r_g),\
                    srgb_to_linear(g_g),\
                    srgb_to_linear(b_g),\
                    1.0,\
                );",
            )
        } else {
            NV12_SHADER_SRC.replace(
                "// FRAGMENT_RETURN",
                "return vec4<f32>(r_g, g_g, b_g, 1.0);",
            )
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("nv12_shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("nv12_bind_group_layout"),
                entries: &[
                    // Y plane texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // UV plane texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // NDC-rect uniform (vec4<f32>): left, top, right,
                    // bottom in the clamped-viewport's NDC space. Lets
                    // us draw the correct un-squished slice of the
                    // video when the video rect extends past the
                    // visible area (egui clamps the viewport, so a
                    // hardcoded full-NDC quad would otherwise look
                    // like a resize on pan).
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nv12_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("nv12_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("nv12_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            sampler: Arc::new(sampler),
        }
    }
}

/// Per-video NV12 texture state: the Y plane texture, the UV plane
/// texture, and the bind group that binds them to the `YuvRenderer`
/// pipeline.
struct Nv12GpuState {
    y_texture: wgpu::Texture,
    uv_texture: wgpu::Texture,
    ndc_buffer: wgpu::Buffer,
    bind_group: Arc<wgpu::BindGroup>,
    width: u32,
    height: u32,
}

/// Per-frame PaintCallback that runs inside egui_wgpu's render pass
/// to draw the NV12 video. Holds cheap Arc-clones of the pipeline and
/// bind group plus the NDC quad coordinates for the video area.
struct Nv12PaintCallback {
    pipeline: Arc<wgpu::RenderPipeline>,
    bind_group: Arc<wgpu::BindGroup>,
}

impl egui_wgpu::CallbackTrait for Nv12PaintCallback {
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &egui_wgpu::CallbackResources,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..6, 0..1);
    }
}

fn is_high_bit_depth_format(fmt: ffmpeg::format::Pixel) -> bool {
    use ffmpeg::format::Pixel::*;
    matches!(
        fmt,
        YUV420P10LE | YUV420P10BE
        | YUV422P10LE | YUV422P10BE
        | YUV444P10LE | YUV444P10BE
        | YUV420P12LE | YUV420P12BE
        | YUV422P12LE | YUV422P12BE
        | YUV444P12LE | YUV444P12BE
        | YUV420P16LE | YUV420P16BE
        | YUV422P16LE | YUV422P16BE
        | YUV444P16LE | YUV444P16BE
        | P010LE | P010BE
        | P016LE | P016BE
    )
}

type AudioProducer = <HeapRb<f32> as Split>::Prod;
type Resampler = ffmpeg::software::resampling::context::Context;

/// swresample can't work with a layout that has only a channel count
/// and no named channels (`ChannelOrder::Unspecified` — typical of PCM
/// in Matroska, AVI and some camera MOVs). It rejects every such frame
/// with `AVERROR_INPUT_CHANGED`, so those files played no audio at all,
/// and with no audio to drive the clock their video froze on the first
/// frame. Substitute the standard layout for the channel count.
fn normalize_frame_layout(frame: &mut ffmpeg::frame::Audio) {
    let (order, channels) = {
        let layout = frame.ch_layout();
        (layout.order(), layout.channels())
    };
    if order == ffmpeg::util::channel_layout::ChannelOrder::Unspecified {
        frame.set_ch_layout(ffmpeg::ChannelLayout::default_for_channels(channels.max(1)));
    }
}

/// Append a planar-stereo f32 frame to `out` as interleaved L/R pairs.
fn append_interleaved(frame: &ffmpeg::frame::Audio, out: &mut Vec<f32>) {
    let n = frame.samples();
    if n == 0 {
        return;
    }
    let left = frame.plane::<f32>(0);
    let right: &[f32] = if frame.planes() > 1 {
        frame.plane::<f32>(1)
    } else {
        left
    };
    out.reserve(n * 2);
    for i in 0..n {
        out.push(*left.get(i).unwrap_or(&0.0));
        out.push(*right.get(i).unwrap_or(&0.0));
    }
}

/// Run one frame through swresample into a freshly allocated output
/// frame that swresample sizes itself.
///
/// The binding's `Context::run` sizes its output to the *input* sample
/// count. That is too small whenever the output rate is higher (44.1 kHz
/// audio on a 48 kHz device), and when the output frame is reused its
/// capacity shrinks to the smallest frame seen — the short final frame
/// of a file, so after the first loop every 1024-sample frame came out
/// as 768. Whatever doesn't fit waits in swresample's internal FIFO and
/// comes out later, so the sound heard falls progressively behind its
/// timestamps. (The binding's `delay()` reports that backlog in whole
/// seconds, so the flush meant to drain it almost never fired.) Handing
/// swresample an unallocated frame makes it allocate room for all the
/// output available — the documented behavior of `swr_convert_frame`.
fn convert_frame(
    resampler: &mut Resampler,
    input: &ffmpeg::frame::Audio,
    target_rate: u32,
) -> Result<ffmpeg::frame::Audio, ffmpeg::Error> {
    let mut output = ffmpeg::frame::Audio::empty();
    // SAFETY: `output` is a freshly allocated AVFrame with no buffers;
    // we set the three fields swr_convert_frame requires and let it
    // allocate the data. `input` is a valid decoded frame.
    unsafe {
        let o = output.as_mut_ptr();
        (*o).format = ffmpeg::ffi::AVSampleFormat::AV_SAMPLE_FMT_FLTP as i32;
        (*o).sample_rate = target_rate as i32;
        ffmpeg::ffi::av_channel_layout_default(&mut (*o).ch_layout, 2);
        match ffmpeg::ffi::swr_convert_frame(resampler.as_mut_ptr(), o, input.as_ptr()) {
            0 => Ok(output),
            e => Err(ffmpeg::Error::from(e)),
        }
    }
}

/// Resample `frame` to interleaved stereo f32 at `target_rate`,
/// appending to `out`.
///
/// The resampler is built from the frame's own format on first use and
/// rebuilt once if a frame is rejected, which happens when a stream
/// changes format mid-flight (broadcast recordings switching between
/// stereo and 5.1, for instance).
fn resample_into(
    slot: &mut Option<Resampler>,
    frame: &ffmpeg::frame::Audio,
    out: &mut Vec<f32>,
    target_rate: u32,
) -> Result<(), String> {
    for attempt in 0..2 {
        if slot.is_none() || attempt > 0 {
            let built = Resampler::get2(
                frame.format(),
                frame.ch_layout(),
                frame.rate(),
                ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar),
                ffmpeg::ChannelLayout::STEREO,
                target_rate,
            )
            .map_err(|e| format!("resampler: {}", e))?;
            *slot = Some(built);
        }
        let resampler = slot.as_mut().expect("resampler just built");
        match convert_frame(resampler, frame, target_rate) {
            Ok(converted) => {
                append_interleaved(&converted, out);
                return Ok(());
            }
            Err(e) if attempt > 0 => return Err(format!("resample: {}", e)),
            Err(_) => {}
        }
    }
    unreachable!("the second attempt always returns")
}

/// Push interleaved stereo samples into the output ring, publishing an
/// `AudioSpan` ahead of each run so the output callback always knows
/// which generation and timestamp the samples it plays belong to.
///
/// A span goes out *before* its samples and covers exactly the samples
/// that follow it: the run length is fixed from the ring's free space
/// up front, and since only this thread pushes, that space can only
/// grow before the push lands. Runs are whole stereo pairs, so left and
/// right can't drift out of step.
///
/// If the ring stays full past a short deadline the remainder is
/// dropped. The ring only stays full when the output isn't draining,
/// and blocking here would back-pressure the demuxer and stall video
/// priming after a seek; the next span simply starts later, so the
/// clock skips the gap rather than drifting.
fn push_timed(
    samples: &[f32],
    mut pts_us: i64,
    seq: u64,
    pushed_total: &mut u64,
    producer: &mut AudioProducer,
    spans: &Mutex<VecDeque<AudioSpan>>,
    us_per_pair: f64,
    abandon: impl Fn() -> bool,
) {
    const PUSH_DEADLINE: Duration = Duration::from_millis(4);
    let started = Instant::now();
    let mut offset = 0usize;
    while offset < samples.len() {
        let room = producer.vacant_len() & !1;
        let n = room.min(samples.len() - offset);
        if n == 0 {
            if started.elapsed() > PUSH_DEADLINE || abandon() {
                return;
            }
            thread::yield_now();
            continue;
        }
        {
            let mut queue = match spans.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            queue.push_back(AudioSpan {
                seq,
                pos: *pushed_total,
                len: n as u64,
                pts_us,
            });
        }
        let pushed = producer.push_slice(&samples[offset..offset + n]);
        debug_assert_eq!(pushed, n, "ring space shrank under a single producer");
        *pushed_total += pushed as u64;
        offset += pushed;
        pts_us += ((pushed / 2) as f64 * us_per_pair).round() as i64;
    }
}

/// Audio decode worker: decodes, resamples to stereo f32 at the output
/// rate, trims anything before the current seek target, and pushes the
/// result into the output ring behind timestamped `AudioSpan`s.
fn audio_decode_loop(
    mut decoder: ffmpeg::decoder::Audio,
    time_base: ffmpeg::Rational,
    rx: mpsc::Receiver<DecodeMsg>,
    shared: Arc<Shared>,
    mut producer: AudioProducer,
    target_rate: u32,
) {
    let us_per_pair = 1_000_000.0 / target_rate.max(1) as f64;
    let mut frame = ffmpeg::frame::Audio::empty();
    let mut resampler: Option<Resampler> = None;
    let mut chunk: Vec<f32> = Vec::new();
    let mut pushed_total: u64 = 0;
    let mut next_pts_us: Option<i64> = None;
    let mut last_flush_seq: u64 = 0;
    let mut reported_failure = false;

    loop {
        let msg = match rx.recv() {
            Ok(m) => m,
            Err(_) => return,
        };
        let cur_seq = shared.flush_seq.load(Ordering::Acquire);
        if cur_seq != last_flush_seq {
            last_flush_seq = cur_seq;
            decoder.flush();
            next_pts_us = None;
            // swresample keeps a little audio internally (the resampling
            // filter's history). After a seek that is audio from the old
            // position; start the new one clean.
            resampler = None;
        }
        let draining = match msg {
            DecodeMsg::Drain(tag) => {
                if tag != last_flush_seq {
                    continue;
                }
                let _ = decoder.send_eof();
                true
            }
            DecodeMsg::Packet(p, tag) => {
                if tag != last_flush_seq {
                    continue;
                }
                if decoder.send_packet(&p).is_err() {
                    continue;
                }
                false
            }
        };

        while decoder.receive_frame(&mut frame).is_ok() {
            // A seek arrived mid-decode: everything from here on is
            // stale, and the output would discard it anyway.
            if shared.flush_seq.load(Ordering::Acquire) != last_flush_seq {
                break;
            }
            normalize_frame_layout(&mut frame);
            chunk.clear();
            if let Err(e) = resample_into(&mut resampler, &frame, &mut chunk, target_rate) {
                if !reported_failure {
                    eprintln!("video_player: skipping undecodable audio: {}", e);
                    reported_failure = true;
                }
                // Don't let a broken track hold the picture hostage.
                shared.audio_exhausted.store(true, Ordering::Relaxed);
                continue;
            }
            let pairs = chunk.len() / 2;
            if pairs == 0 {
                continue;
            }

            // Prefer the frame's own timestamp; fall back to where the
            // previous frame ended when a container doesn't carry one.
            let pts_us = frame
                .pts()
                .or_else(|| frame.timestamp())
                .map(|t| ts_to_us(t, time_base))
                .or(next_pts_us)
                .unwrap_or(0);
            next_pts_us = Some(pts_us + (pairs as f64 * us_per_pair).round() as i64);

            // Drop audio that precedes the seek target.
            let trim_before = shared.audio_trim_before_us.load(Ordering::Relaxed);
            let skip_pairs = if pts_us < trim_before {
                ((trim_before.saturating_sub(pts_us) as f64 / us_per_pair).round() as usize).min(pairs)
            } else {
                0
            };
            if skip_pairs == pairs {
                continue;
            }
            let start_pts = pts_us + (skip_pairs as f64 * us_per_pair).round() as i64;

            shared.audio_exhausted.store(false, Ordering::Relaxed);
            let seq = last_flush_seq;
            push_timed(
                &chunk[skip_pairs * 2..],
                start_pts,
                seq,
                &mut pushed_total,
                &mut producer,
                &shared.audio_spans,
                us_per_pair,
                || {
                    shared.stopping.load(Ordering::Relaxed)
                        || shared.flush_seq.load(Ordering::Acquire) != seq
                },
            );
        }

        if draining {
            decoder.flush();
            if shared.flush_seq.load(Ordering::Acquire) == last_flush_seq {
                shared.audio_exhausted.store(true, Ordering::Relaxed);
            }
        }
    }
}

fn ts_to_us(ts: i64, tb: ffmpeg::Rational) -> i64 {
    // PTS (in stream time_base units) → microseconds.
    (ts as f64 * tb.numerator() as f64 / tb.denominator() as f64 * 1_000_000.0) as i64
}

// =========================================================================
// D3D11VA hardware acceleration (Phase 3)
// =========================================================================

/// Picked by avcodec_open2 / get_format when negotiating the decoder's
/// output pixel format. Prefer D3D11 if it's in the offered list so the
/// decoder will produce GPU surfaces; otherwise accept the first listed
/// (software) format and decode on the CPU.
unsafe extern "C" fn get_hw_format(
    _ctx: *mut AVCodecContext,
    mut pix_fmts: *const AVPixelFormat,
) -> AVPixelFormat {
    let mut fallback = AVPixelFormat::AV_PIX_FMT_NONE;
    while !pix_fmts.is_null() && *pix_fmts != AVPixelFormat::AV_PIX_FMT_NONE {
        if fallback == AVPixelFormat::AV_PIX_FMT_NONE {
            fallback = *pix_fmts;
        }
        if *pix_fmts == AVPixelFormat::AV_PIX_FMT_D3D11 {
            return AVPixelFormat::AV_PIX_FMT_D3D11;
        }
        pix_fmts = pix_fmts.add(1);
    }
    fallback
}

/// Set up a D3D11VA device context and attach it to the given codec
/// context. Must be called BEFORE avcodec_open2 (before
/// `.decoder().video()` in ffmpeg-the-third terms).
unsafe fn try_enable_hw_accel(cctx: *mut AVCodecContext) -> bool {
    let mut hw_dev: *mut AVBufferRef = std::ptr::null_mut();
    let ret = av_hwdevice_ctx_create(
        &mut hw_dev,
        AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
        std::ptr::null(),
        std::ptr::null_mut(),
        0,
    );
    if ret < 0 || hw_dev.is_null() {
        return false;
    }
    // The codec context takes its own ref; we unref our local handle
    // after the assignment, leaving only the decoder's ownership.
    (*cctx).hw_device_ctx = av_buffer_ref(hw_dev);
    (*cctx).get_format = Some(get_hw_format);
    av_buffer_unref(&mut hw_dev);
    true
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a pop-closure that drains `samples` front-to-back and
    /// returns `None` once empty. Used by the mix_audio_output tests
    /// to simulate a ring buffer with a fixed number of stereo samples.
    fn make_pop(samples: Vec<f32>) -> impl FnMut() -> Option<f32> {
        let mut idx = 0usize;
        move || {
            if idx < samples.len() {
                let v = samples[idx];
                idx += 1;
                Some(v)
            } else {
                None
            }
        }
    }

    #[test]
    fn mix_full_stereo_copies_samples_verbatim() {
        let mut data = vec![0.0f32; 6];
        let played =
            mix_audio_output(&mut data, 2, 1.0, make_pop(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]));
        assert_eq!(played, 3);
        assert_eq!(data, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    }

    #[test]
    fn mix_applies_volume() {
        let mut data = vec![0.0f32; 4];
        let played = mix_audio_output(&mut data, 2, 0.5, make_pop(vec![1.0, 1.0, 1.0, 1.0]));
        assert_eq!(played, 2);
        for s in &data {
            assert!((*s - 0.5).abs() < 1e-6, "sample {} not attenuated", s);
        }
    }

    /// The core A/V-sync regression: a fully-empty ring must return 0
    /// so the caller doesn't advance the audio clock. If this returns
    /// anything > 0, video drifts permanently ahead of audio on every
    /// underrun.
    #[test]
    fn mix_full_underrun_returns_zero_and_zero_fills() {
        // Preload with garbage so we can verify the function overwrites
        // the entire buffer with silence.
        let mut data = vec![0.7f32; 8];
        let played = mix_audio_output(&mut data, 2, 1.0, || None);
        assert_eq!(played, 0, "underrun must not count any played frames");
        assert!(
            data.iter().all(|&s| s == 0.0),
            "underrun must zero-fill the entire output; got {:?}",
            data
        );
    }

    /// Partial underrun: ring has some samples, cpal asks for more
    /// than it can supply. Clock must advance only over the samples
    /// that were actually popped, and the tail of the output must be
    /// silence so we don't play whatever stale data was in the buffer.
    #[test]
    fn mix_partial_underrun_counts_only_real_frames() {
        // 2 stereo frames in the ring (4 samples), cpal asks for 6
        // frames (12 samples).
        let mut data = vec![0.9f32; 12];
        let played = mix_audio_output(&mut data, 2, 1.0, make_pop(vec![0.1, 0.2, 0.3, 0.4]));
        assert_eq!(played, 2, "must only count popped frames");
        assert_eq!(&data[..4], &[0.1, 0.2, 0.3, 0.4]);
        assert!(
            data[4..].iter().all(|&s| s == 0.0),
            "tail after underrun must be silence; got {:?}",
            &data[4..]
        );
    }

    /// Regression check: after an underrun the silence the function
    /// writes must not leak into the next call's played count.
    /// Running mix_audio_output back-to-back against a drained source
    /// must always return 0, not phantom frames from the prior
    /// zero-fill.
    #[test]
    fn mix_underrun_does_not_leak_into_next_call() {
        // 3 stereo frames in the ring, cpal asks for 5 each call.
        let mut pop = make_pop(vec![0.1, 0.1, 0.2, 0.2, 0.3, 0.3]);
        let mut data = vec![0.0f32; 10];
        let p1 = mix_audio_output(&mut data, 2, 1.0, &mut pop);
        let p2 = mix_audio_output(&mut data, 2, 1.0, &mut pop);
        let p3 = mix_audio_output(&mut data, 2, 1.0, &mut pop);
        assert_eq!(p1, 3, "first call must return only the available frames");
        assert_eq!(p2, 0, "second call must return 0 — silence is not a played frame");
        assert_eq!(p3, 0, "further calls must stay at 0");
    }

    #[test]
    fn mix_mono_output_takes_left_and_drops_right() {
        let mut data = vec![0.0f32; 2];
        let played = mix_audio_output(&mut data, 1, 1.0, make_pop(vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(played, 2);
        assert_eq!(data, vec![1.0, 3.0]);
    }

    #[test]
    fn mix_surround_duplicates_stereo_to_extra_channels() {
        let mut data = vec![0.0f32; 4];
        let played = mix_audio_output(&mut data, 4, 1.0, make_pop(vec![0.2, 0.4]));
        assert_eq!(played, 1);
        assert!((data[0] - 0.2).abs() < 1e-6);
        assert!((data[1] - 0.4).abs() < 1e-6);
        assert!(
            (data[2] - 0.3).abs() < 1e-6,
            "channel 2 should mix L+R (got {})",
            data[2]
        );
        assert!(
            (data[3] - 0.3).abs() < 1e-6,
            "channel 3 should mix L+R (got {})",
            data[3]
        );
    }

    // ---- AudioTimeline -------------------------------------------------

    const RATE: u32 = 48_000;

    fn span(seq: u64, pos: u64, len: u64, pts_us: i64) -> AudioSpan {
        AudioSpan { seq, pos, len, pts_us }
    }

    /// A stand-in for the output ring: samples numbered by position so a
    /// test can see exactly which ones were played or discarded.
    fn ring_of(n: usize) -> VecDeque<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn head_pts_follows_the_samples_actually_played() {
        let mut t = AudioTimeline::new(RATE);
        t.spans.push_back(span(0, 0, 960, 1_000_000)); // 480 pairs = 10 ms
        t.advance(480); // 240 pairs = 5 ms
        assert_eq!(t.head_pts_us(), Some(1_005_000));
        t.advance(480); // the rest of the span
        assert_eq!(t.head_pts_us(), Some(1_010_000));
        assert!(t.is_idle(), "a fully played span must be dropped");
    }

    /// The core seek fix: audio from before a seek is discarded exactly,
    /// and audio from after it is kept even when both are already in the
    /// ring at the moment the callback runs. The old blind drain threw
    /// both away, which is how audio ended up ahead of the picture.
    #[test]
    fn stale_generation_is_discarded_and_fresh_audio_survives() {
        let mut ring = ring_of(16);
        let mut t = AudioTimeline::new(RATE);
        t.spans.push_back(span(0, 0, 8, 0));
        t.spans.push_back(span(1, 8, 8, 5_000_000));

        t.discard_stale(1, || ring.pop_front());
        assert_eq!(t.popped, 8);
        assert_eq!(ring.front().copied(), Some(8.0), "fresh audio must still be queued");
        assert_eq!(t.playable_end(1), 16);

        t.advance(4);
        assert_eq!(t.head_pts_us(), Some(5_000_000 + 42), "2 pairs past the new span's start");
    }

    #[test]
    fn discard_waits_for_stale_samples_that_have_not_arrived() {
        // Span published for 8 samples, only 4 pushed so far.
        let mut ring = ring_of(4);
        let mut t = AudioTimeline::new(RATE);
        t.spans.push_back(span(0, 0, 8, 0));
        t.spans.push_back(span(1, 8, 8, 1_000));

        t.discard_stale(1, || ring.pop_front());
        assert_eq!(t.popped, 4);
        assert_eq!(t.spans.len(), 2, "the stale span stays until it is fully discarded");
        assert_eq!(t.playable_end(1), 4, "nothing may play while stale audio is ahead of it");
    }

    #[test]
    fn samples_without_a_published_span_are_not_playable() {
        let t = {
            let mut t = AudioTimeline::new(RATE);
            t.spans.push_back(span(3, 0, 8, 0));
            t
        };
        // The ring might hold more than 8 samples, but only 8 are
        // accounted for.
        assert_eq!(t.playable_end(3), 8);
    }

    #[test]
    fn a_newer_generation_is_left_alone() {
        // The callback read generation 1; a seek to generation 2 landed
        // while it ran. Generation-2 audio is the future, not garbage.
        let mut ring = ring_of(8);
        let mut t = AudioTimeline::new(RATE);
        t.spans.push_back(span(2, 0, 8, 0));
        t.discard_stale(1, || ring.pop_front());
        assert_eq!(t.popped, 0);
        assert_eq!(ring.len(), 8);
        assert_eq!(t.playable_end(1), 0, "and it isn't played under the old generation either");
    }

    #[test]
    fn playable_end_spans_contiguous_runs_of_one_generation() {
        let mut t = AudioTimeline::new(RATE);
        t.spans.push_back(span(1, 0, 8, 0));
        t.spans.push_back(span(1, 8, 8, 83));
        t.spans.push_back(span(2, 16, 8, 0));
        assert_eq!(t.playable_end(1), 16);
    }

    #[test]
    fn pts_lookups_map_ring_positions_to_presentation_time() {
        let mut t = AudioTimeline::new(RATE);
        t.spans.push_back(span(1, 0, 960, 1_000_000)); // 10 ms
        t.spans.push_back(span(1, 960, 960, 1_010_000));
        assert_eq!(t.pts_at(0), Some(1_000_000));
        assert_eq!(t.pts_at(480), Some(1_005_000));
        assert_eq!(t.pts_at(960), Some(1_010_000));
        assert_eq!(t.pts_at(1920), None, "past the published audio");

        // An out-point inside the second span stops just at or after it.
        let stop = t.position_at_pts(1, 1_012_500).unwrap();
        assert_eq!(stop, 960 + 240);
        assert!(t.pts_at(stop).unwrap() >= 1_012_500);
        assert_eq!(t.position_at_pts(1, 900_000), Some(0), "before the audio starts");
        assert_eq!(t.position_at_pts(1, 2_000_000), None, "not decoded that far yet");
        assert_eq!(t.position_at_pts(2, 1_005_000), None, "another generation");
    }

    fn shared_for_test() -> Shared {
        Shared {
            state: Mutex::new(SharedState {
                player_state: PlayerState::Playing,
                duration_us: 10_000_000,
                width: 1,
                height: 1,
                has_audio: true,
                frame_interval_us: 33_333,
                error: None,
            }),
            video_queue: Mutex::new(VecDeque::new()),
            audio_clock_us: AtomicI64::new(0),
            volume_bits: AtomicU32::new(1.0f32.to_bits()),
            clock_frozen: std::sync::atomic::AtomicBool::new(false),
            audio_spans: Mutex::new(VecDeque::new()),
            audio_paused: std::sync::atomic::AtomicBool::new(false),
            audio_trim_before_us: AtomicI64::new(i64::MIN),
            audio_exhausted: std::sync::atomic::AtomicBool::new(false),
            flush_pending: std::sync::atomic::AtomicBool::new(false),
            min_display_pts_us: AtomicI64::new(i64::MIN),
            post_seek_frames_pushed: std::sync::atomic::AtomicU32::new(0),
            stopping: std::sync::atomic::AtomicBool::new(false),
            flush_seq: std::sync::atomic::AtomicU64::new(3),
            last_pushed_pts_us: AtomicI64::new(0),
            drain_complete: std::sync::atomic::AtomicBool::new(false),
            loop_start_us: AtomicI64::new(NO_LOOP_POINT),
            loop_end_us: AtomicI64::new(NO_LOOP_POINT),
            probe_first_sample: Mutex::new(None),
            output_latency_us: AtomicI64::new(0),
            audio_anchor: Mutex::new(None),
            audio_output_failed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// The audible position is interpolated from the anchor: behind it
    /// before the buffer is heard (earlier audio is still playing), and
    /// advancing in real time after — so video can be scheduled against
    /// what the listener hears, compensated for output latency.
    #[test]
    fn audible_position_is_interpolated_from_the_anchor() {
        let shared = shared_for_test();
        let heard_at = Instant::now() + Duration::from_millis(40);
        *shared.audio_anchor.lock().unwrap() = Some(AudioAnchor {
            seq: 3,
            pts_us: 5_000_000,
            heard_at,
            end_pts_us: 5_010_000,
        });

        let at = |offset_ms: i64| {
            if offset_ms >= 0 {
                heard_at + Duration::from_millis(offset_ms as u64)
            } else {
                heard_at - Duration::from_millis((-offset_ms) as u64)
            }
        };
        assert_eq!(shared.audible_pts_at(at(0)), Some(5_000_000));
        assert_eq!(shared.audible_pts_at(at(-40)), Some(4_960_000), "40 ms of latency still to go");
        assert_eq!(shared.audible_pts_at(at(6)), Some(5_006_000));
        assert_eq!(
            shared.audible_pts_at(at(50)),
            Some(5_010_000),
            "can't pass the end of what has been written"
        );
    }

    #[test]
    fn a_seek_retires_the_anchor() {
        let shared = shared_for_test();
        *shared.audio_anchor.lock().unwrap() = Some(AudioAnchor {
            seq: 3,
            pts_us: 5_000_000,
            heard_at: Instant::now(),
            end_pts_us: 5_010_000,
        });
        shared.flush_seq.store(4, Ordering::Release);
        assert_eq!(shared.audible_pts_at(Instant::now()), None);
        shared.audio_clock_us.store(1_234_000, Ordering::Relaxed);
        assert_eq!(shared.clock_now_us(), 1_234_000, "falls back to the stored clock");
    }

    #[test]
    fn push_timed_publishes_exact_contiguous_spans() {
        let rb = HeapRb::<f32>::new(10);
        let (mut prod, mut cons) = rb.split();
        let spans = Mutex::new(VecDeque::new());
        let mut pushed_total = 0u64;
        let us_per_pair = 1_000_000.0 / RATE as f64;
        let samples: Vec<f32> = (0..16).map(|i| i as f32).collect();

        // Room for 10: one span of 10, the rest dropped at the deadline.
        push_timed(&samples, 0, 7, &mut pushed_total, &mut prod, &spans, us_per_pair, || false);
        assert_eq!(pushed_total, 10);
        assert_eq!(spans.lock().unwrap().back().copied(), Some(span(7, 0, 10, 0)));

        // Free 5 slots: only 4 are usable, because runs are whole pairs.
        for _ in 0..5 {
            cons.try_pop();
        }
        push_timed(&samples[..6], 104, 7, &mut pushed_total, &mut prod, &spans, us_per_pair, || false);
        assert_eq!(spans.lock().unwrap().back().copied(), Some(span(7, 10, 4, 104)));
        assert_eq!(pushed_total, 14);
    }

    /// End-to-end clock-advance check: simulate a realistic cpal
    /// callback sequence at 48 kHz where the decoder produces only
    /// 0.5 s worth of audio but cpal asks for 1 s. The returned
    /// played count, multiplied by `us_per_frame`, must be 500_000 us
    /// — not 1_000_000 us. This is the exact failure pattern that
    /// caused the original desync-over-time bug: each underrun used
    /// to advance the clock past the actual audio played.
    #[test]
    fn mix_clock_advance_matches_real_samples_only() {
        const SR: usize = 48_000;
        let us_per_frame = 1_000_000.0_f64 / SR as f64;
        let in_frames = SR / 2; // 0.5 s of stereo
        let samples: Vec<f32> = (0..in_frames * 2).map(|i| (i as f32) * 1e-6).collect();
        let mut data = vec![0.0f32; SR * 2];
        let played = mix_audio_output(&mut data, 2, 1.0, make_pop(samples));
        assert_eq!(played, in_frames);
        let advance_us = (played as f64 * us_per_frame) as i64;
        assert_eq!(
            advance_us, 500_000,
            "clock must advance 0.5 s, not the full 1 s request"
        );
        // And the unfilled tail must be silence — otherwise the cpal
        // device would replay whatever was in the buffer previously.
        for i in in_frames * 2..data.len() {
            assert_eq!(
                data[i], 0.0,
                "sample {} past underrun must be silence",
                i
            );
        }
    }
}
