//! Headless integration tests for `video_player::Player`.
//!
//! This binary spins up a **real wgpu device** (without a window or
//! display surface) and a minimal `egui_wgpu::Renderer` so the player's
//! direct-wgpu NV12 / RGBA8 / RGBA16 upload paths are actually
//! exercised. The earlier version of this test used the fallback
//! `egui::ColorImage` path, which meant bugs in the NV12/wgpu pipeline
//! slipped through.
//!
//! Run with:
//!
//!     cargo run --release --bin video_test -- "C:/path/to/video.mp4"
//!
//! Add `--only <text>` to run just the scenarios whose name contains it.
//!
//! If no argument is given, the binary searches common locations
//! (Downloads, Videos, Pictures) for the first .mp4/.mov/.mkv it can
//! find.
//!
//! The binary runs a series of scenarios against the same file and
//! exits with a non-zero code if any of them fail.

#[path = "../video_player.rs"]
mod video_player;

use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;

use video_player::{Player, PlayerState, WgpuBackend};

const PTS_TOLERANCE_US: i64 = 150_000; // ±150 ms tolerance for seeks.

fn find_test_video() -> Option<PathBuf> {
    if let Some(p) = env::args().nth(1) {
        let path = PathBuf::from(p);
        if path.exists() && path.is_file() {
            return Some(path);
        }
        eprintln!("video_test: arg path does not exist: {}", path.display());
    }
    for root in [
        "C:/Users/david/Downloads",
        "C:/Users/david/Videos",
        "C:/Users/david/Pictures",
    ] {
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if matches!(
                        ext.to_lowercase().as_str(),
                        "mp4" | "mov" | "mkv" | "webm" | "m4v"
                    ) {
                        return Some(path);
                    }
                }
            }
        }
    }
    None
}

/// The headless device, shared with `tick_and_flush` so every place the
/// harness drives the player can submit its uploads.
static GPU: std::sync::OnceLock<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> =
    std::sync::OnceLock::new();

/// Tick the player, then submit and poll the queue.
///
/// `Player::tick` uploads frames with `queue.write_texture`, which only
/// stages the data: wgpu keeps each staging buffer, and the texture it
/// targets, alive until the next `submit`. In the real app eframe
/// submits once per frame. This harness has no render loop, so without
/// an explicit submit every uploaded frame stayed resident for the
/// whole run — harmless for a few seconds of 640x360, but a long run on
/// 4K footage exhausts GPU and system memory.
fn tick_and_flush(player: &mut Player) {
    player.tick();
    if let Some((device, queue)) = GPU.get() {
        queue.submit(std::iter::empty());
        device.poll(wgpu::Maintain::Poll);
    }
}

/// Create a headless wgpu device + queue + renderer that mimics what
/// eframe's wgpu backend gives us at runtime, so the Player's NV12
/// path is actually exercised during tests.
fn make_headless_backend() -> (egui::Context, WgpuBackend) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("no wgpu adapter available");

    let supported_features = adapter.features();
    let want_16bit = wgpu::Features::TEXTURE_FORMAT_16BIT_NORM;
    let required_features = if supported_features.contains(want_16bit) {
        want_16bit
    } else {
        wgpu::Features::empty()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("video_test"),
            required_features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .expect("wgpu device creation failed");
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let _ = GPU.set((device.clone(), queue.clone()));
    let target_format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let renderer = egui_wgpu::Renderer::new(&device, target_format, None, 1, false);
    let renderer = Arc::new(egui::mutex::RwLock::new(renderer));
    let supports_16bit_norm = device.features().contains(want_16bit);
    let backend = WgpuBackend {
        device,
        queue,
        renderer,
        target_format,
        supports_16bit_norm,
    };
    (egui::Context::default(), backend)
}

#[derive(Clone)]
struct ScenarioResult {
    name: &'static str,
    pass: bool,
    details: String,
}

impl ScenarioResult {
    fn ok(name: &'static str, details: impl Into<String>) -> Self {
        Self {
            name,
            pass: true,
            details: details.into(),
        }
    }
    fn fail(name: &'static str, details: impl Into<String>) -> Self {
        Self {
            name,
            pass: false,
            details: details.into(),
        }
    }
}

/// Mimics what `eframe` actually does: only runs an "update cycle"
/// (tick() + a synthetic paint/run) when egui says it has a pending
/// repaint. This catches bugs where the player's worker or
/// `Player::seek` fails to request a repaint, which would leave the
/// GUI staring at a stale frame even though the test harness's own
/// `spin_and_collect` (which unconditionally ticks 60×/sec) would
/// be happy.
fn gated_spin_and_collect(
    ctx: &egui::Context,
    player: &mut Player,
    wall: Duration,
) -> Vec<i64> {
    let start = Instant::now();
    let mut seen: Vec<i64> = Vec::new();
    let mut last = i64::MIN;
    // Always run one initial tick so the first poll isn't empty.
    tick_and_flush(player);
    {
        let pts = player.uploaded_pts_us_for_test();
        if pts != i64::MIN {
            seen.push(pts);
            last = pts;
        }
    }
    while start.elapsed() < wall {
        // eframe only runs `update()` again if egui has requested a
        // repaint (from last update, from an event, or from another
        // thread). If neither Player::seek nor the decoder worker
        // request one, the loop stalls and no new frame is shown.
        if ctx.has_requested_repaint() {
            tick_and_flush(player);
            let pts = player.uploaded_pts_us_for_test();
            if pts != last && pts != i64::MIN {
                seen.push(pts);
                last = pts;
            }
        }
        std::thread::sleep(Duration::from_millis(4));
    }
    seen
}

/// Run `tick()` at ~60 Hz for `wall` duration, collecting every unique
/// uploaded pts the Player reports.
fn spin_and_collect(player: &mut Player, wall: Duration) -> Vec<i64> {
    let start = Instant::now();
    let mut seen: Vec<i64> = Vec::new();
    let mut last = i64::MIN;
    let mut ticks = 0u32;
    while start.elapsed() < wall {
        tick_and_flush(player);
        ticks += 1;
        let pts = player.uploaded_pts_us_for_test();
        if pts != last && pts != i64::MIN {
            seen.push(pts);
            last = pts;
        }
        std::thread::sleep(Duration::from_millis(16));
    }
    eprintln!(
        "spin_and_collect: wall={}ms ticks={} seen={}",
        wall.as_millis(),
        ticks,
        seen.len()
    );
    seen
}

// ----- Scenarios -----

fn scenario_basic_playback(ctx: &egui::Context, backend: &WgpuBackend, path: &Path) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => return ScenarioResult::fail("basic_playback", format!("open: {}", e)),
        };
    player.set_volume(0.0);
    player.play();
    let seen = spin_and_collect(&mut player, Duration::from_secs(3));
    let elapsed = player.elapsed_ms();
    let distinct = seen.len();
    let final_pts_us = seen.last().copied().unwrap_or(i64::MIN);
    let details = format!(
        "distinct={} elapsed={}ms final_pts_us={} state={:?} audio={} latency={}us",
        distinct,
        elapsed,
        final_pts_us,
        player.state(),
        player.audio_backend(),
        player.audio_output_latency_us()
    );
    if distinct >= 30 && final_pts_us >= 1_500_000 {
        ScenarioResult::ok("basic_playback", details)
    } else {
        ScenarioResult::fail("basic_playback", details)
    }
}

fn scenario_seek_while_playing(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => return ScenarioResult::fail("seek_while_playing", format!("open: {}", e)),
        };
    player.set_volume(0.0);
    player.play();
    let pre = spin_and_collect(&mut player, Duration::from_millis(600));
    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("seek_while_playing", "zero duration");
    }
    let target_frac = 0.5_f32;
    let target_us = (duration_us as f32 * target_frac) as i64;
    player.seek(target_frac);
    let seen = spin_and_collect(&mut player, Duration::from_millis(2500));
    let first_post = seen
        .iter()
        .find(|p| (**p - target_us).abs() < duration_us / 3)
        .copied();
    let sample: Vec<i64> = seen.iter().step_by((seen.len() / 10).max(1)).copied().collect();
    let details = format!(
        "target_us={} first_post_seek_pts={:?} pre_seen={} post_seen={} post_sample={:?}",
        target_us,
        first_post,
        pre.len(),
        seen.len(),
        sample
    );
    match first_post {
        Some(p) if (p - target_us).abs() <= PTS_TOLERANCE_US => {
            ScenarioResult::ok("seek_while_playing", details)
        }
        _ => ScenarioResult::fail("seek_while_playing", details),
    }
}

fn scenario_seek_while_paused(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => return ScenarioResult::fail("seek_while_paused", format!("open: {}", e)),
        };
    player.set_volume(0.0);
    // Do NOT call play(). Player is in Paused state after open.
    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("seek_while_paused", "zero duration");
    }
    let target_frac = 0.3_f32;
    let target_us = (duration_us as f32 * target_frac) as i64;
    player.seek(target_frac);
    // Give the priming phase enough wall time.
    let seen = spin_and_collect(&mut player, Duration::from_millis(2000));
    let final_pts = seen.last().copied().unwrap_or(i64::MIN);
    let details = format!(
        "target_us={} final_pts={} seen={}",
        target_us,
        final_pts,
        seen.len()
    );
    if final_pts != i64::MIN && (final_pts - target_us).abs() <= PTS_TOLERANCE_US {
        ScenarioResult::ok("seek_while_paused", details)
    } else {
        ScenarioResult::fail("seek_while_paused", details)
    }
}

/// Regression test: play for a moment, pause, seek, then resume. Real
/// GUI flow. Must land on the seek target without fast-forwarding,
/// and playback must actually resume (state transitions from Paused
/// to Playing, uploaded pts continues past the seek target).
fn scenario_play_pause_seek_resume(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail("play_pause_seek_resume", format!("open: {}", e))
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(500));
    player.pause();
    // Give the pause a moment to actually quiesce the audio stream.
    let _ = spin_and_collect(&mut player, Duration::from_millis(200));
    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("play_pause_seek_resume", "zero duration");
    }
    let target_frac = 0.5_f32;
    let target_us = (duration_us as f32 * target_frac) as i64;
    player.seek(target_frac);
    // Before resuming, verify the seek landed on a frame at/near the
    // target (not a fast-forward from before).
    let paused_post_seek = spin_and_collect(&mut player, Duration::from_millis(800));
    let landed = paused_post_seek
        .iter()
        .find(|p| (**p - target_us).abs() <= PTS_TOLERANCE_US)
        .copied();
    // Now resume.
    player.play();
    let resumed = spin_and_collect(&mut player, Duration::from_millis(1200));
    let final_pts = resumed.last().copied().unwrap_or(i64::MIN);
    let paused_sample: Vec<i64> = paused_post_seek.iter().copied().collect();
    let resumed_sample: Vec<i64> = resumed
        .iter()
        .step_by((resumed.len() / 10).max(1))
        .copied()
        .collect();
    let details = format!(
        "target_us={} landed={:?} paused_pts={:?} resumed_count={} resumed_sample={:?} final_pts={} state={:?}",
        target_us, landed, paused_sample, resumed.len(), resumed_sample, final_pts, player.state()
    );
    let landed_ok =
        landed.map(|p| (p - target_us).abs() <= PTS_TOLERANCE_US).unwrap_or(false);
    let resumed_ok = resumed.len() >= 10
        && final_pts > target_us + 200_000
        && matches!(player.state(), PlayerState::Playing);
    if landed_ok && resumed_ok {
        ScenarioResult::ok("play_pause_seek_resume", details)
    } else {
        ScenarioResult::fail("play_pause_seek_resume", details)
    }
}

/// Regression test for the frame-step keybinds (comma / period).
/// Pause, then call `step_frames(1)` repeatedly — each call should
/// advance `uploaded_pts` by roughly one frame interval, not stay
/// stuck on the starting frame.
fn scenario_paused_frame_step(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail("paused_frame_step", format!("open: {}", e))
            }
        };
    player.set_volume(0.0);
    player.play();
    // Warm up so the player is past the first-frame state.
    let _ = spin_and_collect(&mut player, Duration::from_millis(400));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    // Seek to a known anchor so successive step-forward results
    // are predictable.
    player.seek(0.50);
    let _ = spin_and_collect(&mut player, Duration::from_millis(1500));
    let start_pts = player.uploaded_pts_us_for_test();

    // Hammer step-forward a lot — on long-GOP HEVC the first few
    // steps drain the buffer the decoder filled during priming, so
    // we also need the demuxer to keep topping up the queue during
    // pause for this to keep working past ~16 steps.
    let mut landed: Vec<i64> = vec![start_pts];
    for _ in 0..20 {
        player.step_frames(1);
        let _ = spin_and_collect(&mut player, Duration::from_millis(400));
        landed.push(player.uploaded_pts_us_for_test());
    }
    let unique: std::collections::BTreeSet<i64> = landed.iter().copied().collect();
    let details = format!("pts_sequence={:?}", landed);
    let monotonic = landed.windows(2).all(|w| w[1] >= w[0]);
    let all_distinct = unique.len() == landed.len();
    if monotonic && all_distinct {
        ScenarioResult::ok("paused_frame_step", details)
    } else {
        ScenarioResult::fail("paused_frame_step", details)
    }
}

/// Gated version of `paused_frame_step`: same check but only ticks
/// when egui has requested a repaint.
fn scenario_paused_frame_step_gated(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "paused_frame_step_gated",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(400));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    player.seek(0.50);
    let _ = gated_spin_and_collect(ctx, &mut player, Duration::from_millis(1500));
    let start_pts = player.uploaded_pts_us_for_test();

    let mut landed: Vec<i64> = vec![start_pts];
    for _ in 0..20 {
        player.step_frames(1);
        let _ = gated_spin_and_collect(ctx, &mut player, Duration::from_millis(400));
        landed.push(player.uploaded_pts_us_for_test());
    }
    let unique: std::collections::BTreeSet<i64> = landed.iter().copied().collect();
    let details = format!("pts_sequence={:?}", landed);
    let monotonic = landed.windows(2).all(|w| w[1] >= w[0]);
    let all_distinct = unique.len() == landed.len();
    if monotonic && all_distinct {
        ScenarioResult::ok("paused_frame_step_gated", details)
    } else {
        ScenarioResult::fail("paused_frame_step_gated", details)
    }
}

fn scenario_rapid_seek(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => return ScenarioResult::fail("rapid_seek", format!("open: {}", e)),
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(400));
    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("rapid_seek", "zero duration");
    }
    // Blast many seek commands back-to-back. The final one should win.
    for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.4, 0.6] {
        player.seek(frac);
    }
    let final_target_us = (duration_us as f32 * 0.6_f32) as i64;
    let seen = spin_and_collect(&mut player, Duration::from_millis(2500));
    // The FIRST frame that lands in the final-target window is what
    // proves the seek coalesced correctly and landed accurately.
    // Later frames just reflect subsequent playback drift.
    let first_near_target = seen
        .iter()
        .find(|p| (**p - final_target_us).abs() < duration_us / 5)
        .copied();
    let sample: Vec<i64> = seen
        .iter()
        .step_by((seen.len() / 12).max(1))
        .copied()
        .collect();
    let first = seen.first().copied().unwrap_or(i64::MIN);
    let last = seen.last().copied().unwrap_or(i64::MIN);
    let details = format!(
        "final_target_us={} first_near_target={:?} seen={} first={} last={} sample={:?}",
        final_target_us, first_near_target, seen.len(), first, last, sample
    );
    match first_near_target {
        Some(p) if (p - final_target_us).abs() <= PTS_TOLERANCE_US => {
            ScenarioResult::ok("rapid_seek", details)
        }
        _ => ScenarioResult::fail("rapid_seek", details),
    }
}

/// Regression test for the GUI path: looping=true (as the GUI sets it
/// by default). On a short, fast-decoding file the demuxer races to
/// EOF in ~200 ms, and without the "wait for playback to catch up"
/// gate in the EOF handler the clock keeps getting reset to 0 and
/// playback appears frozen on the first frame.
fn scenario_looping_short_file(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => return ScenarioResult::fail("looping_short_file", format!("open: {}", e)),
        };
    player.set_volume(0.0);
    player.set_looping(true);
    player.play();
    let seen = spin_and_collect(&mut player, Duration::from_secs(3));
    let elapsed = player.elapsed_ms();
    let distinct = seen.len();
    let final_pts_us = seen.last().copied().unwrap_or(i64::MIN);
    let details = format!(
        "distinct={} elapsed={}ms final_pts_us={} state={:?}",
        distinct, elapsed, final_pts_us, player.state()
    );
    // Must make real forward progress even with looping enabled.
    if distinct >= 30 && final_pts_us >= 1_500_000 {
        ScenarioResult::ok("looping_short_file", details)
    } else {
        ScenarioResult::fail("looping_short_file", details)
    }
}

// =========================================================================
// End-of-file behaviour: the tail must play, and the loop must restart
// at real speed
// =========================================================================

/// Play the last stretch of the file and check that the final frames
/// actually reach the screen.
///
/// A frame-threaded decoder holds several finished frames internally
/// and only releases them when told the stream has ended. Without that
/// flush the demuxer hit EOF, the buffered frames were dropped, and the
/// video ended early — the "sometimes it cuts the video short" report.
fn scenario_tail_is_not_cut_short(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    const NAME: &str = "tail_is_not_cut_short";
    let mut player = match Player::open_with_backend(ctx, path, Some(backend.clone())) {
        Ok(p) => p,
        Err(e) => return ScenarioResult::fail("tail_is_not_cut_short", format!("open: {}", e)),
    };
    let _ = NAME;
    player.set_volume(0.0);
    player.set_looping(false);

    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("tail_is_not_cut_short", "file reports no duration");
    }
    let frame_us = player.frame_interval_us();

    // Start ~1.5 s from the end so the test doesn't depend on how long
    // the whole file is.
    let start_us = (duration_us - 1_500_000).max(0);
    player.seek_us(start_us);
    player.play();

    let seen = spin_and_collect(&mut player, Duration::from_millis(3500));
    let last_pts = seen.iter().copied().max().unwrap_or(i64::MIN);
    let shortfall_us = duration_us - last_pts;

    let details = format!(
        "duration={}us last_pts={}us shortfall={}ms ({:.1} frames) distinct={}",
        duration_us,
        last_pts,
        shortfall_us / 1000,
        shortfall_us as f64 / frame_us as f64,
        seen.len(),
    );

    // The final frame's PTS sits one frame interval before the
    // duration, so allow three intervals of slack for container
    // rounding — but no more. The bug this guards against dropped
    // whole thread-count batches of frames.
    if shortfall_us <= frame_us * 3 {
        ScenarioResult::ok("tail_is_not_cut_short", details)
    } else {
        ScenarioResult::fail("tail_is_not_cut_short", details)
    }
}

/// After a loop wraps, playback must resume at ordinary speed.
///
/// This is the "loops the last second over and over" regression. On a
/// file with no audio track the wrap reset the shared clock to zero,
/// but `tick()` immediately overwrote it from a wall-clock origin
/// captured when playback began — still out at the end of the file —
/// so every freshly decoded frame looked overdue and the player raced
/// through the replay as fast as it could decode.
fn scenario_loop_restarts_at_real_speed(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player = match Player::open_with_backend(ctx, path, Some(backend.clone())) {
        Ok(p) => p,
        Err(e) => {
            return ScenarioResult::fail(
                "loop_restarts_at_real_speed",
                format!("open: {}", e),
            )
        }
    };
    player.set_volume(0.0);
    player.set_looping(true);

    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("loop_restarts_at_real_speed", "file reports no duration");
    }

    // Park near the end so the wrap happens almost immediately.
    player.seek_us((duration_us - 700_000).max(0));
    player.play();

    // Ride through the wrap.
    let before = spin_and_collect(&mut player, Duration::from_millis(1600));
    let wrapped = before.windows(2).any(|w| w[1] < w[0]);
    if !wrapped {
        return ScenarioResult::fail(
            "loop_restarts_at_real_speed",
            format!(
                "never wrapped: duration={}us pts range {:?}..{:?}",
                duration_us,
                before.first(),
                before.last()
            ),
        );
    }

    // Now measure the pace of the new pass. One second of wall time
    // should advance the position by about one second of video.
    let wall = Duration::from_millis(1200);
    let after = spin_and_collect(&mut player, wall);
    if after.len() < 2 {
        return ScenarioResult::fail(
            "loop_restarts_at_real_speed",
            format!("only {} frames after wrap", after.len()),
        );
    }
    let lo = after.iter().copied().min().unwrap_or(0);
    let hi = after.iter().copied().max().unwrap_or(0);
    let span_us = hi - lo;
    let wall_us = wall.as_micros() as i64;
    let ratio = span_us as f64 / wall_us as f64;

    let details = format!(
        "post-wrap span={}ms over {}ms wall (ratio {:.2}), frames={}",
        span_us / 1000,
        wall_us / 1000,
        ratio,
        after.len()
    );

    // Real-time playback is a ratio near 1.0. The runaway bug produced
    // several complete passes per second, so anything past 2.0 is the
    // failure. A wrap landing inside the measurement window can also
    // drag the span down, so only the upper bound is asserted.
    if ratio <= 2.0 {
        ScenarioResult::ok("loop_restarts_at_real_speed", details)
    } else {
        ScenarioResult::fail("loop_restarts_at_real_speed", details)
    }
}

/// An A-B clip loop must keep playback inside the marked range and
/// repeat it, rather than running on to the end of the file.
fn scenario_clip_loop_stays_in_range(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player = match Player::open_with_backend(ctx, path, Some(backend.clone())) {
        Ok(p) => p,
        Err(e) => {
            return ScenarioResult::fail(
                "clip_loop_stays_in_range",
                format!("open: {}", e),
            )
        }
    };
    player.set_volume(0.0);
    player.set_looping(true);

    let duration_us = player.duration_ms() * 1000;
    if duration_us < 2_000_000 {
        return ScenarioResult::fail(
            "clip_loop_stays_in_range",
            format!("file too short to clip ({}us)", duration_us),
        );
    }
    let frame_us = player.frame_interval_us();

    // A one-second window starting a quarter of the way in.
    let a_us = duration_us / 4;
    let b_us = a_us + 1_000_000;
    player.set_loop_range(Some(a_us), Some(b_us));
    player.seek_us(a_us);
    player.play();

    let seen = spin_and_collect(&mut player, Duration::from_millis(4000));
    if seen.len() < 10 {
        return ScenarioResult::fail(
            "clip_loop_stays_in_range",
            format!("only {} frames observed", seen.len()),
        );
    }

    // Seeking lands on the first frame at or after the in-point, and
    // the out-point check runs against the display clock, so allow a
    // couple of frames of slack at each end.
    let slack = frame_us * 3;
    let strays: Vec<i64> = seen
        .iter()
        .copied()
        .filter(|&pts| pts < a_us - slack || pts > b_us + slack)
        .collect();
    let wraps = seen.windows(2).filter(|w| w[1] < w[0]).count();

    let details = format!(
        "range {}..{}us, frames={} wraps={} strays={} {:?}",
        a_us,
        b_us,
        seen.len(),
        wraps,
        strays.len(),
        strays.iter().take(4).collect::<Vec<_>>(),
    );

    // Four seconds over a one-second clip should wrap about three
    // times; require at least two so the check can't pass on a single
    // straight-through play.
    if strays.is_empty() && wraps >= 2 {
        ScenarioResult::ok("clip_loop_stays_in_range", details)
    } else {
        ScenarioResult::fail("clip_loop_stays_in_range", details)
    }
}

/// Clearing the clip markers must hand the whole file back.
fn scenario_clip_loop_clears(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player = match Player::open_with_backend(ctx, path, Some(backend.clone())) {
        Ok(p) => p,
        Err(e) => return ScenarioResult::fail("clip_loop_clears", format!("open: {}", e)),
    };
    player.set_volume(0.0);
    player.set_looping(true);

    let duration_us = player.duration_ms() * 1000;
    if duration_us < 2_000_000 {
        return ScenarioResult::fail("clip_loop_clears", "file too short");
    }

    let a_us = duration_us / 4;
    let b_us = a_us + 600_000;
    player.set_loop_range(Some(a_us), Some(b_us));
    assert_eq!(player.loop_range(), (Some(a_us), Some(b_us)));
    player.seek_us(a_us);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(900));

    player.set_loop_range(None, None);
    if player.loop_range() != (None, None) {
        return ScenarioResult::fail("clip_loop_clears", "markers survived the clear");
    }

    let seen = spin_and_collect(&mut player, Duration::from_millis(1600));
    let hi = seen.iter().copied().max().unwrap_or(i64::MIN);
    let details = format!(
        "cleared at {}us, reached {}us over {} frames",
        b_us,
        hi,
        seen.len()
    );

    // Without the out-point, playback must run past where it used to
    // wrap.
    if hi > b_us {
        ScenarioResult::ok("clip_loop_clears", details)
    } else {
        ScenarioResult::fail("clip_loop_clears", details)
    }
}

// =========================================================================
// A/V sync while moving around the video
// =========================================================================

/// Seconds of content time per unit of sample value in the sync fixture.
/// The fixture's audio is a ramp, `value = t / 20`, so a sample of 0.25
/// was recorded at t = 5 s.
const RAMP_SECONDS_PER_UNIT: f64 = 20.0;

/// The content time audible right now, in milliseconds, read from the
/// ramp: the first sample of the latest output buffer, advanced by the
/// time elapsed since it reached the listener.
fn audible_ms(player: &Player) -> Option<f64> {
    let (sample, heard_at) = player.first_played_sample_for_test()?;
    let now = Instant::now();
    let since_ms = if now >= heard_at {
        (now - heard_at).as_secs_f64() * 1000.0
    } else {
        -((heard_at - now).as_secs_f64() * 1000.0)
    };
    Some(sample as f64 * RAMP_SECONDS_PER_UNIT * 1000.0 + since_ms)
}

/// Sound-minus-picture offset right now, in milliseconds; positive means
/// the sound is ahead.
///
/// Measured against the middle of the frame on screen. A frame is shown
/// for its whole duration, so with perfect sync the audible moment sits
/// anywhere inside it and this reads within ±half a frame, averaging
/// zero; comparing against the frame's start instead would read half a
/// frame high even when nothing is wrong.
fn av_offset_ms(player: &Player) -> Option<f64> {
    let audible = audible_ms(player)?;
    let video_us = player.uploaded_pts_us_for_test();
    if video_us == i64::MIN {
        return None;
    }
    let frame_ms = player.frame_interval_us() as f64 / 1000.0;
    Some(audible - (video_us as f64 / 1000.0 + frame_ms / 2.0))
}

/// Let playback settle after an action, then sample the offset for a
/// while and return the median (robust to the odd tick that lands
/// between an audio callback and a frame upload).
fn settled_offset_ms(player: &mut Player) -> Option<f64> {
    let _ = spin_and_collect(player, Duration::from_millis(700));
    let mut samples: Vec<f64> = Vec::new();
    let end = Instant::now() + Duration::from_millis(600);
    while Instant::now() < end {
        tick_and_flush(player);
        if let Some(o) = av_offset_ms(player) {
            samples.push(o);
        }
        std::thread::sleep(Duration::from_millis(16));
    }
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(samples[samples.len() / 2])
}

/// Audio must stay locked to the picture through everything a viewer
/// does while moving around a video: seeks while playing, arrow-key
/// jumps, seeks while paused, frame steps, and A-B clip wraps.
///
/// This needs the `av_sync_ramp.mkv` fixture, whose audio encodes its
/// own timestamp so the audible position can be read back independently
/// of the player's clock. Generate it with:
///
///     ffmpeg -f lavfi -i "testsrc2=size=640x360:rate=30:duration=10" \
///            -f lavfi -i "aevalsrc=exprs='t/20|t/20':s=48000:d=10" \
///            -c:v libx264 -pix_fmt yuv420p -g 75 -keyint_min 75 \
///            -sc_threshold 0 -c:a pcm_s16le av_sync_ramp.mkv
///
/// Keyframes every 2.5 s mean most seeks land well inside a GOP, which
/// is the case that exposed the original bug: video skipped ahead to
/// the seek target while audio started from the preceding keyframe.
fn scenario_av_sync_while_moving_around(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    const NAME: &str = "av_sync_while_moving_around";
    let is_fixture = path
        .file_name()
        .map(|n| n.to_string_lossy().contains("av_sync_ramp"))
        .unwrap_or(false);
    if !is_fixture {
        return ScenarioResult::ok(NAME, "SKIPPED (run against av_sync_ramp.mkv)");
    }

    let mut player = match Player::open_with_backend(ctx, path, Some(backend.clone())) {
        Ok(p) => p,
        Err(e) => return ScenarioResult::fail(NAME, format!("open: {}", e)),
    };
    if !player.has_audio() {
        return ScenarioResult::fail(NAME, "fixture has no audio stream");
    }
    player.set_volume(0.0);
    player.set_looping(true);
    let duration_us = player.duration_ms() * 1000;
    let at = |frac: f64| (duration_us as f64 * frac) as i64;

    let mut rows: Vec<(String, Option<f64>)> = Vec::new();
    let mut record = |label: &str, player: &mut Player| {
        let o = settled_offset_ms(player);
        eprintln!("    {:<34} offset {}", label, match o {
            Some(v) => format!("{:+.0} ms", v),
            None => "n/a".to_string(),
        });
        rows.push((label.to_string(), o));
    };

    player.play();
    record("play from start", &mut player);

    player.seek_us(at(0.55));
    record("seek to 55% while playing", &mut player);

    player.seek_us(at(0.20));
    record("seek to 20% while playing", &mut player);

    let now = player.elapsed_ms() * 1000;
    player.seek_us(now + 3_000_000);
    record("jump +3 s", &mut player);

    let now = player.elapsed_ms() * 1000;
    player.seek_us(now - 3_000_000);
    record("jump -3 s", &mut player);

    player.pause();
    player.seek_us(at(0.70));
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));
    player.play();
    record("paused seek to 70%, then play", &mut player);

    player.pause();
    for frac in [0.30, 0.45, 0.35, 0.62] {
        player.seek_us(at(frac));
        let _ = spin_and_collect(&mut player, Duration::from_millis(150));
    }
    player.play();
    record("4 paused seeks, then play", &mut player);

    player.pause();
    for _ in 0..15 {
        player.step_frames(1);
        let _ = spin_and_collect(&mut player, Duration::from_millis(30));
    }
    player.play();
    record("15 frame steps forward, then play", &mut player);

    player.pause();
    for _ in 0..5 {
        player.step_frames(-1);
        let _ = spin_and_collect(&mut player, Duration::from_millis(30));
    }
    player.play();
    record("5 frame steps back, then play", &mut player);

    // A-B clip over a keyframe-free stretch, so every wrap is a
    // mid-GOP seek.
    let a = at(0.31);
    let b = a + 1_300_000;
    player.set_loop_range(Some(a), Some(b));
    player.seek_us(a);
    record("clip loop, first pass", &mut player);
    let _ = spin_and_collect(&mut player, Duration::from_millis(900));
    record("clip loop, after wraps", &mut player);
    player.set_loop_range(None, None);

    // Tolerance on each action's median offset. Individual readings
    // swing ±half a frame (±17 ms at 30 fps) as the audible moment moves
    // through the frame on screen, but the median of perfectly synced
    // playback sits within a few ms of zero. Uncompensated output
    // latency alone reads -40 ms on USB speakers; the seek bug this
    // started from read hundreds of ms to seconds.
    const TOLERANCE_MS: f64 = 12.0;
    let bad: Vec<&(String, Option<f64>)> = rows
        .iter()
        .filter(|(_, o)| o.map(|v| v.abs() > TOLERANCE_MS).unwrap_or(true))
        .collect();
    let worst = rows
        .iter()
        .filter_map(|(_, o)| *o)
        .fold(0.0f64, |m, v| if v.abs() > m.abs() { v } else { m });

    let details = format!(
        "{} actions, worst offset {:+.0} ms, {} outside ±{} ms{}",
        rows.len(),
        worst,
        bad.len(),
        TOLERANCE_MS,
        if bad.is_empty() {
            String::new()
        } else {
            format!(
                ": {}",
                bad.iter()
                    .map(|(l, o)| format!(
                        "[{} {}]",
                        l,
                        o.map(|v| format!("{:+.0}ms", v)).unwrap_or("n/a".into())
                    ))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        }
    );
    if bad.is_empty() {
        ScenarioResult::ok(NAME, details)
    } else {
        ScenarioResult::fail(NAME, details)
    }
}

/// Audio must stay locked to the picture across the end-of-file loop
/// wrap, and must actually resume on the new pass rather than falling
/// silent. Needs the `av_sync_ramp` fixture (see
/// `scenario_av_sync_while_moving_around`).
fn scenario_av_sync_across_loop_wrap(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    const NAME: &str = "av_sync_across_loop_wrap";
    let is_fixture = path
        .file_name()
        .map(|n| n.to_string_lossy().contains("av_sync_ramp"))
        .unwrap_or(false);
    if !is_fixture {
        return ScenarioResult::ok(NAME, "SKIPPED (run against av_sync_ramp.mkv)");
    }
    let mut player = match Player::open_with_backend(ctx, path, Some(backend.clone())) {
        Ok(p) => p,
        Err(e) => return ScenarioResult::fail(NAME, format!("open: {}", e)),
    };
    player.set_volume(0.0);
    player.set_looping(true);
    let duration_us = player.duration_ms() * 1000;

    player.seek_us(duration_us - 1_000_000);
    player.play();

    // Sample through the wrap and well into the second pass.
    let verbose = std::env::var("VIDEO_TEST_VERBOSE").is_ok();
    let start = Instant::now();
    let mut wrapped_at: Option<Duration> = None;
    let mut last_video = i64::MIN;
    let mut after_wrap: Vec<f64> = Vec::new();
    while start.elapsed() < Duration::from_millis(3200) {
        tick_and_flush(&mut player);
        let video = player.uploaded_pts_us_for_test();
        if wrapped_at.is_none() && last_video != i64::MIN && video < last_video - 1_000_000 {
            wrapped_at = Some(start.elapsed());
        }
        last_video = video;
        let offset = av_offset_ms(&player);
        if verbose {
            eprintln!(
                "    t={:>5}ms clock={:>6}ms frozen={} video={:>8}us audio={:>8.0}ms offset={}",
                start.elapsed().as_millis(),
                player.elapsed_ms(),
                player.is_seeking() as u8,
                video,
                audible_ms(&player).unwrap_or(f64::NAN),
                offset.map(|o| format!("{:+.0}ms", o)).unwrap_or("n/a".into())
            );
        }
        // Judge the second pass once it has had half a second to settle.
        if let (Some(w), Some(o)) = (wrapped_at, offset) {
            if start.elapsed() > w + Duration::from_millis(500) {
                after_wrap.push(o);
            }
        }
        std::thread::sleep(Duration::from_millis(16));
    }

    let Some(w) = wrapped_at else {
        return ScenarioResult::fail(NAME, "never wrapped");
    };
    if after_wrap.is_empty() {
        return ScenarioResult::fail(NAME, "no readings after the wrap");
    }
    after_wrap.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = after_wrap[after_wrap.len() / 2];
    let details = format!(
        "wrapped at {}ms; second-pass median offset {:+.0} ms over {} readings",
        w.as_millis(),
        median,
        after_wrap.len()
    );
    if median.abs() <= 12.0 {
        ScenarioResult::ok(NAME, details)
    } else {
        ScenarioResult::fail(NAME, details)
    }
}

/// If the audio device disappears for good mid-playback (unplugged, and
/// nothing to reopen on), the picture must keep playing from the wall
/// clock instead of freezing on the last frame the audio clock reached.
fn scenario_survives_audio_output_loss(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    const NAME: &str = "survives_audio_output_loss";
    let mut player = match Player::open_with_backend(ctx, path, Some(backend.clone())) {
        Ok(p) => p,
        Err(e) => return ScenarioResult::fail(NAME, format!("open: {}", e)),
    };
    if player.audio_backend() == "none" {
        return ScenarioResult::ok(NAME, "SKIPPED (no audio output in use)");
    }
    player.set_volume(0.0);
    player.set_looping(false);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(1000));

    player.simulate_audio_output_loss_for_test();
    let wall = Duration::from_millis(1500);
    let after = spin_and_collect(&mut player, wall);
    let span_us = match (after.first(), after.last()) {
        (Some(a), Some(b)) => b - a,
        _ => 0,
    };
    let ratio = span_us as f64 / wall.as_micros() as f64;
    let details = format!(
        "after loss: {} frames spanning {} ms over {} ms (ratio {:.2})",
        after.len(),
        span_us / 1000,
        wall.as_millis(),
        ratio
    );
    // Real-time progress, give or take the first frame's alignment.
    if after.len() >= 20 && (0.8..=1.2).contains(&ratio) {
        ScenarioResult::ok(NAME, details)
    } else {
        ScenarioResult::fail(NAME, details)
    }
}

fn scenario_no_first_frame_stuck(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => return ScenarioResult::fail("no_first_frame_stuck", format!("open: {}", e)),
        };
    player.set_volume(0.0);
    player.play();
    let seen = spin_and_collect(&mut player, Duration::from_secs(2));
    let distinct = seen.len();
    let first = seen.first().copied().unwrap_or(i64::MIN);
    let last = seen.last().copied().unwrap_or(i64::MIN);
    let details = format!(
        "distinct={} first={} last={} span_ms={}",
        distinct,
        first,
        last,
        (last - first) / 1000
    );
    // Specifically a regression check for "plays first frame and then
    // nothing more": we must observe many distinct frames AND the span
    // between first and last uploaded pts must cover a meaningful
    // fraction of wall time.
    if distinct >= 30 && (last - first) >= 1_000_000 {
        ScenarioResult::ok("no_first_frame_stuck", details)
    } else {
        ScenarioResult::fail("no_first_frame_stuck", details)
    }
}

// ============================================================================
// Aggressive "VLC-parity" scenarios
// ============================================================================
//
// These scenarios simulate the user-reported bug: the display frame
// does not update when seeking (or frame-stepping) multiple times in
// rapid succession at random positions in the video. Unlike the
// earlier scenarios which seek to 3-5 known targets with generous
// settle windows, these fire 30+ operations in quick succession and
// verify that EVERY one of them resolves to a new, distinct frame at
// or near the requested target — exactly like VLC does.

/// Deterministic LCG so random-seek scenarios are reproducible across
/// runs. Mixes the 64-bit state with a standard MCG64 constant.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform float in [0.05, 0.95] so we stay inside the file.
    fn next_frac(&mut self) -> f32 {
        let r = self.next_u32() as f32 / (u32::MAX / 2) as f32;
        0.05 + (r * 0.45).clamp(0.0, 0.9)
    }
}

/// Wait up to `timeout` for the player's uploaded pts to land within
/// `PTS_TOLERANCE_US` of `target_us`. Returns (landed_pts, elapsed_ms).
/// Ticks the player ~every 4 ms so the check runs at ~60 Hz (the
/// actual tick budget egui would give the GUI).
fn wait_for_seek_target(
    player: &mut Player,
    target_us: i64,
    timeout: Duration,
) -> (i64, u64) {
    let start = Instant::now();
    while start.elapsed() < timeout {
        tick_and_flush(player);
        let pts = player.uploaded_pts_us_for_test();
        if pts != i64::MIN && (pts - target_us).abs() <= PTS_TOLERANCE_US {
            return (pts, start.elapsed().as_millis() as u64);
        }
        std::thread::sleep(Duration::from_millis(4));
    }
    (player.uploaded_pts_us_for_test(), timeout.as_millis() as u64)
}

/// Wait up to `timeout` for `uploaded_pts` to change from `prev`. Used
/// for frame-step tests where we don't know the exact target pts.
/// Returns the new pts (or `prev` if nothing changed).
fn wait_for_new_pts(player: &mut Player, prev: i64, timeout: Duration) -> i64 {
    let start = Instant::now();
    while start.elapsed() < timeout {
        tick_and_flush(player);
        let pts = player.uploaded_pts_us_for_test();
        if pts != prev && pts != i64::MIN {
            return pts;
        }
        std::thread::sleep(Duration::from_millis(4));
    }
    player.uploaded_pts_us_for_test()
}

/// VLC-parity: pause, then spam 40 random-position seeks with only a
/// short settle window between each. Every single seek must resolve
/// to a distinct frame at or near its target — if any one of them
/// leaves the display stuck, the test fails with a detailed report.
fn scenario_paused_random_spam_seek(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "paused_random_spam_seek",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("paused_random_spam_seek", "zero duration");
    }

    let mut rng = Lcg::new(0xCAFEBABE);
    let mut failures: Vec<(usize, f32, i64, i64, u64)> = Vec::new();
    let mut landed: Vec<i64> = Vec::new();
    const N: usize = 40;
    for i in 0..N {
        let frac = rng.next_frac();
        let target_us = (duration_us as f32 * frac) as i64;
        player.seek(frac);
        let (pts, ms) = wait_for_seek_target(
            &mut player,
            target_us,
            Duration::from_millis(1500),
        );
        if pts == i64::MIN || (pts - target_us).abs() > PTS_TOLERANCE_US {
            failures.push((i, frac, target_us, pts, ms));
        } else {
            landed.push(pts);
        }
    }

    // Count how often a pts value repeats. Two random targets can
    // legitimately land on the same displayed frame when the
    // video's GOP structure puts multiple targets in the same
    // reachable-keyframe bucket — `landed == target ± tolerance`
    // is what the bug check cares about, not strict uniqueness.
    // A large dup cluster (one pts repeated many times) is what
    // indicates the "stuck frame" bug; a handful of coincidental
    // dups on random targets is fine.
    let unique: std::collections::BTreeSet<i64> = landed.iter().copied().collect();
    let mut max_repeat = 0usize;
    for u in &unique {
        let c = landed.iter().filter(|&&p| p == *u).count();
        if c > max_repeat {
            max_repeat = c;
        }
    }
    let details = format!(
        "landed_ok={}/{} unique_landed={} max_repeat={} first_failures={:?}",
        landed.len(),
        N,
        unique.len(),
        max_repeat,
        failures.iter().take(5).collect::<Vec<_>>()
    );
    // Accept up to a 3-way tie on random targets. A stuck-frame
    // bug would have max_repeat close to N (every seek lands on
    // the same stale pts).
    if failures.is_empty() && max_repeat <= 3 {
        ScenarioResult::ok("paused_random_spam_seek", details)
    } else {
        ScenarioResult::fail("paused_random_spam_seek", details)
    }
}

/// Same as `paused_random_spam_seek` but while playing. VLC would
/// resume playback from each new seek target without any frame
/// getting stuck.
fn scenario_playing_random_spam_seek(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "playing_random_spam_seek",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));

    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("playing_random_spam_seek", "zero duration");
    }

    let mut rng = Lcg::new(0xDEADBEEF);
    let mut failures: Vec<(usize, f32, i64, i64, u64)> = Vec::new();
    const N: usize = 30;
    for i in 0..N {
        let frac = rng.next_frac();
        let target_us = (duration_us as f32 * frac) as i64;
        player.seek(frac);
        let (pts, ms) = wait_for_seek_target(
            &mut player,
            target_us,
            Duration::from_millis(1500),
        );
        if pts == i64::MIN || (pts - target_us).abs() > PTS_TOLERANCE_US {
            failures.push((i, frac, target_us, pts, ms));
        }
    }

    let details = format!(
        "failures={}/{} first_failures={:?}",
        failures.len(),
        N,
        failures.iter().take(5).collect::<Vec<_>>()
    );
    if failures.is_empty() {
        ScenarioResult::ok("playing_random_spam_seek", details)
    } else {
        ScenarioResult::fail("playing_random_spam_seek", details)
    }
}

/// Burst seeks with zero wall time between them: simulates a user
/// mashing the left/right arrow keys before any decode catches up.
/// The LAST seek must always land — coalescing is allowed and
/// expected (VLC does the same), but the display must eventually
/// show the last requested frame.
fn scenario_burst_coalesce_seek(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "burst_coalesce_seek",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("burst_coalesce_seek", "zero duration");
    }

    // Run a bunch of independent bursts. Each burst fires M seeks with
    // no wall time in between, then waits for the LAST target to land.
    // Fail loudly if any burst fails to surface its last target.
    let mut rng = Lcg::new(0x1234_5678_9ABC_DEF0);
    let bursts: [usize; 6] = [3, 5, 8, 2, 10, 4];
    let mut failed: Vec<(usize, f32, i64, i64)> = Vec::new();
    for (burst_idx, &m) in bursts.iter().enumerate() {
        let mut last_frac = 0.5f32;
        for _ in 0..m {
            last_frac = rng.next_frac();
            player.seek(last_frac);
            // NO sleep / NO tick between seeks — simulate key mashing.
        }
        let last_target = (duration_us as f32 * last_frac) as i64;
        let (pts, _ms) = wait_for_seek_target(
            &mut player,
            last_target,
            Duration::from_millis(2000),
        );
        if (pts - last_target).abs() > PTS_TOLERANCE_US {
            failed.push((burst_idx, last_frac, last_target, pts));
        }
    }

    let details = format!(
        "failed_bursts={}/{} detail={:?}",
        failed.len(),
        bursts.len(),
        failed
    );
    if failed.is_empty() {
        ScenarioResult::ok("burst_coalesce_seek", details)
    } else {
        ScenarioResult::fail("burst_coalesce_seek", details)
    }
}

/// Long run of frame-forward steps. Step 60 times and verify each
/// step produced a distinct, monotonically-advancing frame. If any
/// step silently repeats the previous frame, we reproduce the user's
/// "display doesn't update when I hold period" complaint.
fn scenario_frame_step_forward_long_run(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "frame_step_forward_long_run",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    player.seek(0.30);
    let _ = spin_and_collect(&mut player, Duration::from_millis(1500));
    let start_pts = player.uploaded_pts_us_for_test();

    let mut pts_seq: Vec<i64> = vec![start_pts];
    let mut stuck: Vec<(usize, i64)> = Vec::new();
    const N: usize = 60;
    for i in 0..N {
        let prev = *pts_seq.last().unwrap();
        player.step_frames(1);
        let new = wait_for_new_pts(&mut player, prev, Duration::from_millis(1500));
        if new == prev {
            stuck.push((i, prev));
        }
        pts_seq.push(new);
    }

    let monotonic = pts_seq.windows(2).all(|w| w[1] >= w[0]);
    let unique: std::collections::BTreeSet<i64> = pts_seq.iter().copied().collect();
    let details = format!(
        "unique={}/{} monotonic={} stuck_at={:?} head={:?} tail={:?}",
        unique.len(),
        pts_seq.len(),
        monotonic,
        stuck.iter().take(5).collect::<Vec<_>>(),
        &pts_seq[..pts_seq.len().min(6)],
        &pts_seq[pts_seq.len().saturating_sub(6)..],
    );
    if stuck.is_empty() && monotonic && unique.len() == pts_seq.len() {
        ScenarioResult::ok("frame_step_forward_long_run", details)
    } else {
        ScenarioResult::fail("frame_step_forward_long_run", details)
    }
}

/// Long run of frame-backward steps. Exercises the
/// `step_frames(-1)` path, which always falls back to a full seek
/// because the decoded queue only grows forward.
fn scenario_frame_step_backward_long_run(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "frame_step_backward_long_run",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    player.seek(0.70);
    let _ = spin_and_collect(&mut player, Duration::from_millis(1500));
    let start_pts = player.uploaded_pts_us_for_test();

    let mut pts_seq: Vec<i64> = vec![start_pts];
    let mut stuck: Vec<(usize, i64)> = Vec::new();
    const N: usize = 40;
    for i in 0..N {
        let prev = *pts_seq.last().unwrap();
        player.step_frames(-1);
        let new = wait_for_new_pts(&mut player, prev, Duration::from_millis(1500));
        if new == prev {
            stuck.push((i, prev));
        }
        pts_seq.push(new);
    }

    let monotonic_back = pts_seq.windows(2).all(|w| w[1] <= w[0]);
    let unique: std::collections::BTreeSet<i64> = pts_seq.iter().copied().collect();
    let details = format!(
        "unique={}/{} monotonic_back={} stuck_at={:?} head={:?} tail={:?}",
        unique.len(),
        pts_seq.len(),
        monotonic_back,
        stuck.iter().take(5).collect::<Vec<_>>(),
        &pts_seq[..pts_seq.len().min(6)],
        &pts_seq[pts_seq.len().saturating_sub(6)..],
    );
    if stuck.is_empty() && monotonic_back && unique.len() == pts_seq.len() {
        ScenarioResult::ok("frame_step_backward_long_run", details)
    } else {
        ScenarioResult::fail("frame_step_backward_long_run", details)
    }
}

/// Interleaved random seeks + frame steps. Simulates a user scrubbing
/// around the file and then fine-tuning with the comma/period keys.
fn scenario_mixed_random_seek_and_step(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "mixed_random_seek_and_step",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);
    player.play();
    let _ = spin_and_collect(&mut player, Duration::from_millis(300));
    player.pause();
    let _ = spin_and_collect(&mut player, Duration::from_millis(100));

    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("mixed_random_seek_and_step", "zero duration");
    }

    let mut rng = Lcg::new(0xF00D_FACE_CAFE_0001);
    #[derive(Debug)]
    enum Op {
        Seek(f32),
        StepF(i32),
        StepB(i32),
    }
    let mut ops: Vec<Op> = Vec::new();
    for _ in 0..25 {
        match rng.next_u32() % 3 {
            0 => ops.push(Op::Seek(rng.next_frac())),
            1 => ops.push(Op::StepF(1 + (rng.next_u32() % 3) as i32)),
            _ => ops.push(Op::StepB(1 + (rng.next_u32() % 3) as i32)),
        }
    }

    let mut last_pts = player.uploaded_pts_us_for_test();
    let mut failures: Vec<(usize, String, i64, i64)> = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        let prev = last_pts;
        match op {
            Op::Seek(f) => {
                let target = (duration_us as f32 * f) as i64;
                player.seek(*f);
                let (got, _) = wait_for_seek_target(
                    &mut player,
                    target,
                    Duration::from_millis(1500),
                );
                if (got - target).abs() > PTS_TOLERANCE_US {
                    failures.push((i, format!("{:?}", op), target, got));
                }
                last_pts = got;
            }
            Op::StepF(n) => {
                for _ in 0..*n {
                    player.step_frames(1);
                }
                let got = wait_for_new_pts(
                    &mut player,
                    prev,
                    Duration::from_millis(1500),
                );
                if got <= prev {
                    failures.push((i, format!("{:?}", op), prev, got));
                }
                last_pts = got;
            }
            Op::StepB(n) => {
                for _ in 0..*n {
                    player.step_frames(-1);
                }
                let got = wait_for_new_pts(
                    &mut player,
                    prev,
                    Duration::from_millis(1500),
                );
                if got >= prev {
                    failures.push((i, format!("{:?}", op), prev, got));
                }
                last_pts = got;
            }
        }
    }

    let details = format!(
        "failures={}/{} first_failures={:?}",
        failures.len(),
        ops.len(),
        failures.iter().take(5).collect::<Vec<_>>()
    );
    if failures.is_empty() {
        ScenarioResult::ok("mixed_random_seek_and_step", details)
    } else {
        ScenarioResult::fail("mixed_random_seek_and_step", details)
    }
}

/// Eframe-faithful simulation: drive the player through an explicit
/// `ctx.begin_pass()` + `ctx.end_pass()` cycle on every "frame", just
/// like real eframe does. Between passes, sleep exactly the amount
/// eframe would sleep based on egui's repaint-delay request. If the
/// player forgets to request a repaint at any point during a seek
/// storm, this scenario hangs waiting for the next pass — and we
/// fail the test.
fn scenario_eframe_gui_seek_storm(
    ctx: &egui::Context,
    backend: &WgpuBackend,
    path: &Path,
) -> ScenarioResult {
    let mut player =
        match Player::open_with_backend(ctx, path, Some(backend.clone())) {
            Ok(p) => p,
            Err(e) => {
                return ScenarioResult::fail(
                    "eframe_gui_seek_storm",
                    format!("open: {}", e),
                )
            }
        };
    player.set_volume(0.0);

    // Helper: runs one "frame" of eframe: begin_pass → tick (GUI
    // would also process input and draw) → end_pass. Returns whether
    // the player bumped last_uploaded_pts_us during this pass.
    fn run_pass(ctx: &egui::Context, player: &mut Player) -> i64 {
        let raw_input = egui::RawInput::default();
        ctx.begin_pass(raw_input);
        tick_and_flush(player);
        let pts = player.uploaded_pts_us_for_test();
        let _ = ctx.end_pass();
        pts
    }

    // Warm up: run a few "playing" passes so the decoder is
    // producing frames.
    player.play();
    for _ in 0..30 {
        let _ = run_pass(ctx, &mut player);
        std::thread::sleep(Duration::from_millis(16));
    }
    player.pause();
    for _ in 0..6 {
        let _ = run_pass(ctx, &mut player);
        std::thread::sleep(Duration::from_millis(16));
    }

    let duration_us = player.duration_ms() * 1000;
    if duration_us <= 0 {
        return ScenarioResult::fail("eframe_gui_seek_storm", "zero duration");
    }

    // Now simulate a user mashing seek keys: each "keypress" is its
    // own pass, where input handling (the seek) runs INSIDE the pass
    // (after tick, same as real GUI). After each seek pass we run up
    // to 40 more passes waiting for the target frame to land,
    // sleeping a fraction of a frame between each.
    let mut rng = Lcg::new(0xABCD_1234_5678_9ABC);
    let mut failures: Vec<(usize, f32, i64, i64)> = Vec::new();
    const N: usize = 30;
    for i in 0..N {
        let frac = rng.next_frac();
        let target = (duration_us as f32 * frac) as i64;

        // Pass 0: the "keypress" pass — tick first (simulating the
        // `player.tick()` call at the top of the GUI update()), then
        // seek (simulating the input handler), then end_pass.
        let raw_input = egui::RawInput::default();
        ctx.begin_pass(raw_input);
        tick_and_flush(&mut player);
        player.seek(frac);
        let _ = ctx.end_pass();

        // Passes 1..=120: wait for the seek to land. We only run a
        // pass when egui says it wants one (simulating eframe's own
        // schedule). Long-GOP codecs on HW decode can take ~1 s to
        // walk from a keyframe to a random target on some GPUs,
        // so the budget is ~2 s of wall time (120 × ~16 ms).
        let mut landed = player.uploaded_pts_us_for_test();
        for _ in 0..120 {
            std::thread::sleep(Duration::from_millis(16));
            if !ctx.has_requested_repaint() {
                continue;
            }
            let pts = run_pass(ctx, &mut player);
            if (pts - target).abs() <= PTS_TOLERANCE_US {
                landed = pts;
                break;
            }
            landed = pts;
        }
        if (landed - target).abs() > PTS_TOLERANCE_US {
            failures.push((i, frac, target, landed));
        }
    }

    let details = format!(
        "failures={}/{} first_failures={:?}",
        failures.len(),
        N,
        failures.iter().take(5).collect::<Vec<_>>()
    );
    if failures.is_empty() {
        ScenarioResult::ok("eframe_gui_seek_storm", details)
    } else {
        ScenarioResult::fail("eframe_gui_seek_storm", details)
    }
}

fn main() {
    let path = match find_test_video() {
        Some(p) => p,
        None => {
            eprintln!("video_test: no test video found. Pass a path as the first argument.");
            std::process::exit(2);
        }
    };
    eprintln!("video_test: using {}", path.display());

    {
        use cpal::traits::{DeviceTrait, HostTrait};
        let host = cpal::default_host();
        match host.default_output_device() {
            Some(dev) => eprintln!(
                "video_test: audio output = {:?} ({:?})",
                dev.name().unwrap_or_default(),
                dev.default_output_config().map(|c| (c.sample_rate().0, c.channels())).ok()
            ),
            None => eprintln!("video_test: no audio output device"),
        }
    }

    let (ctx, backend) = make_headless_backend();
    eprintln!(
        "video_test: wgpu ready, supports_16bit_norm={}",
        backend.supports_16bit_norm
    );

    let scenarios: &[(
        &'static str,
        fn(&egui::Context, &WgpuBackend, &Path) -> ScenarioResult,
    )] = &[
        ("basic_playback", scenario_basic_playback),
        ("looping_short_file", scenario_looping_short_file),
        ("no_first_frame_stuck", scenario_no_first_frame_stuck),
        ("seek_while_playing", scenario_seek_while_playing),
        ("seek_while_paused", scenario_seek_while_paused),
        ("play_pause_seek_resume", scenario_play_pause_seek_resume),
        ("paused_frame_step", scenario_paused_frame_step),
        ("paused_frame_step_gated", scenario_paused_frame_step_gated),
        ("rapid_seek", scenario_rapid_seek),
        // VLC-parity aggressive scenarios — these exercise the
        // user-reported "display doesn't update after rapid seeks"
        // bug pattern.
        ("paused_random_spam_seek", scenario_paused_random_spam_seek),
        ("playing_random_spam_seek", scenario_playing_random_spam_seek),
        ("burst_coalesce_seek", scenario_burst_coalesce_seek),
        ("frame_step_forward_long_run", scenario_frame_step_forward_long_run),
        ("frame_step_backward_long_run", scenario_frame_step_backward_long_run),
        ("mixed_random_seek_and_step", scenario_mixed_random_seek_and_step),
        ("eframe_gui_seek_storm", scenario_eframe_gui_seek_storm),
        // End-of-file and clip-loop regressions.
        ("tail_is_not_cut_short", scenario_tail_is_not_cut_short),
        ("loop_restarts_at_real_speed", scenario_loop_restarts_at_real_speed),
        ("clip_loop_stays_in_range", scenario_clip_loop_stays_in_range),
        ("clip_loop_clears", scenario_clip_loop_clears),
        ("av_sync_while_moving_around", scenario_av_sync_while_moving_around),
        ("av_sync_across_loop_wrap", scenario_av_sync_across_loop_wrap),
        ("survives_audio_output_loss", scenario_survives_audio_output_loss),
    ];

    // `--only <text>` runs just the scenarios whose name contains <text>.
    let only: Option<String> = {
        let args: Vec<String> = env::args().collect();
        args.iter()
            .position(|a| a == "--only")
            .and_then(|i| args.get(i + 1).cloned())
    };

    let mut results: Vec<ScenarioResult> = Vec::new();
    for (name, f) in scenarios {
        if let Some(filter) = &only {
            if !name.contains(filter.as_str()) {
                continue;
            }
        }
        eprintln!("\n---- {} ----", name);
        let r = f(&ctx, &backend, &path);
        eprintln!("  [{}] {}", if r.pass { "PASS" } else { "FAIL" }, r.details);
        results.push(r);
    }

    eprintln!("\n===== SUMMARY =====");
    let mut pass_count = 0;
    let mut fail_count = 0;
    for r in &results {
        let tag = if r.pass { "PASS" } else { "FAIL" };
        eprintln!("  [{}] {}: {}", tag, r.name, r.details);
        if r.pass {
            pass_count += 1;
        } else {
            fail_count += 1;
        }
    }
    eprintln!(
        "\n{} scenario(s) passed, {} failed",
        pass_count, fail_count
    );
    if fail_count > 0 {
        std::process::exit(1);
    }
}

// Keep this alias around so `cargo` doesn't warn about unused imports
// from the shared `video_player` module that the scenarios don't all
// touch directly.
#[allow(dead_code)]
fn _type_touch(_s: PlayerState) {}
