//! Native WASAPI audio output with device-clock timing.
//!
//! cpal can say when it asked for a buffer, but not when that buffer will
//! actually be heard: on Windows its estimate is just the free space in
//! the buffer, which leaves out the audio engine's processing and the
//! device's own latency — tens of milliseconds on USB speakers, and
//! hundreds on Bluetooth. A player that schedules video against that
//! estimate shows each picture before its sound.
//!
//! WASAPI's `IAudioClock` reports the position of the sample currently
//! playing at the device, stamped with the performance-counter time it
//! was measured at. Knowing how many frames have been written, that
//! gives the exact moment every newly written frame will be heard. This
//! is the same clock VLC and mpv use on Windows.
//!
//! The output also discards the device's buffered audio on seek (the way
//! VLC flushes its output), so audio from the old position never plays
//! over the new one, and it reopens on the new default device if the
//! current one is unplugged or disabled.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use windows::core::GUID;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Media::Audio::{
    eConsole, eRender, IAudioClient, IAudioClock, IAudioRenderClient, IMMDeviceEnumerator,
    MMDeviceEnumerator, AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
    WAVEFORMATEX, WAVEFORMATEXTENSIBLE,
};
use windows::Win32::Media::KernelStreaming::WAVE_FORMAT_EXTENSIBLE;
use windows::Win32::Media::Multimedia::{KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_ALL,
    COINIT_MULTITHREADED,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

/// What the output thread renders from.
pub trait RenderSource: Send + 'static {
    /// Fill `data` (interleaved f32, `channels` per frame). `heard_at` is
    /// when `data`'s first frame will reach the listener.
    fn fill(&mut self, data: &mut [f32], channels: usize, heard_at: Instant);
    /// True, once, when everything written so far has been superseded by
    /// a seek and should be dropped from the device.
    fn take_flush(&mut self) -> bool;
    /// The output has stopped for good; timing must come from elsewhere.
    fn on_failed(&mut self, reason: &str);
}

/// Buffer requested from the audio engine. Timing is compensated from the
/// device clock, so a roomier buffer costs nothing in sync; it only makes
/// the output harder to starve when the machine is busy.
const BUFFER_DURATION: Duration = Duration::from_millis(40);

/// How long to keep trying to reopen after the device disappears, which
/// covers Windows switching to the next default device.
const REOPEN_WINDOW: Duration = Duration::from_secs(5);

pub struct WasapiOutput {
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

/// Why the output didn't start.
pub enum StartError<S> {
    /// The device couldn't be used; the source comes back so another
    /// backend can take it.
    Unavailable(String, S),
    /// Setup failed in a way that kept the source (the thread couldn't
    /// be created, or device setup hung). Timing has to come from
    /// somewhere other than audio.
    Lost(String),
}

impl WasapiOutput {
    /// Start rendering `source` on the default output device. The device
    /// must run at `rate` (the rate the audio was resampled to).
    pub fn start<S: RenderSource>(source: S, rate: u32) -> Result<Self, StartError<S>> {
        let stop = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), (String, S)>>();
        let stop_thread = stop.clone();
        let thread = thread::Builder::new()
            .name("wasapi-output".into())
            .spawn(move || run_output(source, rate, stop_thread, ready_tx))
            .map_err(|e| StartError::Lost(format!("spawn output thread: {}", e)))?;
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => Ok(Self {
                stop,
                thread: Some(thread),
            }),
            Ok(Err((reason, source))) => {
                let _ = thread.join();
                Err(StartError::Unavailable(reason, source))
            }
            Err(_) => {
                // Device setup is wedged. Tell the thread to give up as
                // soon as it gets anywhere, and don't wait on it.
                stop.store(true, Ordering::Relaxed);
                Err(StartError::Lost("device setup did not finish within 5 s".into()))
            }
        }
    }
}

impl Drop for WasapiOutput {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            // The loop wakes at least every 200 ms.
            let deadline = Instant::now() + Duration::from_millis(500);
            while !t.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            if t.is_finished() {
                let _ = t.join();
            }
        }
    }
}

fn run_output<S: RenderSource>(
    mut source: S,
    rate: u32,
    stop: Arc<AtomicBool>,
    ready: mpsc::Sender<Result<(), (String, S)>>,
) {
    // SAFETY: paired with CoUninitialize below on this same thread.
    let com = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    if com.is_err() {
        let _ = ready.send(Err((format!("COM init failed: {:?}", com), source)));
        return;
    }

    let mut session = match Session::open(rate) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready.send(Err((e, source)));
            unsafe { CoUninitialize() };
            return;
        }
    };
    let _ = ready.send(Ok(()));

    loop {
        let end = session.run(&mut source, &stop);
        drop(session);
        match end {
            SessionEnd::Stopped => break,
            SessionEnd::Lost(reason) => {
                // The device went away (unplugged, disabled, or its format
                // changed). Windows promotes another device to default;
                // follow it, as long as it runs at the rate the audio is
                // already resampled to.
                let deadline = Instant::now() + REOPEN_WINDOW;
                let reopened = loop {
                    if stop.load(Ordering::Relaxed) {
                        break None;
                    }
                    if let Ok(s) = Session::open(rate) {
                        break Some(s);
                    }
                    if Instant::now() > deadline {
                        break None;
                    }
                    thread::sleep(Duration::from_millis(200));
                };
                match reopened {
                    Some(s) => session = s,
                    None => {
                        if !stop.load(Ordering::Relaxed) {
                            source.on_failed(&reason);
                        }
                        break;
                    }
                }
            }
        }
    }

    // SAFETY: balances the successful CoInitializeEx above.
    unsafe { CoUninitialize() };
}

enum SessionEnd {
    Stopped,
    Lost(String),
}

/// One opened, initialized audio client on the default render device.
struct Session {
    client: IAudioClient,
    render: IAudioRenderClient,
    clock: IAudioClock,
    event: HANDLE,
    buffer_frames: u32,
    channels: usize,
    rate: u32,
    clock_freq: u64,
    /// Frames handed to the device since the last start or reset. The
    /// device position counts from the same origin.
    written: u64,
}

/// Owns the format block `GetMixFormat` allocates.
struct MixFormat(*mut WAVEFORMATEX);

impl Drop for MixFormat {
    fn drop(&mut self) {
        // SAFETY: allocated by GetMixFormat with CoTaskMemAlloc.
        unsafe { CoTaskMemFree(Some(self.0 as *const std::ffi::c_void)) };
    }
}

fn guid_eq(a: &GUID, b: &GUID) -> bool {
    (a.data1, a.data2, a.data3, a.data4) == (b.data1, b.data2, b.data3, b.data4)
}

impl Session {
    fn open(rate: u32) -> Result<Self, String> {
        // SAFETY: plain COM calls on an initialized MTA thread; every
        // returned interface is owned and released on drop.
        unsafe {
            let enumerator =
                CoCreateInstance::<_, IMMDeviceEnumerator>(&MMDeviceEnumerator, None, CLSCTX_ALL)
                    .map_err(|e| format!("device enumerator: {}", e))?;
            let device = enumerator
                .GetDefaultAudioEndpoint(eRender, eConsole)
                .map_err(|e| format!("default output device: {}", e))?;
            let client: IAudioClient = device
                .Activate(CLSCTX_ALL, None)
                .map_err(|e| format!("activate audio client: {}", e))?;

            let format = MixFormat(
                client
                    .GetMixFormat()
                    .map_err(|e| format!("mix format: {}", e))?,
            );
            let fmt = &*format.0;
            let is_float32 = fmt.wBitsPerSample == 32
                && match fmt.wFormatTag as u32 {
                    WAVE_FORMAT_IEEE_FLOAT => true,
                    WAVE_FORMAT_EXTENSIBLE => {
                        let ext = &*(format.0 as *const WAVEFORMATEXTENSIBLE);
                        let sub = ext.SubFormat;
                        guid_eq(&sub, &KSDATAFORMAT_SUBTYPE_IEEE_FLOAT)
                    }
                    _ => false,
                };
            if !is_float32 {
                return Err("device mix format is not 32-bit float".into());
            }
            let device_rate = fmt.nSamplesPerSec;
            if device_rate != rate {
                return Err(format!(
                    "device runs at {} Hz, audio was prepared for {} Hz",
                    device_rate, rate
                ));
            }
            let channels = fmt.nChannels as usize;
            if channels == 0 {
                return Err("device reports zero channels".into());
            }

            // 100 ns units.
            let buffer_hns = (BUFFER_DURATION.as_nanos() / 100) as i64;
            client
                .Initialize(
                    AUDCLNT_SHAREMODE_SHARED,
                    AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                    buffer_hns,
                    0,
                    format.0,
                    None,
                )
                .map_err(|e| format!("initialize audio client: {}", e))?;

            let event = CreateEventW(None, false, false, windows::core::PCWSTR::null())
                .map_err(|e| format!("create event: {}", e))?;
            if let Err(e) = client.SetEventHandle(event) {
                let _ = CloseHandle(event);
                return Err(format!("set event handle: {}", e));
            }
            let buffer_frames = client
                .GetBufferSize()
                .map_err(|e| format!("buffer size: {}", e))?;
            let render: IAudioRenderClient = client
                .GetService()
                .map_err(|e| format!("render client: {}", e))?;
            let clock: IAudioClock = client
                .GetService()
                .map_err(|e| format!("audio clock: {}", e))?;
            let clock_freq = clock
                .GetFrequency()
                .map_err(|e| format!("clock frequency: {}", e))?;

            Ok(Self {
                client,
                render,
                clock,
                event,
                buffer_frames,
                channels,
                rate: device_rate,
                clock_freq,
                written: 0,
            })
        }
    }

    fn run<S: RenderSource>(&mut self, source: &mut S, stop: &AtomicBool) -> SessionEnd {
        // Prime the buffer before starting so playback doesn't open on
        // an underrun.
        if let Err(e) = self.fill_available(source) {
            return SessionEnd::Lost(e);
        }
        // SAFETY: initialized client.
        if let Err(e) = unsafe { self.client.Start() } {
            return SessionEnd::Lost(format!("start: {}", e));
        }
        loop {
            if stop.load(Ordering::Relaxed) {
                return SessionEnd::Stopped;
            }
            // SAFETY: `event` stays valid for the session's lifetime. The
            // timeout keeps stop requests and flushes responsive even if
            // the device stops signalling.
            unsafe { WaitForSingleObject(self.event, 200) };

            if source.take_flush() {
                if let Err(e) = self.flush(source) {
                    return SessionEnd::Lost(e);
                }
                continue;
            }
            if let Err(e) = self.fill_available(source) {
                return SessionEnd::Lost(e);
            }
        }
    }

    /// Drop everything queued in the device and start again from what
    /// the source provides now. This is what keeps audio from the old
    /// position from playing after a seek.
    fn flush<S: RenderSource>(&mut self, source: &mut S) -> Result<(), String> {
        // SAFETY: Stop/Reset/Start on an initialized client; Reset
        // requires the stream to be stopped, which Stop guarantees.
        unsafe {
            self.client.Stop().map_err(|e| format!("stop: {}", e))?;
            self.client.Reset().map_err(|e| format!("reset: {}", e))?;
        }
        self.written = 0;
        self.fill_available(source)?;
        unsafe { self.client.Start() }.map_err(|e| format!("restart: {}", e))
    }

    fn fill_available<S: RenderSource>(&mut self, source: &mut S) -> Result<(), String> {
        // SAFETY: initialized client.
        let padding = unsafe { self.client.GetCurrentPadding() }
            .map_err(|e| format!("padding: {}", e))?;
        let avail = self.buffer_frames.saturating_sub(padding);
        if avail == 0 {
            return Ok(());
        }
        let heard_at = self.first_new_frame_heard_at(padding);
        // SAFETY: the buffer from GetBuffer holds exactly `avail` frames
        // of `channels` f32s (the mix format was verified to be 32-bit
        // float) and is released before the next GetBuffer.
        unsafe {
            let ptr = self
                .render
                .GetBuffer(avail)
                .map_err(|e| format!("get buffer: {}", e))?;
            let data =
                std::slice::from_raw_parts_mut(ptr as *mut f32, avail as usize * self.channels);
            source.fill(data, self.channels, heard_at);
            self.render
                .ReleaseBuffer(avail, 0)
                .map_err(|e| format!("release buffer: {}", e))?;
        }
        self.written += avail as u64;
        Ok(())
    }

    /// When the next frame written will reach the listener, from the
    /// device clock: the frames still queued ahead of it, counted from
    /// the moment the device reported its position.
    fn first_new_frame_heard_at(&self, padding: u32) -> Instant {
        let now = Instant::now();
        let mut position: u64 = 0;
        let mut qpc_position: u64 = 0;
        // SAFETY: valid clock interface and out-pointers.
        let ok = unsafe {
            self.clock
                .GetPosition(&mut position, Some(&mut qpc_position))
                .is_ok()
        };
        let measured = if ok && self.clock_freq > 0 {
            qpc_now_100ns().map(|qpc_now| {
                seconds_until_heard(
                    self.written,
                    position,
                    self.clock_freq,
                    self.rate,
                    qpc_now.saturating_sub(qpc_position),
                )
            })
        } else {
            None
        };
        // Without a usable clock reading, the queued padding is the best
        // estimate available.
        let secs = measured.unwrap_or(padding as f64 / self.rate as f64);
        offset_instant(now, secs)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: stopping an already-stopped client is harmless; the
        // event handle is owned by this session.
        unsafe {
            let _ = self.client.Stop();
            let _ = CloseHandle(self.event);
        }
    }
}

/// How long from now until frame number `written` (counting from the
/// last start or reset) is heard.
///
/// `position` is the device position in `clock_freq` units per second,
/// sampled `sampled_ago_100ns` ago. Frames between the device position
/// and `written` are queued ahead of the new frame.
pub(crate) fn seconds_until_heard(
    written: u64,
    position: u64,
    clock_freq: u64,
    rate: u32,
    sampled_ago_100ns: u64,
) -> f64 {
    let played_frames = position as f64 * rate as f64 / clock_freq as f64;
    let queued_secs = (written as f64 - played_frames).max(0.0) / rate as f64;
    let sampled_ago_secs = sampled_ago_100ns as f64 / 10_000_000.0;
    queued_secs - sampled_ago_secs
}

fn offset_instant(base: Instant, secs: f64) -> Instant {
    if secs >= 0.0 {
        base + Duration::from_secs_f64(secs)
    } else {
        base.checked_sub(Duration::from_secs_f64(-secs)).unwrap_or(base)
    }
}

/// The performance counter now, in the 100 ns units `IAudioClock`
/// stamps its positions with.
fn qpc_now_100ns() -> Option<u64> {
    #[link(name = "kernel32")]
    extern "system" {
        fn QueryPerformanceCounter(count: *mut i64) -> i32;
        fn QueryPerformanceFrequency(freq: *mut i64) -> i32;
    }
    let mut count = 0i64;
    let mut freq = 0i64;
    // SAFETY: both write a single i64 through a valid pointer.
    let ok = unsafe { QueryPerformanceCounter(&mut count) != 0 && QueryPerformanceFrequency(&mut freq) != 0 };
    if !ok || freq <= 0 || count < 0 {
        return None;
    }
    Some((count as i128 * 10_000_000 / freq as i128) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_frames_set_the_delay() {
        // Device clock in frames (freq == rate). 480 frames written, the
        // device is at frame 0, measured just now: 10 ms until heard.
        let s = seconds_until_heard(480, 0, 48_000, 48_000, 0);
        assert!((s - 0.010).abs() < 1e-9, "{}", s);
    }

    #[test]
    fn a_stale_position_reading_is_aged() {
        // Same queue, but the position was read 4 ms ago: the device has
        // played 4 ms more since, so the new frame is 6 ms away.
        let s = seconds_until_heard(480, 0, 48_000, 48_000, 40_000);
        assert!((s - 0.006).abs() < 1e-9, "{}", s);
    }

    #[test]
    fn position_units_are_converted_by_the_clock_frequency() {
        // Shared-mode clocks often count bytes: 8 bytes per stereo f32
        // frame at 48 kHz is 384000 units per second. Position 3840
        // bytes = 480 frames played; 960 written leaves 480 queued.
        let s = seconds_until_heard(960, 3_840, 384_000, 48_000, 0);
        assert!((s - 0.010).abs() < 1e-9, "{}", s);
    }

    #[test]
    fn a_device_ahead_of_the_writer_means_now() {
        // Position past what was written (a glitch or a reset race) must
        // not produce a delay in the past beyond the reading's age.
        let s = seconds_until_heard(100, 48_000, 48_000, 48_000, 0);
        assert_eq!(s, 0.0);
    }
}
