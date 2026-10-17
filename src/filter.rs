//! Media filtering: which files the viewer steps through.
//!
//! The filter is plain data plus a pure `matches` predicate, so it can be
//! tested without a window. Most of what it can express (type, length,
//! resolution, audio track) is not visible from the file name, so the
//! app indexes files in the background into `MediaMeta` and passes what
//! it has to `matches`. A file that hasn't been indexed yet is judged by
//! its extension alone, which keeps the quick filters instant and lets
//! them refine themselves as indexing finishes.

use std::path::Path;

use crate::media::{is_audio_file, is_image_file, is_video_file};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MediaKind {
    Image,
    Video,
    Audio,
}

/// What indexing learned about one file.
#[derive(Clone, Debug, PartialEq)]
pub struct MediaMeta {
    pub kind: MediaKind,
    /// Playback length; `None` for images and for files that don't say.
    pub duration_us: Option<i64>,
    /// Picture size; 0x0 for audio and for images whose size is unknown.
    pub width: u32,
    pub height: u32,
    pub has_audio: bool,
}

/// The kind a file claims to be from its extension alone. A video
/// container (`.mp4`, `.mkv`) may really hold only audio; indexing
/// corrects that.
pub fn kind_by_extension(path: &Path) -> Option<MediaKind> {
    if is_image_file(path) {
        Some(MediaKind::Image)
    } else if is_video_file(path) {
        Some(MediaKind::Video)
    } else if is_audio_file(path) {
        Some(MediaKind::Audio)
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AudioTrack {
    Any,
    /// Has sound: videos with an audio track, and audio files.
    With,
    /// Videos with no audio track.
    Without,
}

/// Picture shape, judged by the dimensions the viewer actually displays
/// (stored pixel size; EXIF rotation is not applied by the viewer).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Orientation {
    Any,
    /// Wider than tall.
    Landscape,
    /// Taller than wide.
    Portrait,
    /// Exactly as wide as tall.
    Square,
}

/// Resolution thresholds offered in the UI, as the shorter side in
/// pixels. The shorter side so that a portrait phone clip and a
/// landscape one of the same quality land in the same bucket.
pub const RESOLUTION_STEPS: &[(u32, &str)] = &[
    (0, "Any"),
    (480, "480p"),
    (720, "720p (HD)"),
    (1080, "1080p (Full HD)"),
    (1440, "1440p (QHD)"),
    (2160, "2160p (4K)"),
];

#[derive(Clone, PartialEq, Debug)]
pub struct MediaFilter {
    pub images: bool,
    pub videos: bool,
    pub audio: bool,
    /// Playback length bounds in seconds; 0 means "no limit". Images
    /// have no length, so an active bound excludes them.
    pub min_duration_s: u32,
    pub max_duration_s: u32,
    /// Bounds on the shorter side in pixels; 0 means "no limit". Audio
    /// has no picture, so an active bound excludes it.
    pub min_short_side: u32,
    pub max_short_side: u32,
    pub orientation: Orientation,
    pub audio_track: AudioTrack,
}

impl Default for MediaFilter {
    fn default() -> Self {
        Self::all()
    }
}

impl MediaFilter {
    pub fn all() -> Self {
        Self {
            images: true,
            videos: true,
            audio: true,
            min_duration_s: 0,
            max_duration_s: 0,
            min_short_side: 0,
            max_short_side: 0,
            orientation: Orientation::Any,
            audio_track: AudioTrack::Any,
        }
    }

    fn only(images: bool, videos: bool, audio: bool) -> Self {
        Self { images, videos, audio, ..Self::all() }
    }

    pub fn images_only() -> Self {
        Self::only(true, false, false)
    }

    pub fn videos_only() -> Self {
        Self::only(false, true, false)
    }

    pub fn audio_only() -> Self {
        Self::only(false, false, true)
    }

    /// The quick-cycle order behind the `M` key.
    pub fn cycle(&self) -> Self {
        let presets = [
            Self::all(),
            Self::images_only(),
            Self::videos_only(),
            Self::audio_only(),
        ];
        match presets.iter().position(|p| p == self) {
            Some(i) => presets[(i + 1) % presets.len()].clone(),
            // From a custom filter, the next press goes back to the start.
            None => Self::all(),
        }
    }

    pub fn is_default(&self) -> bool {
        *self == Self::all()
    }

    fn has_range_constraints(&self) -> bool {
        self.min_duration_s > 0
            || self.max_duration_s > 0
            || self.min_short_side > 0
            || self.max_short_side > 0
            || self.orientation != Orientation::Any
            || self.audio_track != AudioTrack::Any
    }

    /// Whether answering correctly requires looking inside files, as
    /// opposed to reading their extensions. Telling a video from an
    /// audio-only `.mp4` needs it too, but only when the filter treats
    /// the two differently.
    pub fn needs_index(&self) -> bool {
        self.has_range_constraints() || (self.videos != self.audio)
    }

    pub fn label(&self) -> String {
        let types = match (self.images, self.videos, self.audio) {
            (true, true, true) => "All Media".to_string(),
            (true, false, false) => "Images".to_string(),
            (false, true, false) => "Video".to_string(),
            (false, false, true) => "Audio".to_string(),
            (true, true, false) => "Images & Video".to_string(),
            (true, false, true) => "Images & Audio".to_string(),
            (false, true, true) => "Video & Audio".to_string(),
            (false, false, false) => "Nothing".to_string(),
        };
        if self.has_range_constraints() {
            format!("{} (custom)", types)
        } else {
            types
        }
    }

    /// Does `path` pass the filter? `meta` is what indexing found, if it
    /// has got to this file yet.
    pub fn matches(&self, path: &Path, meta: Option<&MediaMeta>) -> bool {
        let kind = match meta.map(|m| m.kind).or_else(|| kind_by_extension(path)) {
            Some(k) => k,
            None => return false,
        };
        let type_ok = match kind {
            MediaKind::Image => self.images,
            MediaKind::Video => self.videos,
            MediaKind::Audio => self.audio,
        };
        if !type_ok {
            return false;
        }
        if !self.has_range_constraints() {
            return true;
        }
        // Everything below needs facts about the file; until indexing
        // has them the file is held back rather than guessed at.
        let meta = match meta {
            Some(m) => m,
            None => return false,
        };

        if self.min_duration_s > 0 || self.max_duration_s > 0 {
            let secs = match (kind, meta.duration_us) {
                (MediaKind::Video | MediaKind::Audio, Some(us)) if us > 0 => us as f64 / 1e6,
                _ => return false,
            };
            if self.min_duration_s > 0 && secs < self.min_duration_s as f64 {
                return false;
            }
            if self.max_duration_s > 0 && secs > self.max_duration_s as f64 {
                return false;
            }
        }

        if self.min_short_side > 0 || self.max_short_side > 0 {
            let short = meta.width.min(meta.height);
            if kind == MediaKind::Audio || short == 0 {
                return false;
            }
            if self.min_short_side > 0 && short < self.min_short_side {
                return false;
            }
            if self.max_short_side > 0 && short > self.max_short_side {
                return false;
            }
        }

        if self.orientation != Orientation::Any {
            let (w, h) = (meta.width, meta.height);
            if kind == MediaKind::Audio || w == 0 || h == 0 {
                return false;
            }
            let shape_ok = match self.orientation {
                Orientation::Any => true,
                Orientation::Landscape => w > h,
                Orientation::Portrait => h > w,
                Orientation::Square => w == h,
            };
            if !shape_ok {
                return false;
            }
        }

        match self.audio_track {
            AudioTrack::Any => {}
            AudioTrack::With => {
                let has_sound = match kind {
                    MediaKind::Audio => true,
                    MediaKind::Video => meta.has_audio,
                    MediaKind::Image => false,
                };
                if !has_sound {
                    return false;
                }
            }
            AudioTrack::Without => {
                if !(kind == MediaKind::Video && !meta.has_audio) {
                    return false;
                }
            }
        }
        true
    }
}

/// Learn what the filter needs to know about `path`. Opens the file
/// header only; nothing is decoded.
pub fn probe_meta(path: &Path) -> Option<MediaMeta> {
    if is_image_file(path) {
        let (width, height) = image::image_dimensions(path).unwrap_or((0, 0));
        return Some(MediaMeta {
            kind: MediaKind::Image,
            duration_us: None,
            width,
            height,
            has_audio: false,
        });
    }
    let probe = crate::video_player::probe_media(path).ok()?;
    Some(MediaMeta {
        kind: if probe.has_video { MediaKind::Video } else { MediaKind::Audio },
        duration_us: (probe.duration_us > 0).then_some(probe.duration_us),
        width: probe.width,
        height: probe.height,
        has_audio: probe.has_audio,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn video(secs: f64, w: u32, h: u32, audio: bool) -> MediaMeta {
        MediaMeta {
            kind: MediaKind::Video,
            duration_us: Some((secs * 1e6) as i64),
            width: w,
            height: h,
            has_audio: audio,
        }
    }

    fn sound(secs: f64) -> MediaMeta {
        MediaMeta {
            kind: MediaKind::Audio,
            duration_us: Some((secs * 1e6) as i64),
            width: 0,
            height: 0,
            has_audio: true,
        }
    }

    fn photo(w: u32, h: u32) -> MediaMeta {
        MediaMeta { kind: MediaKind::Image, duration_us: None, width: w, height: h, has_audio: false }
    }

    fn p(name: &str) -> PathBuf {
        PathBuf::from(name)
    }

    #[test]
    fn default_filter_accepts_every_supported_type_and_nothing_else() {
        let f = MediaFilter::all();
        for name in ["a.jpg", "a.mp4", "a.mp3", "a.NEF", "a.m4a"] {
            assert!(f.matches(&p(name), None), "{}", name);
        }
        assert!(!f.matches(&p("a.txt"), None));
        assert!(!f.matches(&p("noext"), None));
        assert!(f.is_default());
        assert!(!f.needs_index());
    }

    #[test]
    fn type_presets_select_exactly_one_kind() {
        let cases = [
            (MediaFilter::images_only(), [true, false, false]),
            (MediaFilter::videos_only(), [false, true, false]),
            (MediaFilter::audio_only(), [false, false, true]),
        ];
        for (f, [img, vid, aud]) in cases {
            assert_eq!(f.matches(&p("a.jpg"), None), img);
            assert_eq!(f.matches(&p("a.mkv"), None), vid);
            assert_eq!(f.matches(&p("a.flac"), None), aud);
        }
    }

    #[test]
    fn types_can_be_combined() {
        let f = MediaFilter { images: true, videos: false, audio: true, ..MediaFilter::all() };
        assert!(f.matches(&p("a.png"), None));
        assert!(f.matches(&p("a.mp3"), None));
        assert!(!f.matches(&p("a.mp4"), None));
        assert_eq!(f.label(), "Images & Audio");
    }

    #[test]
    fn an_audio_only_mp4_counts_as_audio_once_indexed() {
        let memo = p("memo.mp4");
        // Before indexing the extension says video...
        assert!(MediaFilter::videos_only().matches(&memo, None));
        assert!(!MediaFilter::audio_only().matches(&memo, None));
        // ...and the index corrects it.
        let meta = sound(30.0);
        assert!(!MediaFilter::videos_only().matches(&memo, Some(&meta)));
        assert!(MediaFilter::audio_only().matches(&memo, Some(&meta)));
    }

    #[test]
    fn an_indexed_video_stays_video() {
        let clip = video(10.0, 1920, 1080, true);
        assert!(MediaFilter::videos_only().matches(&p("clip.mp4"), Some(&clip)));
        assert!(!MediaFilter::audio_only().matches(&p("clip.mp4"), Some(&clip)));
    }

    #[test]
    fn duration_bounds_apply_to_timed_media_and_hold_back_unindexed_files() {
        let f = MediaFilter { min_duration_s: 60, max_duration_s: 300, ..MediaFilter::all() };
        let m = |s| video(s, 1280, 720, true);
        assert!(!f.matches(&p("a.mp4"), Some(&m(59.0))));
        assert!(f.matches(&p("a.mp4"), Some(&m(60.0))));
        assert!(f.matches(&p("a.mp4"), Some(&m(300.0))));
        assert!(!f.matches(&p("a.mp4"), Some(&m(301.0))));
        // Songs have a length too.
        assert!(f.matches(&p("a.mp3"), Some(&sound(180.0))));
        // Not indexed yet: unknown, so not shown.
        assert!(!f.matches(&p("a.mp4"), None));
        // Photos have no length at all.
        assert!(!f.matches(&p("a.jpg"), Some(&photo(4000, 3000))));
        assert!(f.needs_index());
    }

    #[test]
    fn open_ended_duration_bounds() {
        let at_least = MediaFilter { min_duration_s: 120, ..MediaFilter::all() };
        assert!(at_least.matches(&p("a.mp4"), Some(&video(3600.0, 1280, 720, true))));
        assert!(!at_least.matches(&p("a.mp4"), Some(&video(5.0, 1280, 720, true))));
        let at_most = MediaFilter { max_duration_s: 10, ..MediaFilter::all() };
        assert!(at_most.matches(&p("a.mp4"), Some(&video(5.0, 1280, 720, true))));
        assert!(!at_most.matches(&p("a.mp4"), Some(&video(11.0, 1280, 720, true))));
    }

    #[test]
    fn a_file_with_unknown_length_fails_a_duration_bound() {
        let f = MediaFilter { max_duration_s: 600, ..MediaFilter::all() };
        let mut m = video(1.0, 640, 480, true);
        m.duration_us = None;
        assert!(!f.matches(&p("a.mp4"), Some(&m)));
    }

    #[test]
    fn resolution_uses_the_shorter_side_so_portrait_and_landscape_agree() {
        let f = MediaFilter { min_short_side: 1080, ..MediaFilter::all() };
        assert!(f.matches(&p("a.mp4"), Some(&video(5.0, 1920, 1080, true))));
        assert!(f.matches(&p("a.mp4"), Some(&video(5.0, 1080, 1920, true)))); // portrait
        assert!(f.matches(&p("a.mp4"), Some(&video(5.0, 3840, 2160, true))));
        assert!(!f.matches(&p("a.mp4"), Some(&video(5.0, 1280, 720, true))));
        assert!(!f.matches(&p("a.mp4"), Some(&video(5.0, 1920, 800, true)))); // scope crop
    }

    #[test]
    fn resolution_applies_to_photos_and_excludes_audio() {
        let f = MediaFilter { min_short_side: 2000, ..MediaFilter::all() };
        assert!(f.matches(&p("a.jpg"), Some(&photo(6000, 4000))));
        assert!(!f.matches(&p("a.jpg"), Some(&photo(1600, 1200))));
        assert!(!f.matches(&p("a.mp3"), Some(&sound(10.0))), "no picture, so no resolution");
        // A photo whose size couldn't be read can't prove it is big enough.
        assert!(!f.matches(&p("a.nef"), Some(&photo(0, 0))));
    }

    #[test]
    fn maximum_resolution() {
        let f = MediaFilter { max_short_side: 720, ..MediaFilter::all() };
        assert!(f.matches(&p("a.mp4"), Some(&video(5.0, 1280, 720, true))));
        assert!(!f.matches(&p("a.mp4"), Some(&video(5.0, 1920, 1080, true))));
    }

    #[test]
    fn orientation_filter_for_photos_and_video() {
        let land = MediaFilter { orientation: Orientation::Landscape, ..MediaFilter::all() };
        let port = MediaFilter { orientation: Orientation::Portrait, ..MediaFilter::all() };
        let sq = MediaFilter { orientation: Orientation::Square, ..MediaFilter::all() };
        let wide = photo(6000, 4000);
        let tall = photo(4000, 6000);
        let even = photo(2048, 2048);
        assert!(land.matches(&p("a.jpg"), Some(&wide)));
        assert!(!land.matches(&p("a.jpg"), Some(&tall)));
        assert!(!land.matches(&p("a.jpg"), Some(&even)));
        assert!(port.matches(&p("a.jpg"), Some(&tall)));
        assert!(!port.matches(&p("a.jpg"), Some(&wide)));
        assert!(sq.matches(&p("a.jpg"), Some(&even)));
        assert!(!sq.matches(&p("a.jpg"), Some(&wide)));
        // Videos have a shape too (a phone clip held upright).
        assert!(port.matches(&p("a.mp4"), Some(&video(5.0, 1080, 1920, true))));
        assert!(land.matches(&p("a.mp4"), Some(&video(5.0, 1920, 1080, true))));
    }

    #[test]
    fn orientation_cannot_be_judged_without_a_known_picture_size() {
        let land = MediaFilter { orientation: Orientation::Landscape, ..MediaFilter::all() };
        assert!(!land.matches(&p("a.mp3"), Some(&sound(5.0))), "audio has no shape");
        assert!(!land.matches(&p("a.nef"), Some(&photo(0, 0))), "unreadable size");
        assert!(!land.matches(&p("a.jpg"), None), "not indexed yet");
        assert!(land.needs_index());
        assert_eq!(land.label(), "All Media (custom)");
    }

    #[test]
    fn orientation_combines_with_type_and_resolution() {
        let f = MediaFilter {
            images: true,
            videos: false,
            audio: false,
            orientation: Orientation::Portrait,
            min_short_side: 2000,
            ..MediaFilter::all()
        };
        assert!(f.matches(&p("a.jpg"), Some(&photo(3000, 4000))));
        assert!(!f.matches(&p("a.jpg"), Some(&photo(1000, 1500))), "too small");
        assert!(!f.matches(&p("a.jpg"), Some(&photo(4000, 3000))), "wrong shape");
        assert!(!f.matches(&p("a.mp4"), Some(&video(5.0, 2160, 3840, true))), "wrong type");
    }

    #[test]
    fn audio_track_filter() {
        let with = MediaFilter { audio_track: AudioTrack::With, ..MediaFilter::all() };
        let without = MediaFilter { audio_track: AudioTrack::Without, ..MediaFilter::all() };
        let talky = video(5.0, 640, 480, true);
        let silent = video(5.0, 640, 480, false);
        assert!(with.matches(&p("a.mp4"), Some(&talky)));
        assert!(!with.matches(&p("a.mp4"), Some(&silent)));
        assert!(with.matches(&p("a.mp3"), Some(&sound(5.0))));
        assert!(!with.matches(&p("a.jpg"), Some(&photo(10, 10))));
        assert!(without.matches(&p("a.mp4"), Some(&silent)));
        assert!(!without.matches(&p("a.mp4"), Some(&talky)));
        assert!(!without.matches(&p("a.mp3"), Some(&sound(5.0))));
    }

    #[test]
    fn constraints_combine_with_and() {
        let f = MediaFilter {
            videos: true,
            images: false,
            audio: false,
            min_duration_s: 30,
            min_short_side: 1080,
            audio_track: AudioTrack::With,
            ..MediaFilter::all()
        };
        assert!(f.matches(&p("a.mp4"), Some(&video(45.0, 1920, 1080, true))));
        assert!(!f.matches(&p("a.mp4"), Some(&video(10.0, 1920, 1080, true))), "too short");
        assert!(!f.matches(&p("a.mp4"), Some(&video(45.0, 1280, 720, true))), "too small");
        assert!(!f.matches(&p("a.mp4"), Some(&video(45.0, 1920, 1080, false))), "silent");
        assert!(!f.matches(&p("a.jpg"), Some(&photo(4000, 3000))), "wrong type");
    }

    #[test]
    fn quick_cycle_walks_the_presets_and_recovers_from_custom() {
        let mut f = MediaFilter::all();
        let mut seen = vec![f.label()];
        for _ in 0..4 {
            f = f.cycle();
            seen.push(f.label());
        }
        assert_eq!(seen, ["All Media", "Images", "Video", "Audio", "All Media"]);

        let custom = MediaFilter { min_duration_s: 5, ..MediaFilter::videos_only() };
        assert_eq!(custom.cycle(), MediaFilter::all());
        assert_eq!(custom.label(), "Video (custom)");
    }

    #[test]
    fn indexing_is_only_requested_when_it_would_change_the_answer() {
        assert!(!MediaFilter::all().needs_index());
        assert!(!MediaFilter::images_only().needs_index(), "images are exact by extension");
        assert!(MediaFilter::videos_only().needs_index(), "an .mp4 might be audio-only");
        assert!(MediaFilter::audio_only().needs_index());
        let both = MediaFilter { images: false, ..MediaFilter::all() };
        assert!(!both.needs_index(), "video and audio are treated alike");
        assert!(MediaFilter { min_duration_s: 1, ..MediaFilter::all() }.needs_index());
    }

    #[test]
    fn probe_meta_reads_the_real_fixtures() {
        let _ = ffmpeg_the_third::init();
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures");

        let clip = probe_meta(&dir.join("video_with_audio.mp4")).unwrap();
        assert_eq!(clip.kind, MediaKind::Video);
        assert_eq!((clip.width, clip.height), (64, 48));
        assert!(clip.has_audio);
        assert!(clip.duration_us.unwrap() > 900_000);

        let memo = probe_meta(&dir.join("audio_only.mp4")).unwrap();
        assert_eq!(memo.kind, MediaKind::Audio, "an audio-only .mp4 is audio");
        assert_eq!((memo.width, memo.height), (0, 0));

        let cover = probe_meta(&dir.join("tone_cover.mp3")).unwrap();
        assert_eq!(cover.kind, MediaKind::Audio, "album art is not video");

        let silent = probe_meta(&dir.join("video_only.mp4")).unwrap();
        assert!(!silent.has_audio);

        assert!(probe_meta(&dir.join("nope.mp4")).is_none());
    }

    #[test]
    fn probed_fixtures_work_through_the_filter_end_to_end() {
        let _ = ffmpeg_the_third::init();
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures");
        let all = [
            "audio_only.mp4",
            "tone.m4a",
            "tone.mp3",
            "tone_cover.mp3",
            "video_with_audio.mp4",
            "video_only.mp4",
        ];
        let pick = |f: &MediaFilter| -> Vec<&str> {
            all.iter()
                .copied()
                .filter(|n| {
                    let path = dir.join(n);
                    f.matches(&path, probe_meta(&path).as_ref())
                })
                .collect()
        };
        assert_eq!(pick(&MediaFilter::videos_only()), ["video_with_audio.mp4", "video_only.mp4"]);
        assert_eq!(
            pick(&MediaFilter::audio_only()),
            ["audio_only.mp4", "tone.m4a", "tone.mp3", "tone_cover.mp3"]
        );
        let silent = MediaFilter { audio_track: AudioTrack::Without, ..MediaFilter::all() };
        assert_eq!(pick(&silent), ["video_only.mp4"]);
        let hd = MediaFilter { min_short_side: 720, ..MediaFilter::all() };
        assert!(pick(&hd).is_empty(), "the fixture clips are tiny");
    }
}
