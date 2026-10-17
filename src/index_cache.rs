//! A small on-disk memory of what indexing has learned about each file.
//!
//! Reading a file's details means opening it, which is cheap once and
//! painful across a library of tens of thousands of files. Remembering
//! the answer between runs means each file is only ever opened again if
//! it has changed. An entry is trusted only while the file's size and
//! modification time are exactly what they were when it was recorded.
//!
//! The format is one tab-separated line per file with the path last, so
//! it needs no extra dependencies and survives hand-editing or damage:
//! a line that doesn't parse is skipped, and a file with an unknown
//! header is ignored wholesale (the index is rebuilt, never migrated).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::filter::{MediaKind, MediaMeta};

const HEADER: &str = "fast-photo-viewer index v1";

/// What a file looked like when it was indexed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FileStamp {
    pub size: u64,
    pub mtime_ns: u64,
}

impl FileStamp {
    pub fn of(path: &Path) -> Option<Self> {
        let md = fs::metadata(path).ok()?;
        let mtime_ns = md
            .modified()
            .ok()?
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_nanos() as u64;
        Some(Self { size: md.len(), mtime_ns })
    }
}

/// (files answered from the index, files that still need reading)
pub type KnownSplit = (Vec<(PathBuf, Option<MediaMeta>)>, Vec<(PathBuf, Option<FileStamp>)>);

#[derive(Default)]
pub struct DiskIndex {
    entries: HashMap<PathBuf, (FileStamp, Option<MediaMeta>)>,
    unsaved: usize,
}

fn kind_code(kind: MediaKind) -> char {
    match kind {
        MediaKind::Image => 'i',
        MediaKind::Video => 'v',
        MediaKind::Audio => 'a',
    }
}

fn kind_from_code(c: &str) -> Option<MediaKind> {
    match c {
        "i" => Some(MediaKind::Image),
        "v" => Some(MediaKind::Video),
        "a" => Some(MediaKind::Audio),
        _ => None,
    }
}

impl DiskIndex {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Entries recorded since the last save.
    pub fn unsaved(&self) -> usize {
        self.unsaved
    }

    /// The remembered answer for `path`, if the file is unchanged.
    /// The outer `Option` is "do we know"; the inner one is the answer
    /// (`None` there means the file was unreadable when last tried).
    pub fn get(&self, path: &Path, stamp: FileStamp) -> Option<Option<MediaMeta>> {
        match self.entries.get(path) {
            Some((saved, meta)) if *saved == stamp => Some(meta.clone()),
            _ => None,
        }
    }

    /// Sort `files` into those the index can answer without opening them
    /// and those that still need reading. A file whose stamp couldn't be
    /// taken (it vanished, or can't be statted) always needs reading.
    pub fn split_known(&self, files: Vec<(PathBuf, Option<FileStamp>)>) -> KnownSplit {
        let mut known = Vec::with_capacity(files.len());
        let mut unknown = Vec::new();
        for (path, stamp) in files {
            match stamp.and_then(|st| self.get(&path, st)) {
                Some(meta) => known.push((path, meta)),
                None => unknown.push((path, stamp)),
            }
        }
        (known, unknown)
    }

    pub fn insert(&mut self, path: PathBuf, stamp: FileStamp, meta: Option<MediaMeta>) {
        self.entries.insert(path, (stamp, meta));
        self.unsaved += 1;
    }

    pub fn serialize(&self) -> String {
        let mut out = String::with_capacity(64 + self.entries.len() * 96);
        out.push_str(HEADER);
        out.push('\n');
        // Sorted so the file is stable between saves and diff-able.
        let mut paths: Vec<&PathBuf> = self.entries.keys().collect();
        paths.sort();
        for path in paths {
            let (stamp, meta) = &self.entries[path];
            let path = path.to_string_lossy();
            // Tabs and newlines can't occur in a Windows path, but a
            // line break in one would corrupt the file, so refuse them.
            if path.contains(['\t', '\n', '\r']) {
                continue;
            }
            match meta {
                Some(m) => out.push_str(&format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
                    stamp.size,
                    stamp.mtime_ns,
                    kind_code(m.kind),
                    m.duration_us.unwrap_or(-1),
                    m.width,
                    m.height,
                    m.has_audio as u8,
                    path,
                )),
                None => out.push_str(&format!(
                    "{}\t{}\tx\t-1\t0\t0\t0\t{}\n",
                    stamp.size, stamp.mtime_ns, path
                )),
            }
        }
        out
    }

    pub fn parse(text: &str) -> Self {
        let mut lines = text.lines();
        if lines.next() != Some(HEADER) {
            return Self::empty();
        }
        let mut entries = HashMap::new();
        for line in lines {
            if let Some((path, entry)) = Self::parse_line(line) {
                entries.insert(path, entry);
            }
        }
        Self { entries, unsaved: 0 }
    }

    fn parse_line(line: &str) -> Option<(PathBuf, (FileStamp, Option<MediaMeta>))> {
        // splitn so the path, last and unrestricted, keeps any odd characters.
        let mut f = line.splitn(8, '\t');
        let size = f.next()?.parse().ok()?;
        let mtime_ns = f.next()?.parse().ok()?;
        let kind = f.next()?;
        let duration: i64 = f.next()?.parse().ok()?;
        let width = f.next()?.parse().ok()?;
        let height = f.next()?.parse().ok()?;
        let has_audio = f.next()? == "1";
        let path = f.next()?;
        if path.is_empty() {
            return None;
        }
        let stamp = FileStamp { size, mtime_ns };
        let meta = if kind == "x" {
            None
        } else {
            Some(MediaMeta {
                kind: kind_from_code(kind)?,
                duration_us: (duration >= 0).then_some(duration),
                width,
                height,
                has_audio,
            })
        };
        Some((PathBuf::from(path), (stamp, meta)))
    }

    pub fn load_from(file: &Path) -> Self {
        fs::read_to_string(file)
            .map(|t| Self::parse(&t))
            .unwrap_or_default()
    }

    /// Write via a temporary file and rename, so a crash mid-save can't
    /// leave a half-written index behind.
    pub fn save_to(&mut self, file: &Path) -> std::io::Result<()> {
        if let Some(dir) = file.parent() {
            fs::create_dir_all(dir)?;
        }
        let tmp = file.with_extension("tmp");
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(self.serialize().as_bytes())?;
            f.sync_all()?;
        }
        fs::rename(&tmp, file)?;
        self.unsaved = 0;
        Ok(())
    }
}

/// How many files to read at once. Reading is mostly waiting on the disk,
/// so a few threads help a lot; but while nobody is waiting for the
/// answers (no filter needs them yet) it stays small so browsing stays
/// smooth.
pub fn worker_count(cores: usize, urgent: bool) -> usize {
    if urgent {
        cores.clamp(2, 8)
    } else {
        (cores / 2).clamp(1, 4)
    }
}

/// Where the index lives: `%LOCALAPPDATA%\fast-photo-viewer\index-v1.tsv`.
/// `None` when there's no such place, in which case nothing is remembered.
pub fn default_path() -> Option<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")?;
    Some(PathBuf::from(base).join("fast-photo-viewer").join("index-v1.tsv"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(size: u64, mtime_ns: u64) -> FileStamp {
        FileStamp { size, mtime_ns }
    }

    fn clip() -> MediaMeta {
        MediaMeta {
            kind: MediaKind::Video,
            duration_us: Some(12_500_000),
            width: 1920,
            height: 1080,
            has_audio: true,
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fpv_index_{}_{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_unchanged_file_is_remembered() {
        let mut idx = DiskIndex::empty();
        idx.insert(PathBuf::from("C:/a/clip.mp4"), stamp(100, 5), Some(clip()));
        assert_eq!(idx.get(Path::new("C:/a/clip.mp4"), stamp(100, 5)), Some(Some(clip())));
    }

    #[test]
    fn a_changed_file_is_not_trusted() {
        let mut idx = DiskIndex::empty();
        idx.insert(PathBuf::from("a.mp4"), stamp(100, 5), Some(clip()));
        assert_eq!(idx.get(Path::new("a.mp4"), stamp(101, 5)), None, "size changed");
        assert_eq!(idx.get(Path::new("a.mp4"), stamp(100, 6)), None, "modified since");
        assert_eq!(idx.get(Path::new("other.mp4"), stamp(100, 5)), None, "never seen");
    }

    #[test]
    fn round_trips_every_kind_including_unreadable_files() {
        let mut idx = DiskIndex::empty();
        idx.insert(PathBuf::from("v.mp4"), stamp(1, 2), Some(clip()));
        idx.insert(
            PathBuf::from("song.mp3"),
            stamp(3, 4),
            Some(MediaMeta {
                kind: MediaKind::Audio,
                duration_us: Some(180_000_000),
                width: 0,
                height: 0,
                has_audio: true,
            }),
        );
        idx.insert(
            PathBuf::from("p.jpg"),
            stamp(5, 6),
            Some(MediaMeta {
                kind: MediaKind::Image,
                duration_us: None,
                width: 6000,
                height: 4000,
                has_audio: false,
            }),
        );
        idx.insert(PathBuf::from("broken.mp4"), stamp(7, 8), None);

        let back = DiskIndex::parse(&idx.serialize());
        assert_eq!(back.len(), 4);
        for (path, st) in [("v.mp4", stamp(1, 2)), ("song.mp3", stamp(3, 4)), ("p.jpg", stamp(5, 6)), ("broken.mp4", stamp(7, 8))] {
            assert_eq!(
                back.get(Path::new(path), st),
                idx.get(Path::new(path), st),
                "{} must survive a save and load",
                path
            );
        }
        assert_eq!(back.get(Path::new("broken.mp4"), stamp(7, 8)), Some(None), "failure is remembered too");
        assert_eq!(back.unsaved(), 0, "a freshly loaded index has nothing to save");
    }

    #[test]
    fn awkward_paths_survive() {
        let mut idx = DiskIndex::empty();
        for name in ["C:/My Photos/summer trip/IMG 001.jpg", "D:/музыка/песня.mp3", "E:/a=b&c [1] (2).mp4"] {
            idx.insert(PathBuf::from(name), stamp(10, 20), Some(clip()));
        }
        let back = DiskIndex::parse(&idx.serialize());
        assert_eq!(back.len(), 3);
        assert!(back.get(Path::new("D:/музыка/песня.mp3"), stamp(10, 20)).is_some());
    }

    #[test]
    fn damaged_lines_are_skipped_and_good_ones_kept() {
        let text = format!(
            "{}\n10\t20\tv\t5000000\t640\t480\t1\tgood.mp4\nnot a line at all\n10\tNaN\tv\t1\t1\t1\t1\tbad.mp4\n10\t20\tq\t1\t1\t1\t1\tunknown_kind.mp4\n11\t21\ti\t-1\t100\t100\t0\talso_good.jpg\n",
            HEADER
        );
        let idx = DiskIndex::parse(&text);
        assert_eq!(idx.len(), 2);
        assert!(idx.get(Path::new("good.mp4"), stamp(10, 20)).is_some());
        assert!(idx.get(Path::new("also_good.jpg"), stamp(11, 21)).is_some());
    }

    #[test]
    fn an_unknown_or_missing_header_means_start_fresh() {
        assert_eq!(DiskIndex::parse("").len(), 0);
        assert_eq!(DiskIndex::parse("fast-photo-viewer index v999\n1\t2\tv\t1\t1\t1\t1\ta.mp4\n").len(), 0);
        assert_eq!(DiskIndex::parse("1\t2\tv\t1\t1\t1\t1\ta.mp4\n").len(), 0);
    }

    #[test]
    fn saving_and_loading_through_a_real_file() {
        let dir = temp_dir("save");
        let file = dir.join("nested").join("index.tsv");
        let mut idx = DiskIndex::empty();
        idx.insert(PathBuf::from("a.mp4"), stamp(1, 1), Some(clip()));
        assert_eq!(idx.unsaved(), 1);
        idx.save_to(&file).unwrap();
        assert_eq!(idx.unsaved(), 0);
        assert!(!file.with_extension("tmp").exists(), "no temp file left behind");

        let back = DiskIndex::load_from(&file);
        assert_eq!(back.get(Path::new("a.mp4"), stamp(1, 1)), Some(Some(clip())));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn split_known_separates_cached_files_from_ones_that_need_reading() {
        let mut idx = DiskIndex::empty();
        idx.insert(PathBuf::from("seen.mp4"), stamp(1, 1), Some(clip()));
        idx.insert(PathBuf::from("edited.mp4"), stamp(2, 2), Some(clip()));
        let (known, unknown) = idx.split_known(vec![
            (PathBuf::from("seen.mp4"), Some(stamp(1, 1))),
            (PathBuf::from("edited.mp4"), Some(stamp(2, 99))),
            (PathBuf::from("new.mp4"), Some(stamp(3, 3))),
            (PathBuf::from("vanished.mp4"), None),
        ]);
        let known_names: Vec<_> = known.iter().map(|(p, _)| p.to_str().unwrap()).collect();
        let unknown_names: Vec<_> = unknown.iter().map(|(p, _)| p.to_str().unwrap()).collect();
        assert_eq!(known_names, ["seen.mp4"]);
        assert_eq!(unknown_names, ["edited.mp4", "new.mp4", "vanished.mp4"]);
    }

    /// The point of the whole module: index real files once, restart,
    /// and the second pass has nothing left to open.
    #[test]
    fn a_second_run_over_unchanged_files_reads_nothing() {
        let _ = ffmpeg_the_third::init();
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures");
        let names = ["audio_only.mp4", "tone.m4a", "tone.mp3", "tone_cover.mp3", "video_with_audio.mp4", "video_only.mp4"];
        let stamped = || -> Vec<(PathBuf, Option<FileStamp>)> {
            names.iter().map(|n| {
                let p = fixtures.join(n);
                let st = FileStamp::of(&p);
                (p, st)
            }).collect()
        };

        // First run: nothing known, everything is read and remembered.
        let dir = temp_dir("twice");
        let file = dir.join("index.tsv");
        let mut idx = DiskIndex::load_from(&file);
        let (known, unknown) = idx.split_known(stamped());
        assert!(known.is_empty());
        assert_eq!(unknown.len(), names.len());
        let mut first_pass = Vec::new();
        for (path, stamp) in unknown {
            let meta = crate::filter::probe_meta(&path);
            idx.insert(path.clone(), stamp.unwrap(), meta.clone());
            first_pass.push((path, meta));
        }
        idx.save_to(&file).unwrap();

        // Second run, as if the app had restarted.
        let idx = DiskIndex::load_from(&file);
        let (mut known, unknown) = idx.split_known(stamped());
        assert!(unknown.is_empty(), "unchanged files must not be reopened");
        known.sort_by(|a, b| a.0.cmp(&b.0));
        first_pass.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(known, first_pass, "and the remembered answers match what was read");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn background_indexing_is_gentle_until_a_filter_needs_it() {
        for cores in [1, 2, 4, 8, 16, 64] {
            let idle = worker_count(cores, false);
            let rushed = worker_count(cores, true);
            assert!((1..=4).contains(&idle), "{} cores: idle {}", cores, idle);
            assert!((2..=8).contains(&rushed), "{} cores: urgent {}", cores, rushed);
            assert!(idle <= rushed, "{} cores", cores);
        }
        assert_eq!(worker_count(1, false), 1, "a single core still makes progress");
        assert_eq!(worker_count(16, true), 8);
    }

    #[test]
    fn a_missing_index_file_is_just_an_empty_index() {
        let idx = DiskIndex::load_from(Path::new("definitely/not/here/index.tsv"));
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn stamps_follow_real_files() {
        let dir = temp_dir("stamp");
        let f = dir.join("x.bin");
        fs::write(&f, b"hello").unwrap();
        let first = FileStamp::of(&f).unwrap();
        assert_eq!(first.size, 5);
        assert_eq!(FileStamp::of(&f), Some(first), "stable while untouched");

        fs::write(&f, b"hello world").unwrap();
        let second = FileStamp::of(&f).unwrap();
        assert_ne!(first, second, "a rewrite changes the stamp");
        assert!(FileStamp::of(&dir.join("gone.bin")).is_none());
        let _ = fs::remove_dir_all(&dir);
    }
}
