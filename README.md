# Fast Photo Viewer

An extremely fast, lightweight photo and video viewer built in **Rust**. Designed to handle directories with tens of thousands of media files efficiently using GPU acceleration.

## Download

Download the latest installer from [GitHub Releases](https://github.com/davidgiang1/fast-photo-viewer/releases).

Run `FastPhotoViewer-x.x.x-setup.exe` and follow the prompts. The installer will:
- Install Fast Photo Viewer to Program Files
- Optionally associate image and video file types
- Create Start Menu shortcuts

To set as your default viewer, go to **Windows Settings > Default Apps** and search for **Fast Photo Viewer**.

## Features

- **Instant Startup:** Compiled native binary.
- **Parallel Scanning:** Uses background threads to scan directories.
- **Image Support:** PNG, JPEG, WEBP, GIF, BMP, TIFF, ICO, SVG. Handles wrong extensions and truncated files.
- **Video Support:** MP4, MKV, AVI, MOV, WMV, FLV, WEBM, M4V, MPG, MPEG, 3GP, OGV, VOB with full playback controls.
- **Audio Support:** MP3, M4A, AAC, WAV, FLAC, OGG, OPUS, WMA, AIFF — and MP4/MKV/etc. files that contain only audio — play with the same controls and a simple placeholder in place of the picture.
- **GPU Rendering:** Hardware-accelerated display.
- **File Associations:** Can be set as the default Windows viewer for images and videos.
- **Random Slideshow:** Press `Space` or `Right Arrow`.
- **Navigation History:** Use `Left Arrow`.
- **Zoom & Pan:** Scroll to zoom, drag to pan. Works on both images and videos.
- **Media Filter:** Cycle between All / Images / Video / Audio with `M`. Press `Shift+M` (or click **Filters**) for advanced filters: any mix of types, video/audio length range, resolution range (by shorter side), horizontal / vertical / square shape (photos and video), and with/without sound. File details are read in the background as soon as a folder opens (gently, until a filter needs them) and remembered in `%LOCALAPPDATA%\fast-photo-viewer\index-v1.tsv` so an unchanged file is never read twice.
- **Fullscreen:** Toggle with `F11` or the fullscreen button.
- **Audio Detection:** Automatically detects whether a video has audio; shows mute indicator if not.
- **Click to Pause:** Click anywhere on a video to toggle play/pause.
- **File Explorer:** Open current file location.

## Usage

Open a file directly from the command line or by double-clicking an associated file:
```bash
fast-photo-viewer.exe "C:\Photos\image.jpg"
```

Or launch the application and use the keyboard shortcuts:

### Image Mode
| Key | Action |
|-----|--------|
| `O` | Open folder |
| `F` | Open file |
| `Space` / `Right Arrow` | Next image |
| `Left Arrow` | Previous image |
| `+` / `-` | Zoom in / out |
| Scroll wheel | Zoom |
| Click + drag | Pan |
| `M` | Cycle media filter (All / Images / Video / Audio); `Shift+M` opens advanced filters |
| `F11` | Toggle fullscreen |
| `Esc` × 2 | Close |

### Video Mode
| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `Left` / `Right Arrow` | Seek ±3 seconds |
| `Ctrl+Left` / `Ctrl+Right` | Previous / Next video |
| `Up` / `Down Arrow` | Volume +/- 2% |
| `+` / `-` | Zoom in / out |
| Scroll wheel | Zoom |
| Click | Play / Pause |
| Click + drag | Pan |
| `M` | Cycle media filter (All / Images / Video / Audio); `Shift+M` opens advanced filters |
| `F11` | Toggle fullscreen |
| `Esc` × 2 | Close |

## Building from Source

Requires **FFmpeg 7.x** specifically (matches `ffmpeg-the-third 2.0`). FFmpeg 8.x will fail to build.

1. Install [Rust](https://rustup.rs/) and [LLVM](https://github.com/llvm/llvm-project/releases) (LLVM provides `libclang`, needed by `bindgen`).
2. Download FFmpeg 7.x **shared** libraries (e.g. an `n7.1` release from [BtbN](https://github.com/BtbN/FFmpeg-Builds/releases) or [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)) and extract into the project root, e.g. `./ffmpeg7/ffmpeg-n7.1-latest-win64-gpl-shared-7.1/`.
3. Create `.cargo/config.toml` so Cargo can find it:
    ```toml
    [env]
    FFMPEG_DIR = { value = "ffmpeg7/ffmpeg-n7.1-latest-win64-gpl-shared-7.1", relative = true }
    ```
   `relative = true` resolves the path against the project root. Adjust the path to match where you extracted FFmpeg, or use an absolute path with `relative = false`.
4. Build:
    ```bash
    cargo build --release
    ```
5. The binary will be at `target/release/fast-photo-viewer.exe`. Copy the FFmpeg DLLs from `<FFMPEG_DIR>/bin/` alongside it.
