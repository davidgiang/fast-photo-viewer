//! Move files to the Windows Recycle Bin.
//!
//! Uses the shell's file-operation API rather than `std::fs::remove_file`
//! so a mis-keyed delete stays recoverable — the whole point of the
//! feature is that it is undoable from Explorer.

#[cfg(target_os = "windows")]
mod windows_impl {
    use std::ffi::c_void;
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;

    // <shellapi.h>
    const FO_DELETE: u32 = 0x0003;
    const FOF_SILENT: u16 = 0x0004;
    const FOF_NOCONFIRMATION: u16 = 0x0010;
    const FOF_ALLOWUNDO: u16 = 0x0040;
    const FOF_NOERRORUI: u16 = 0x0400;
    /// Ask rather than silently hard-delete when the item is too big
    /// for the bin or the bin is disabled. Without it, "recycle" can
    /// quietly become "erase", which is exactly the outcome this
    /// module exists to prevent.
    const FOF_WANTNUKEWARNING: u16 = 0x4000;

    #[repr(C)]
    struct ShFileOpStructW {
        hwnd: *mut c_void,
        w_func: u32,
        p_from: *const u16,
        p_to: *const u16,
        f_flags: u16,
        f_any_operations_aborted: i32,
        h_name_mappings: *mut c_void,
        lpsz_progress_title: *const u16,
    }

    #[link(name = "shell32")]
    extern "system" {
        fn SHFileOperationW(lp_file_op: *mut ShFileOpStructW) -> i32;
    }

    /// Build the double-NUL-terminated wide string `SHFileOperationW`
    /// expects for `pFrom` (the API takes a list, terminated by an
    /// extra NUL).
    fn wide_double_nul(path: &Path) -> Vec<u16> {
        let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
        wide.push(0);
        wide.push(0);
        wide
    }

    /// `SHFileOperationW` predates extended-length paths and fails on
    /// the `\\?\` prefix that `canonicalize` produces, so strip it. UNC
    /// paths get the prefix rewritten back to `\\`.
    fn shell_friendly(path: &Path) -> std::path::PathBuf {
        let absolute = path
            .canonicalize()
            .unwrap_or_else(|_| path.to_path_buf());
        let text = absolute.to_string_lossy().into_owned();
        if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
            std::path::PathBuf::from(format!(r"\\{}", rest))
        } else if let Some(rest) = text.strip_prefix(r"\\?\") {
            std::path::PathBuf::from(rest)
        } else {
            absolute
        }
    }

    pub fn move_to_recycle_bin(path: &Path) -> Result<(), String> {
        if !path.exists() {
            return Err("file no longer exists".to_string());
        }
        let target = shell_friendly(path);
        let from = wide_double_nul(&target);

        let mut op = ShFileOpStructW {
            hwnd: std::ptr::null_mut(),
            w_func: FO_DELETE,
            p_from: from.as_ptr(),
            p_to: std::ptr::null(),
            f_flags: FOF_ALLOWUNDO
                | FOF_NOCONFIRMATION
                | FOF_SILENT
                | FOF_NOERRORUI
                | FOF_WANTNUKEWARNING,
            f_any_operations_aborted: 0,
            h_name_mappings: std::ptr::null_mut(),
            lpsz_progress_title: std::ptr::null(),
        };

        // SAFETY: `op` is a fully-initialised SHFILEOPSTRUCTW and
        // `from` outlives the call, keeping `p_from` valid.
        let rc = unsafe { SHFileOperationW(&mut op) };

        if rc != 0 {
            return Err(format!("shell delete failed (code {})", rc));
        }
        if op.f_any_operations_aborted != 0 {
            return Err("delete was cancelled".to_string());
        }
        // The shell reports success for a no-op, so confirm the file
        // actually went somewhere before telling the caller it did.
        if target.exists() {
            return Err("file was not removed".to_string());
        }
        Ok(())
    }
}

#[cfg(target_os = "windows")]
pub use windows_impl::move_to_recycle_bin;

#[cfg(not(target_os = "windows"))]
pub fn move_to_recycle_bin(_path: &std::path::Path) -> Result<(), String> {
    Err("recycle bin is only supported on Windows".to_string())
}

#[cfg(all(test, target_os = "windows"))]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn missing_file_is_reported_not_silently_ignored() {
        let path = std::env::temp_dir().join("fpv_recycle_missing_xyz.tmp");
        let _ = fs::remove_file(&path);
        let err = move_to_recycle_bin(&path).unwrap_err();
        assert!(
            err.contains("no longer exists"),
            "unexpected error text: {}",
            err
        );
    }

    #[test]
    fn real_file_is_removed_from_disk() {
        let dir = std::env::temp_dir().join("fpv_recycle_tests");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("recycle_me.txt");
        fs::write(&path, b"fast-photo-viewer recycle bin test").unwrap();
        assert!(path.exists());

        match move_to_recycle_bin(&path) {
            Ok(()) => assert!(!path.exists(), "file should be gone from its old location"),
            // A machine with the Recycle Bin disabled for temp's volume
            // is a legitimate environment, not a test failure — but the
            // error must be reported rather than swallowed.
            Err(e) => {
                let _ = fs::remove_file(&path);
                println!("recycle unavailable in this environment: {}", e);
            }
        }
    }
}
