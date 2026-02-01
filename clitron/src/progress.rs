//! Progress reporting and terminal output utilities.
//!
//! This module provides human-friendly progress indicators, styled terminal output,
//! and support for terminal progress protocols like Ghostty's OSC 9;4.

use std::io::{self, Write};
use std::time::{Duration, Instant};

/// Icons for terminal output.
pub mod icons {
    /// Downloading icon.
    pub const DOWNLOAD: &str = "⬇";
    /// Success/checkmark icon.
    pub const SUCCESS: &str = "✓";
    /// Error/cross icon.
    pub const ERROR: &str = "✗";
    /// Info icon.
    pub const INFO: &str = "ℹ";
    /// Warning icon.
    pub const WARNING: &str = "⚠";
    /// Spinner frames for animation.
    pub const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    /// Model/brain icon.
    pub const MODEL: &str = "🧠";
    /// Package icon.
    pub const PACKAGE: &str = "📦";
}

/// ANSI color codes for terminal output.
pub mod colors {
    /// Reset all formatting.
    pub const RESET: &str = "\x1b[0m";
    /// Bold text.
    pub const BOLD: &str = "\x1b[1m";
    /// Dim/faint text.
    pub const DIM: &str = "\x1b[2m";
    /// Green color.
    pub const GREEN: &str = "\x1b[32m";
    /// Yellow color.
    pub const YELLOW: &str = "\x1b[33m";
    /// Blue color.
    pub const BLUE: &str = "\x1b[34m";
    /// Magenta color.
    pub const MAGENTA: &str = "\x1b[35m";
    /// Cyan color.
    pub const CYAN: &str = "\x1b[36m";
    /// Red color.
    pub const RED: &str = "\x1b[31m";
}

/// Check if the terminal supports colors.
pub fn supports_color() -> bool {
    // Check common environment variables
    if std::env::var("NO_COLOR").is_ok() {
        return false;
    }
    if std::env::var("FORCE_COLOR").is_ok() {
        return true;
    }

    // Check if stdout is a terminal
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        unsafe { libc::isatty(io::stdout().as_raw_fd()) != 0 }
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Check if the terminal is Ghostty (supports OSC 9;4 progress).
pub fn is_ghostty() -> bool {
    std::env::var("TERM_PROGRAM")
        .map(|v| v.to_lowercase().contains("ghostty"))
        .unwrap_or(false)
        || std::env::var("GHOSTTY_RESOURCES_DIR").is_ok()
}

/// Terminal progress state for Ghostty OSC 9;4 protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalProgressState {
    /// Progress is ongoing (0-100%).
    InProgress(u8),
    /// Progress completed successfully.
    Done,
    /// Progress failed/errored.
    Error,
}

/// Set terminal progress using OSC 9;4 (Ghostty, ConEmu, etc.).
///
/// This updates the terminal's progress indicator (e.g., in the tab bar or title).
pub fn set_terminal_progress(state: TerminalProgressState) {
    if !is_ghostty() && std::env::var("ConEmuANSI").is_err() {
        return;
    }

    let osc = match state {
        TerminalProgressState::InProgress(pct) => format!("\x1b]9;4;1;{}\x07", pct.min(100)),
        TerminalProgressState::Done => "\x1b]9;4;0;0\x07".to_string(),
        TerminalProgressState::Error => "\x1b]9;4;2;0\x07".to_string(),
    };

    let _ = io::stderr().write_all(osc.as_bytes());
    let _ = io::stderr().flush();
}

/// Clear terminal progress indicator.
pub fn clear_terminal_progress() {
    set_terminal_progress(TerminalProgressState::Done);
}

/// Progress callback for download operations.
pub type ProgressCallback = Box<dyn Fn(DownloadProgress) + Send + Sync>;

/// Download progress information.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    /// Bytes downloaded so far.
    pub downloaded: u64,
    /// Total bytes to download (if known).
    pub total: Option<u64>,
    /// Download speed in bytes per second.
    pub speed: u64,
    /// Estimated time remaining.
    pub eta: Option<Duration>,
}

impl DownloadProgress {
    /// Get progress as a percentage (0-100).
    pub fn percent(&self) -> Option<u8> {
        self.total
            .map(|t| ((self.downloaded as f64 / t as f64) * 100.0) as u8)
    }

    /// Format the downloaded amount as human-readable string.
    pub fn downloaded_str(&self) -> String {
        format_bytes(self.downloaded)
    }

    /// Format the total size as human-readable string.
    pub fn total_str(&self) -> Option<String> {
        self.total.map(format_bytes)
    }

    /// Format the speed as human-readable string.
    pub fn speed_str(&self) -> String {
        format!("{}/s", format_bytes(self.speed))
    }

    /// Format ETA as human-readable string.
    pub fn eta_str(&self) -> Option<String> {
        self.eta.map(format_duration)
    }
}

/// Format bytes as human-readable string.
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format duration as human-readable string.
pub fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs >= 3600 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}s", secs)
    }
}

/// A styled message writer for terminal output.
pub struct TerminalOutput {
    use_color: bool,
}

impl Default for TerminalOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalOutput {
    /// Create a new terminal output handler.
    pub fn new() -> Self {
        Self {
            use_color: supports_color(),
        }
    }

    /// Create a terminal output handler with explicit color setting.
    pub fn with_color(use_color: bool) -> Self {
        Self { use_color }
    }

    /// Print an info message.
    pub fn info(&self, message: &str) {
        if self.use_color {
            eprintln!(
                "{}{} {}{} {}",
                colors::BLUE,
                icons::INFO,
                colors::BOLD,
                message,
                colors::RESET
            );
        } else {
            eprintln!("{} {}", icons::INFO, message);
        }
    }

    /// Print a success message.
    pub fn success(&self, message: &str) {
        if self.use_color {
            eprintln!(
                "{}{} {}{} {}",
                colors::GREEN,
                icons::SUCCESS,
                colors::BOLD,
                message,
                colors::RESET
            );
        } else {
            eprintln!("{} {}", icons::SUCCESS, message);
        }
    }

    /// Print an error message.
    pub fn error(&self, message: &str) {
        if self.use_color {
            eprintln!(
                "{}{} {}{} {}",
                colors::RED,
                icons::ERROR,
                colors::BOLD,
                message,
                colors::RESET
            );
        } else {
            eprintln!("{} {}", icons::ERROR, message);
        }
    }

    /// Print a warning message.
    pub fn warning(&self, message: &str) {
        if self.use_color {
            eprintln!(
                "{}{} {}{} {}",
                colors::YELLOW,
                icons::WARNING,
                colors::BOLD,
                message,
                colors::RESET
            );
        } else {
            eprintln!("{} {}", icons::WARNING, message);
        }
    }

    /// Print a dim/secondary message.
    pub fn dim(&self, message: &str) {
        if self.use_color {
            eprintln!("  {}{}{}", colors::DIM, message, colors::RESET);
        } else {
            eprintln!("  {}", message);
        }
    }

    /// Print a download starting message.
    pub fn download_start(&self, name: &str, url: &str) {
        if self.use_color {
            eprintln!(
                "{}{} {}{} Downloading {}",
                colors::CYAN,
                icons::DOWNLOAD,
                colors::BOLD,
                name,
                colors::RESET
            );
            eprintln!(
                "  {}{}{}",
                colors::DIM,
                truncate_url(url, 60),
                colors::RESET
            );
        } else {
            eprintln!("{} Downloading {}", icons::DOWNLOAD, name);
            eprintln!("  {}", truncate_url(url, 60));
        }
    }

    /// Print download progress.
    pub fn download_progress(&self, progress: &DownloadProgress) {
        let bar = render_progress_bar(progress.percent().unwrap_or(0), 30);
        let pct = progress.percent().unwrap_or(0);

        // Update terminal progress (Ghostty)
        set_terminal_progress(TerminalProgressState::InProgress(pct));

        // Render inline progress
        if self.use_color {
            eprint!(
                "\r  {} {}{:>3}%{} {} {}{}{}  ",
                bar,
                colors::BOLD,
                pct,
                colors::RESET,
                progress.downloaded_str(),
                colors::DIM,
                progress.speed_str(),
                colors::RESET
            );
        } else {
            eprint!(
                "\r  {} {:>3}% {} {}  ",
                bar,
                pct,
                progress.downloaded_str(),
                progress.speed_str()
            );
        }
        let _ = io::stderr().flush();
    }

    /// Print download complete message.
    pub fn download_complete(&self, path: &str) {
        // Clear progress line
        eprint!("\r{}\r", " ".repeat(80));

        // Clear terminal progress
        clear_terminal_progress();

        if self.use_color {
            eprintln!(
                "  {}{} Downloaded successfully{}",
                colors::GREEN,
                icons::SUCCESS,
                colors::RESET
            );
            eprintln!("  {}Location: {}{}", colors::DIM, path, colors::RESET);
        } else {
            eprintln!("  {} Downloaded successfully", icons::SUCCESS);
            eprintln!("  Location: {}", path);
        }
    }

    /// Print download error message.
    pub fn download_error(&self, error: &str) {
        // Clear progress line
        eprint!("\r{}\r", " ".repeat(80));

        // Set error state
        set_terminal_progress(TerminalProgressState::Error);

        if self.use_color {
            eprintln!(
                "  {}{} Download failed: {}{}",
                colors::RED,
                icons::ERROR,
                error,
                colors::RESET
            );
        } else {
            eprintln!("  {} Download failed: {}", icons::ERROR, error);
        }

        // Clear terminal progress after a moment
        clear_terminal_progress();
    }

    /// Print model loading message.
    pub fn model_loading(&self, path: &str) {
        if self.use_color {
            eprintln!(
                "{}{} {}Loading model...{}",
                colors::MAGENTA,
                icons::MODEL,
                colors::BOLD,
                colors::RESET
            );
            eprintln!("  {}{}{}", colors::DIM, path, colors::RESET);
        } else {
            eprintln!("{} Loading model...", icons::MODEL);
            eprintln!("  {}", path);
        }
    }

    /// Print model loaded message.
    pub fn model_loaded(&self) {
        if self.use_color {
            eprintln!(
                "  {}{} Model ready{}",
                colors::GREEN,
                icons::SUCCESS,
                colors::RESET
            );
        } else {
            eprintln!("  {} Model ready", icons::SUCCESS);
        }
    }
}

/// Render a progress bar.
fn render_progress_bar(percent: u8, width: usize) -> String {
    let filled = (percent as usize * width) / 100;
    let empty = width.saturating_sub(filled);

    format!(
        "{}[{}{}]{}",
        colors::CYAN,
        "█".repeat(filled),
        "░".repeat(empty),
        colors::RESET
    )
}

/// Truncate a URL for display.
fn truncate_url(url: &str, max_len: usize) -> String {
    if url.len() <= max_len {
        return url.to_string();
    }

    // Try to keep the domain and end of the path
    let parts: Vec<&str> = url.splitn(4, '/').collect();
    if parts.len() >= 4 {
        let domain = format!("{}//{}", parts[0], parts[2]);
        let remaining = max_len.saturating_sub(domain.len() + 4);
        let path = parts[3];
        if path.len() > remaining {
            format!(
                "{}/...{}",
                domain,
                &path[path.len().saturating_sub(remaining)..]
            )
        } else {
            format!("{}/{}", domain, path)
        }
    } else {
        format!("{}...", &url[..max_len.saturating_sub(3)])
    }
}

/// Progress tracker for download operations.
pub struct DownloadTracker {
    #[allow(dead_code)]
    start_time: Instant,
    last_update: Instant,
    last_downloaded: u64,
    total: Option<u64>,
    output: TerminalOutput,
    min_update_interval: Duration,
}

impl DownloadTracker {
    /// Create a new download tracker.
    pub fn new(total: Option<u64>) -> Self {
        Self {
            start_time: Instant::now(),
            last_update: Instant::now(),
            last_downloaded: 0,
            total,
            output: TerminalOutput::new(),
            min_update_interval: Duration::from_millis(100),
        }
    }

    /// Update the download progress.
    pub fn update(&mut self, downloaded: u64) {
        let now = Instant::now();

        // Rate limit updates
        if now.duration_since(self.last_update) < self.min_update_interval {
            return;
        }

        // Calculate speed
        let elapsed = now.duration_since(self.last_update);
        let bytes_delta = downloaded.saturating_sub(self.last_downloaded);
        let speed = if elapsed.as_secs_f64() > 0.0 {
            (bytes_delta as f64 / elapsed.as_secs_f64()) as u64
        } else {
            0
        };

        // Calculate ETA
        let eta = if let Some(total) = self.total {
            let remaining = total.saturating_sub(downloaded);
            if speed > 0 {
                Some(Duration::from_secs(remaining / speed))
            } else {
                None
            }
        } else {
            None
        };

        let progress = DownloadProgress {
            downloaded,
            total: self.total,
            speed,
            eta,
        };

        self.output.download_progress(&progress);

        self.last_update = now;
        self.last_downloaded = downloaded;
    }

    /// Mark download as complete.
    pub fn complete(&self, path: &str) {
        self.output.download_complete(path);
    }

    /// Mark download as failed.
    pub fn error(&self, error: &str) {
        self.output.download_error(error);
    }

    /// Get the terminal output handler.
    pub fn output(&self) -> &TerminalOutput {
        &self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m 30s");
        assert_eq!(format_duration(Duration::from_secs(3700)), "1h 1m");
    }

    #[test]
    fn test_download_progress_percent() {
        let progress = DownloadProgress {
            downloaded: 50,
            total: Some(100),
            speed: 10,
            eta: None,
        };
        assert_eq!(progress.percent(), Some(50));

        let progress_no_total = DownloadProgress {
            downloaded: 50,
            total: None,
            speed: 10,
            eta: None,
        };
        assert_eq!(progress_no_total.percent(), None);
    }

    #[test]
    fn test_truncate_url() {
        let short_url = "https://example.com/file.txt";
        assert_eq!(truncate_url(short_url, 100), short_url);

        let long_url =
            "https://huggingface.co/chalyi/clitron-gh/resolve/main/clitron-gh-q4_k_m.gguf";
        let truncated = truncate_url(long_url, 50);
        assert!(truncated.len() <= 55); // Allow some slack for formatting
        assert!(truncated.contains("..."));
    }
}
