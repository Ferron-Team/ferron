//! Runtime performance telemetry.
//!
//! [`FrameStats`] is a world resource updated once per frame from the app loop
//! (see `App`'s redraw branch). It keeps a short history of frame times for the
//! editor's performance graph and samples this process's memory a few times a
//! second. Reading it is cheap; the editor's `performance` panel renders it.

use std::collections::VecDeque;

use sysinfo::{Pid, ProcessesToUpdate, System};

/// How many recent frames to keep for the graph (~2-4 s of history).
const HISTORY: usize = 240;

/// Smoothing factor for the displayed frame-time (exponential moving average).
/// Smaller = steadier number, slower to react.
const EMA_ALPHA: f32 = 0.1;

/// How often (seconds) to re-sample process memory. Querying the OS every frame
/// is wasteful and the number barely moves frame-to-frame.
const MEM_REFRESH_SECS: f32 = 0.5;

/// Frame-timing history and process memory, updated once per frame.
pub struct FrameStats {
    /// Recent frame durations in milliseconds, oldest first.
    frame_ms: VecDeque<f32>,
    /// Smoothed frame time in milliseconds, for a stable on-screen readout.
    avg_ms: f32,
    /// Resident set size of this process in bytes (refreshed periodically).
    memory_bytes: u64,
    /// Seconds accumulated since the last memory sample.
    mem_accum: f32,
    sys: System,
    pid: Option<Pid>,
    /// Whole-frame GPU time in ms, `None` if the device can't report it.
    gpu_ms: Option<f32>,
    /// Recent GPU frame times, parallel to `frame_ms`, for the graph.
    gpu_ms_history: VecDeque<f32>,
    /// Device-local VRAM `(used, total)` bytes; `used` is `None` if unsupported.
    vram_used: Option<u64>,
    vram_total: u64,
}

impl Default for FrameStats {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameStats {
    pub fn new() -> Self {
        Self {
            frame_ms: VecDeque::with_capacity(HISTORY),
            avg_ms: 0.0,
            memory_bytes: 0,
            // Sample memory on the very first frame.
            mem_accum: MEM_REFRESH_SECS,
            sys: System::new(),
            pid: sysinfo::get_current_pid().ok(),
            gpu_ms: None,
            gpu_ms_history: VecDeque::with_capacity(HISTORY),
            vram_used: None,
            vram_total: 0,
        }
    }

    /// Record one frame of `dt` seconds. Call once per frame.
    pub fn record(&mut self, dt: f32) {
        let ms = dt * 1000.0;
        if self.frame_ms.len() == HISTORY {
            self.frame_ms.pop_front();
        }
        self.frame_ms.push_back(ms);

        self.avg_ms = if self.avg_ms == 0.0 {
            ms
        } else {
            self.avg_ms + EMA_ALPHA * (ms - self.avg_ms)
        };

        self.mem_accum += dt;
        if self.mem_accum >= MEM_REFRESH_SECS {
            self.mem_accum = 0.0;
            self.refresh_memory();
        }
    }

    fn refresh_memory(&mut self) {
        let Some(pid) = self.pid else {
            return;
        };
        self.sys
            .refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        if let Some(proc) = self.sys.process(pid) {
            self.memory_bytes = proc.memory();
        }
    }

    /// Record GPU timing and VRAM for this frame, sourced from the renderer.
    pub fn set_gpu_stats(
        &mut self,
        gpu_ms: Option<f32>,
        vram_used: Option<u64>,
        vram_total: u64,
    ) {
        self.gpu_ms = gpu_ms;
        if let Some(ms) = gpu_ms {
            if self.gpu_ms_history.len() == HISTORY {
                self.gpu_ms_history.pop_front();
            }
            self.gpu_ms_history.push_back(ms);
        }
        self.vram_used = vram_used;
        self.vram_total = vram_total;
    }

    /// Whole-frame GPU time in ms, if the device reports it.
    pub fn gpu_ms(&self) -> Option<f32> {
        self.gpu_ms
    }

    /// GPU frame-time history (oldest first), for the graph.
    pub fn gpu_history(&self) -> &VecDeque<f32> {
        &self.gpu_ms_history
    }

    /// Device-local VRAM in use, in bytes, if the driver reports it.
    pub fn vram_used(&self) -> Option<u64> {
        self.vram_used
    }

    /// Total device-local VRAM in bytes.
    pub fn vram_total(&self) -> u64 {
        self.vram_total
    }

    /// Smoothed frames per second.
    pub fn fps(&self) -> f32 {
        if self.avg_ms > 0.0 {
            1000.0 / self.avg_ms
        } else {
            0.0
        }
    }

    /// Smoothed frame time in milliseconds.
    pub fn frame_ms(&self) -> f32 {
        self.avg_ms
    }

    /// (min, max) frame time over the kept history, in milliseconds. Returns
    /// `(0, 0)` while no frames have been recorded yet.
    pub fn min_max_ms(&self) -> (f32, f32) {
        if self.frame_ms.is_empty() {
            return (0.0, 0.0);
        }
        self.frame_ms
            .iter()
            .copied()
            .fold((f32::MAX, 0.0), |(lo, hi), ms| (lo.min(ms), hi.max(ms)))
    }

    /// Resident memory of this process in bytes (0 if unavailable).
    pub fn memory_bytes(&self) -> u64 {
        self.memory_bytes
    }

    /// Frame-time history (oldest first), for the graph.
    pub fn history(&self) -> &VecDeque<f32> {
        &self.frame_ms
    }
}
