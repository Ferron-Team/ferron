//! What the device is, and the handful of renderer choices that turn on it.
//!
//! Vulkan is one API over hardware that disagrees about what is cheap, and the
//! two vendor guides this module is written against ([AMD's RDNA performance
//! guide][rdna] and [NVIDIA's Vulkan dos and don'ts][nv]) disagree in exactly
//! the places you would expect: how a barrier is priced, what a wavefront is,
//! and — the one this module actually decides — where a buffer the CPU writes
//! and the GPU reads should live.
//!
//! The rule for what belongs here: a choice earns a place only if the *right
//! answer differs* between devices. Everything the two guides agree on is not a
//! platform choice, it is just the correct code, and it lives at its own call
//! site with no branch. That is most of the advice — 8x8 compute groups, `LOAD_OP_CLEAR`
//! over a clear command, `D32_SFLOAT`, one submission per frame, push constants
//! for per-draw data — and none of it appears here.
//!
//! What is left is small, and this module keeps it in one place rather than
//! spread through the passes, because a vendor check hidden inside a pass is a
//! behaviour nobody can find later.
//!
//! [rdna]: https://gpuopen.com/learn/rdna-performance-guide/
//! [nv]: https://developer.nvidia.com/blog/vulkan-dos-donts/

use std::sync::Arc;

use vulkano::device::physical::PhysicalDevice;
use vulkano::memory::{MemoryHeapFlags, MemoryPropertyFlags};

/// Who made the GPU, by PCI vendor ID.
///
/// Carried for diagnostics rather than for branching: no decision below reads
/// it. That is deliberate — "is this an AMD card" is almost never the question
/// a renderer actually wants answered, and code that asks it ages badly, since
/// the thing it was standing in for (a BAR size, a subgroup width, an extension)
/// is nearly always queryable directly. It is reported at startup because a
/// performance number without the vendor beside it is not reproducible.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Vendor {
    Amd,
    Nvidia,
    Intel,
    Other(u32),
}

impl Vendor {
    /// PCI-SIG vendor IDs. Vulkan reports these verbatim in
    /// `VkPhysicalDeviceProperties::vendorID` for any real PCI device.
    pub fn from_id(id: u32) -> Self {
        match id {
            0x1002 | 0x1022 => Self::Amd,
            0x10DE => Self::Nvidia,
            0x8086 => Self::Intel,
            other => Self::Other(other),
        }
    }

    pub fn name(self) -> String {
        match self {
            Self::Amd => "AMD".into(),
            Self::Nvidia => "NVIDIA".into(),
            Self::Intel => "Intel".into(),
            Self::Other(id) => format!("vendor 0x{id:04X}"),
        }
    }
}

/// The device's memory shape and vendor, resolved once at startup.
///
/// Built from the memory properties rather than from a device name, because the
/// thing that varies is a BIOS setting and a driver version, not a model.
#[derive(Clone, Copy, Debug)]
pub struct GpuProfile {
    pub vendor: Vendor,
    /// Summed size of every `DEVICE_LOCAL` heap: what "VRAM" means here, and on
    /// a unified-memory part, system RAM.
    pub device_local: u64,
    /// The largest heap reachable through a memory type that is both
    /// `DEVICE_LOCAL` and `HOST_VISIBLE` — the window the CPU can write
    /// straight into video memory through.
    ///
    /// Measured per *type* rather than per heap on purpose. Drivers expose this
    /// as a memory type pointing into the ordinary VRAM heap (AMD's RADV lists
    /// one such type against the same 16 GiB heap as the device-local-only
    /// types), so counting heaps with both flags set would find nothing on the
    /// very hardware this is meant to detect.
    pub host_visible_device_local: u64,
}

impl GpuProfile {
    pub fn detect(physical: &Arc<PhysicalDevice>) -> Self {
        let props = physical.memory_properties();

        let device_local = props
            .memory_heaps
            .iter()
            .filter(|heap| heap.flags.intersects(MemoryHeapFlags::DEVICE_LOCAL))
            .map(|heap| heap.size)
            .sum();

        let both = MemoryPropertyFlags::DEVICE_LOCAL | MemoryPropertyFlags::HOST_VISIBLE;
        let host_visible_device_local = props
            .memory_types
            .iter()
            .filter(|ty| ty.property_flags.contains(both))
            .map(|ty| props.memory_heaps[ty.heap_index as usize].size)
            .max()
            .unwrap_or(0);

        Self {
            vendor: Vendor::from_id(physical.properties().vendor_id),
            device_local,
            host_visible_device_local,
        }
    }

    /// Whether the CPU can write into video memory at large — Resizable BAR on
    /// a discrete card, and unconditionally true on a unified-memory part, where
    /// there is no bus to cross in the first place.
    ///
    /// The comparison is a ratio rather than a byte count because the two
    /// configurations differ by orders of magnitude, not by a threshold: the
    /// legacy aperture is a fixed 256 MiB regardless of how much memory the card
    /// has, so on any card since the aperture was standardised it is a rounding
    /// error against the whole heap, while a resized bar covers all of it. Half
    /// is simply a number comfortably between "a rounding error" and "all of
    /// it"; nothing real is expected to land near it.
    ///
    /// `ORRIN_NARROW_BAR=1` forces the answer to `false`. That exists because the
    /// branch it controls is otherwise unreachable on the hardware most likely to
    /// be in front of whoever changes it: a machine with Resizable BAR on always
    /// takes the wide path, so the staged path could rot for a year before an
    /// NVIDIA desktop with the option off discovered the rot. Forcing it is how
    /// the path this engine cannot test on its own hardware stays a tested path.
    pub fn wide_bar(&self) -> bool {
        if std::env::var_os("ORRIN_NARROW_BAR").is_some_and(|forced| forced == "1") {
            return false;
        }
        self.host_visible_device_local * 2 >= self.device_local && self.device_local > 0
    }

    /// One line for the startup log, next to the device name.
    pub fn describe(&self) -> String {
        let gib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        format!(
            "{}, {:.1} GiB device-local, {:.1} GiB host-visible ({})",
            self.vendor.name(),
            gib(self.device_local),
            gib(self.host_visible_device_local),
            if self.wide_bar() {
                "wide BAR: CPU writes go straight to VRAM"
            } else {
                "narrow BAR: static geometry is staged"
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(device_local: u64, host_visible: u64) -> GpuProfile {
        GpuProfile {
            vendor: Vendor::Amd,
            device_local,
            host_visible_device_local: host_visible,
        }
    }

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    /// The two configurations the distinction exists for, at the sizes real
    /// hardware reports them.
    #[test]
    fn the_legacy_aperture_is_not_mistaken_for_a_resized_one() {
        // A 16 GiB card with Resizable BAR off: one 256 MiB window, fixed
        // regardless of the card's size, which is why no byte threshold would
        // separate these two cases across a hardware generation.
        assert!(!profile(16 * GIB, 256 * MIB).wide_bar());
        // The same card with it on.
        assert!(profile(16 * GIB, 16 * GIB).wide_bar());
        // A 2 GiB card with it off. Far smaller than the card above, and still
        // the same 256 MiB — a fixed cutoff tuned for the 16 GiB part would have
        // to sit below this to be right here, and then it would call the 16 GiB
        // card's aperture wide.
        assert!(!profile(2 * GIB, 256 * MIB).wide_bar());
    }

    /// Unified memory: every type is both, so there is no staging to do and
    /// nothing to detect.
    #[test]
    fn a_unified_memory_part_is_wide() {
        assert!(profile(8 * GIB, 8 * GIB).wide_bar());
    }

    /// A device advertising no host-visible video memory at all must not read as
    /// wide, and a device reporting no device-local memory must not divide the
    /// renderer by zero into calling itself either.
    #[test]
    fn the_degenerate_shapes_read_as_narrow() {
        assert!(!profile(16 * GIB, 0).wide_bar());
        assert!(!profile(0, 0).wide_bar());
    }
}
