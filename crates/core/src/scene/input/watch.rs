//! Reload-on-save for the binding file.
//!
//! The same shape as the script watcher, and for the same reason: a change
//! raises a request, and the swap happens at one known point in the frame
//! rather than under whatever was mid-read when the OS delivered the event.
//!
//! **A file that does not parse is inert.** The session keeps the bindings it
//! already had and the failure is text in the console, exactly as a red script
//! build leaves the previous assembly running. Unbinding every control because
//! someone saved halfway through typing would be a worse answer than ignoring
//! the save.
//!
//! The parent directory is watched rather than the file, because editors
//! replace a file by writing a temporary and renaming over it — a watch on the
//! original inode survives one save and then hears nothing.
//!
//! Live reload needs `notify`, which this build only has with the `scripting`
//! feature; without it the file is read once at startup.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use notify::{RecommendedWatcher, RecursiveMode, Watcher};

use orrin_ecs::World;

use super::Actions;
use crate::scene::{LogBuffer, LogLevel, Time};

/// How long the file must stay quiet before it is re-read. Shorter than the
/// script watcher's: there is one small file here and no compiler behind it, so
/// the wait only has to outlast a write-then-rename.
const DEBOUNCE: Duration = Duration::from_millis(150);

pub struct ConfigWatcher {
    /// Held only for its `Drop`, which stops the OS watch. Never read.
    _watcher: RecommendedWatcher,
    changes: Receiver<()>,
    path: PathBuf,
    pending_since: Option<Instant>,
}

impl ConfigWatcher {
    pub fn new(path: &Path) -> Result<Self, String> {
        let directory = path
            .parent()
            .ok_or_else(|| format!("{} has no directory to watch", path.display()))?;
        let target = path.to_path_buf();
        let (changes_tx, changes) = mpsc::channel();

        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                let Ok(event) = event else { return };
                if !(event.kind.is_create() || event.kind.is_modify()) {
                    return;
                }
                if event.paths.contains(&target) {
                    let _ = changes_tx.send(());
                }
            })
            .map_err(|err| format!("could not watch {}: {err}", directory.display()))?;
        watcher
            .watch(directory, RecursiveMode::NonRecursive)
            .map_err(|err| format!("could not watch {}: {err}", directory.display()))?;

        Ok(Self {
            _watcher: watcher,
            changes,
            path: path.to_path_buf(),
            pending_since: None,
        })
    }

    /// Re-read the file if it has settled since it last changed.
    pub fn service(&mut self, world: &World) {
        if self.changes.try_iter().count() > 0 {
            self.pending_since = Some(Instant::now());
        }
        let Some(since) = self.pending_since else {
            return;
        };
        if since.elapsed() < DEBOUNCE {
            return;
        }
        self.pending_since = None;

        let frame = world.resource::<Time>().frame_count();
        let mut log = world.resource_mut::<LogBuffer>();
        match super::config::load(&self.path) {
            Ok(specs) => {
                let count = specs.len();
                world.resource_mut::<Actions>().apply(specs);
                log.push(
                    LogLevel::Info,
                    format!("input bindings reloaded: {count} definitions"),
                    frame,
                );
            }
            Err(error) => log.push(
                LogLevel::Error,
                format!("input bindings unchanged: {error}"),
                frame,
            ),
        }
    }
}
