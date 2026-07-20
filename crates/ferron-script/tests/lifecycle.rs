//! End-to-end Behaviour lifecycle test: boots CoreCLR against the built
//! `Ferron` assembly, drives the dispatch entry points the way the engine's
//! script tick does, and asserts hook ordering through a captured log
//! callback (scripts log via the `FerronApi` table, so the test supplies its
//! own `log` and reads the calls back).
//!
//! Everything lives in one `#[test]`: CoreCLR can only boot once per process.
//!
//! Requires `dotnet build scripting/Ferron` first. Skips with a note when the
//! assembly is missing — this crate is workspace-excluded, so CI never runs it.

use std::ffi::{c_char, CStr, CString};
use std::path::PathBuf;
use std::sync::Mutex;

use ferron_script::{CEntity, ScriptHost};

static LOGS: Mutex<Vec<String>> = Mutex::new(Vec::new());

extern "C" fn capture_log(message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: C# passes a valid, null-terminated UTF-8 buffer.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    LOGS.lock().unwrap().push(text.into_owned());
}

/// Drain and return everything logged since the last call.
fn take_logs() -> Vec<String> {
    std::mem::take(&mut *LOGS.lock().unwrap())
}

/// `FERRON_SCRIPT_DIR` override, else probe the C# build output relative to
/// this crate's root (where `cargo test` runs).
fn assembly_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("FERRON_SCRIPT_DIR") {
        return Some(PathBuf::from(dir));
    }
    let bin = PathBuf::from("../../scripting/Ferron/bin");
    for config in ["Debug", "Release"] {
        let Ok(entries) = std::fs::read_dir(bin.join(config)) else {
            continue;
        };
        for entry in entries.flatten() {
            let dir = entry.path();
            if dir.join("Ferron.dll").is_file() && dir.join("Ferron.runtimeconfig.json").is_file()
            {
                return Some(dir);
            }
        }
    }
    None
}

fn create(host: &ScriptHost, type_name: &str) -> u64 {
    let name = CString::new(type_name).unwrap();
    host.create(
        CEntity {
            index: 1,
            generation: 0,
        },
        &name,
    )
}

#[test]
fn behaviour_lifecycle() {
    let Some(dir) = assembly_dir() else {
        eprintln!("skipping: no built Ferron assembly (run `dotnet build scripting/Ferron`)");
        return;
    };
    let api = ferron_script::FerronApi {
        log: capture_log,
        ..ferron_script::default_api()
    };
    let host = ScriptHost::boot(&api, &dir).expect("CoreCLR failed to boot");
    take_logs(); // discard boot chatter

    let probe = "Ferron.Tests.LifecycleProbe, Ferron";

    // Full happy path, plus redundant transitions being no-ops. The final
    // destroy happens while inactive, so no second OnDisable is owed.
    let handle = create(&host, probe);
    assert_ne!(handle, 0, "probe type not found in Ferron.dll");
    host.enable(handle);
    host.start(handle);
    host.update(handle, 0.016);
    host.enable(handle); // already active: OnEnable must not re-fire
    host.disable(handle);
    host.disable(handle); // already inactive: OnDisable must not re-fire
    ferron_script::destroy_handle(handle);
    assert_eq!(
        take_logs(),
        [
            "probe:ctor",
            "probe:OnEnable",
            "probe:OnStart",
            "probe:OnUpdate",
            "probe:OnDisable",
            "probe:OnDestroy",
        ],
    );

    // Destroy while still active owes OnDisable first, then OnDestroy, then
    // the free — the ordering guarantee this branch exists to establish.
    let handle = create(&host, probe);
    host.enable(handle);
    ferron_script::destroy_handle(handle);
    assert_eq!(
        take_logs(),
        ["probe:ctor", "probe:OnEnable", "probe:OnDisable", "probe:OnDestroy"],
    );

    // A throwing user constructor is contained: 0 handle, logged, no abort.
    let handle = create(&host, "Ferron.Tests.ThrowingConstructor, Ferron");
    assert_eq!(handle, 0);
    let logs = take_logs();
    assert!(
        logs.iter().any(|l| l.contains("exception during create")),
        "expected contained create exception, got {logs:?}"
    );

    // A throwing OnDestroy is contained; reaching the next assertion at all
    // proves the exception never crossed the native boundary.
    let handle = create(&host, "Ferron.Tests.ThrowingDestroy, Ferron");
    assert_ne!(handle, 0);
    ferron_script::destroy_handle(handle);
    let logs = take_logs();
    assert!(
        logs.iter().any(|l| l.contains("exception during destroy")),
        "expected contained destroy exception, got {logs:?}"
    );

    // The fault channel. A clean OnUpdate reports no fault; a throwing one is
    // contained, reports the fault back (so the engine can disable the script),
    // and logs naming the offending type and hook.
    let handle = create(&host, probe);
    host.enable(handle);
    assert!(!host.update(handle, 0.016), "a clean OnUpdate must not report a fault");
    ferron_script::destroy_handle(handle);
    take_logs(); // discard the probe's own hook chatter

    let handle = create(&host, "Ferron.Tests.ThrowingUpdate, Ferron");
    assert_ne!(handle, 0);
    assert!(host.update(handle, 0.016), "a throwing OnUpdate must report a fault");
    let logs = take_logs();
    assert!(
        logs.iter().any(|l| l.contains("ThrowingUpdate") && l.contains("OnUpdate")),
        "expected a fault log naming the script type and hook, got {logs:?}"
    );
    ferron_script::destroy_handle(handle);
}
