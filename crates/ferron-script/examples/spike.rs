//! Standalone lifecycle smoke test: boot CoreCLR, create a Behaviour, tick it.
//! Run with the cwd set to the built `Ferron.dll` directory
//! (`scripting/Ferron/bin/Debug/net10.0`).

use std::ffi::CString;
use std::path::Path;

use ferron_script::{default_api, CEntity, ScriptHost};

fn main() {
    let host = match ScriptHost::boot(&default_api(), Path::new(".")) {
        Ok(host) => host,
        Err(err) => {
            eprintln!("scripting host failed: {err}");
            std::process::exit(1);
        }
    };

    let type_name = CString::new("Ferron.Demo.Spinner, Ferron").unwrap();
    let handle = host.create(CEntity::NULL, &type_name);
    println!("[host] created behaviour handle = {handle:#x}");

    host.start(handle);
    for _ in 0..3 {
        host.update(handle, 0.016);
    }
}
