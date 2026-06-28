/// Links an entity to its C# Behaviour via a .NET `GCHandle` (stored as `u64`).
///
/// Deliberately **not** `Clone`/`Copy`: the handle is uniquely owned and [`Drop`]
/// frees it, so a copy would free the same managed object twice.
#[derive(Debug)]
pub struct ScriptComponent {
    pub handle: u64,
    pub started: bool,
}

impl Drop for ScriptComponent {
    fn drop(&mut self) {
        ferron_script::free_handle(self.handle);
    }
}
