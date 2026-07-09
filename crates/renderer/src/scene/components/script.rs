/// Links an entity to its C# Behaviour via a .NET `GCHandle` (stored as `u64`).
///
/// Deliberately **not** `Clone`/`Copy`: the handle is uniquely owned and [`Drop`]
/// tears it down, so a copy would destroy the same managed object twice.
///
/// Two booleans track activation, deliberately split:
/// - `enabled` is the *desired* state — whoever wants the script on or off
///   (editor, engine code, another script) writes this and nothing else.
/// - `active` is the state the managed side last *observed* — only the script
///   tick writes it, after actually dispatching OnEnable/OnDisable.
///
/// The tick diffs the two to decide which transitions to fire; collapsing them
/// into one flag would make "flag changed" and "script was told" the same
/// event, and they aren't (the change can happen mid-frame, after this tick's
/// dispatch window closed).
#[derive(Debug)]
pub struct ScriptComponent {
    pub handle: u64,
    pub started: bool,
    pub enabled: bool,
    pub active: bool,
}

impl Drop for ScriptComponent {
    fn drop(&mut self) {
        // Managed side fires OnDisable (if still active) and OnDestroy, then
        // frees the GCHandle — ordering lives in C# `Behaviours.Destroy`.
        // Runs outside a dispatch window, so those callbacks get no world
        // access (see `destroy_handle` docs).
        ferron_script::destroy_handle(self.handle);
    }
}
