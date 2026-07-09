namespace Ferron.Tests;

// Test-only behaviours driven by the Rust integration test
// (crates/ferron-script/tests/lifecycle.rs), which swaps the engine's log
// callback for a capture buffer and asserts on hook ordering. They ship in
// Ferron.dll because the host can only instantiate types from the assembly it
// loads; they are never attached by the engine itself.

/// Logs every lifecycle hook with a stable `probe:` prefix.
class LifecycleProbe : Behaviour
{
    public LifecycleProbe() => Native.Log("probe:ctor");
    public override void OnEnable() => Native.Log("probe:OnEnable");
    public override void OnStart() => Native.Log("probe:OnStart");
    public override void OnUpdate(float deltaTime) => Native.Log("probe:OnUpdate");
    public override void OnDisable() => Native.Log("probe:OnDisable");
    public override void OnDestroy() => Native.Log("probe:OnDestroy");
}

/// Exercises the `Create` guard: a user constructor that throws must yield a
/// 0 handle, not a process abort.
class ThrowingConstructor : Behaviour
{
    public ThrowingConstructor() => throw new InvalidOperationException("ctor boom");
}

/// Exercises the `Destroy` guard: a throwing OnDestroy must be logged and the
/// GCHandle still freed.
class ThrowingDestroy : Behaviour
{
    public override void OnDestroy() => throw new InvalidOperationException("destroy boom");
}
