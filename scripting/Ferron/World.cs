namespace Ferron;

// Entity-level world operations. Structural changes (spawns, despawns) are
// queued engine-side and applied after the current script tick, but the
// returned Entity handle is real and usable immediately. One consequence: a
// renderable spawned this tick has no readable Transform until next tick —
// keep your own copy if you need it in the same frame.
public static class World
{
    /// Spawn an entity rendered with a mesh and material registered in the
    /// engine's asset registry (meshes: "cube", "sphere", "plane"; materials:
    /// "gold", "copper", "glossy", "clay", "neon", "textured", "rock",
    /// "ground"). Logs and returns Entity(0v0) if a name is unknown.
    public static Entity SpawnRenderable(string mesh, string material, Transform transform) =>
        Native.SpawnRenderable(mesh, material, transform);

    /// Spawn an empty entity.
    public static Entity Spawn() => Native.Spawn();

    /// Queue the entity for despawn at the end of this tick. Returns false if
    /// the handle was already stale.
    public static bool Despawn(Entity entity) => Native.Despawn(entity);
}
