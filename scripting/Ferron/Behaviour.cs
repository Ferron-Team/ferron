namespace Ferron;

public abstract class Behaviour
{
    public Entity Entity { get; internal set; }

    /// Whether the behaviour is currently active, as last dispatched by the
    /// engine. Owned here (not only in Rust) so the destroy path can decide
    /// whether OnDisable is still owed without a round-trip.
    internal bool Active;

    public virtual void OnEnable() { }

    public virtual void OnStart() { }

    public virtual void OnUpdate(float deltaTime) { }

    public virtual void OnDisable() { }

    public virtual void OnDestroy() { }

    protected Transform Transform
    {
        get => Native.GetTransform(Entity);
        set => Native.SetTransform(Entity, value);
    }
}
