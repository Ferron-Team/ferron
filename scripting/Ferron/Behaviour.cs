namespace Ferron;

public abstract class Behaviour
{
    public Entity Entity { get; internal set; }

    public virtual void OnStart() { }

    public virtual void OnUpdate(float deltaTime) { }

    protected Transform Transform
    {
        get => Native.GetTransform(Entity);
        set => Native.SetTransform(Entity, value);
    }
}
