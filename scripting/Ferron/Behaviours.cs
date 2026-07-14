using System.Runtime.InteropServices;

namespace Ferron;

/// Lifecycle of a Behaviour, in dispatch order:
///
///   1. ctor + Entity assignment   — `Create`, when the engine attaches the script
///   2. OnEnable                   — first activation, and every re-activation
///   3. OnStart                    — once, on the first tick the behaviour is active
///   4. OnUpdate                   — every tick while active
///   5. OnDisable                  — every deactivation; also fired by `Destroy`
///                                   if the behaviour is still active
///   6. OnDestroy                  — once, just before the GCHandle is freed
///   7. GCHandle.Free              — last step of `Destroy`; the object is
///                                   unreachable from Rust after this
///
/// Steps 2–5 can cycle (enable/disable). `Destroy` is the only path that frees
/// the handle, so OnDestroy always precedes the free by construction.
public static unsafe class Behaviours
{
    [UnmanagedCallersOnly]
    public static nint Create(Entity entity, byte* typeName)
    {
        var name = Marshal.PtrToStringUTF8((nint)typeName);
        if (name is null)
            return 0;

        try
        {
            var type = ResolveType(name);
            if (type is null || Activator.CreateInstance(type) is not Behaviour behaviour)
                return 0;

            behaviour.Entity = entity;
            return GCHandle.ToIntPtr(GCHandle.Alloc(behaviour));
        }
        catch (Exception e)
        {
            // A throwing user constructor must not escape to native code; 0 is
            // the existing "creation failed, don't attach" contract.
            Native.Log($"[script] exception during create of {name}: {e}");
            return 0;
        }
    }

    [UnmanagedCallersOnly]
    public static void Start(nint handle)
    {
        try
        {
            if (handle != 0 && GCHandle.FromIntPtr(handle).Target is Behaviour behaviour)
                behaviour.OnStart();
        }
        catch (Exception e)
        {
            Native.Log($"[script] exception during start: {e}");
        }
    }

    [UnmanagedCallersOnly]
    public static void Update(nint handle, float deltaTime)
    {
        try
        {
            if (handle != 0 && GCHandle.FromIntPtr(handle).Target is Behaviour behaviour)
                behaviour.OnUpdate(deltaTime);
        }
        catch (Exception e)
        {
            Native.Log($"[script] exception during update: {e}");
        }
    }

    [UnmanagedCallersOnly]
    public static void Enable(nint handle)
    {
        if (handle != 0 && GCHandle.FromIntPtr(handle).Target is Behaviour behaviour)
        {
            try
            {
                if (!behaviour.Active)
                {
                    behaviour.Active = true;
                    behaviour.OnEnable();
                }
            }
            catch (Exception e)
            {
                Native.Log($"[script] exception during enable: {e}");
            }
        }
    }

    [UnmanagedCallersOnly]
    public static void Disable(nint handle)
    {
        if (handle != 0 && GCHandle.FromIntPtr(handle).Target is Behaviour behaviour)
        {
            try
            {
                if (behaviour.Active)
                {
                    behaviour.Active = false;
                    behaviour.OnDisable();
                }
            }
            catch (Exception e)
            {
                Native.Log($"[script] exception during disable: {e}");
            }
        }
    }

    [UnmanagedCallersOnly]
    public static void CollisionEnter(nint handle, Collision* collision)
    {
        try
        {
            if (handle != 0 && GCHandle.FromIntPtr(handle).Target is Behaviour behaviour)
                behaviour.OnCollisionEnter(*collision);
        }
        catch (Exception e)
        {
            Native.Log($"[script] exception during collision enter: {e}");
        }
    }

    [UnmanagedCallersOnly]
    public static void CollisionExit(nint handle, Collision* collision)
    {
        try
        {
            if (handle != 0 && GCHandle.FromIntPtr(handle).Target is Behaviour behaviour)
                behaviour.OnCollisionExit(*collision);
        }
        catch (Exception e)
        {
            Native.Log($"[script] exception during collision exit: {e}");
        }
    }

    /// Tears the behaviour down and frees its GCHandle. This is the single
    /// managed release point: Rust's `ScriptComponent::drop` lands here.
    [UnmanagedCallersOnly]
    public static void Destroy(nint handle)
    {
        if (handle != 0 && GCHandle.FromIntPtr(handle).Target is Behaviour behaviour)
        {
            try
            {
                if (behaviour.Active)
                {
                    behaviour.Active = false;
                    behaviour.OnDisable();
                }
                behaviour.OnDestroy();
            }
            catch (Exception e)
            {
                Native.Log($"[script] exception during destroy: {e}");
            }
            finally
            {
                GCHandle.FromIntPtr(handle).Free();
            }
        }
    }

    static Type? ResolveType(string name)
    {
        var type = Type.GetType(name);
        if (type is not null)
            return type;

        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            type = assembly.GetType(name);
            if (type is not null)
                return type;
        }
        return null;
    }
}
