using System.Runtime.InteropServices;

namespace Ferron;

public static unsafe class Behaviours
{
    [UnmanagedCallersOnly]
    public static nint Create(Entity entity, byte* typeName)
    {
        var name = Marshal.PtrToStringUTF8((nint)typeName);
        if (name is null)
            return 0;

        var type = ResolveType(name);
        if (type is null || Activator.CreateInstance(type) is not Behaviour behaviour)
            return 0;

        behaviour.Entity = entity;
        return GCHandle.ToIntPtr(GCHandle.Alloc(behaviour));
    }

    [UnmanagedCallersOnly]
    public static void Start(nint handle)
    {
        if (handle != 0 && GCHandle.FromIntPtr(handle).Target is Behaviour behaviour)
            behaviour.OnStart();
    }

    [UnmanagedCallersOnly]
    public static void Update(nint handle, float deltaTime)
    {
        if (handle != 0 && GCHandle.FromIntPtr(handle).Target is Behaviour behaviour)
            behaviour.OnUpdate(deltaTime);
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
