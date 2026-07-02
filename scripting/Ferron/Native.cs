using System.Runtime.InteropServices;
using System.Text;

namespace Ferron;

// Field order and types must match the Rust FerronApi struct.
[StructLayout(LayoutKind.Sequential)]
public unsafe struct FerronApi
{
    public delegate* unmanaged<byte*, void> Log;
    public delegate* unmanaged<Entity> Spawn;
    public delegate* unmanaged<Entity, Transform*, byte> GetTransform;
    public delegate* unmanaged<Entity, Transform*, byte> SetTransform;
    public delegate* unmanaged<uint, byte> KeyDown;
    public delegate* unmanaged<uint, byte> KeyPressed;
    public delegate* unmanaged<uint, byte> KeyReleased;
    public delegate* unmanaged<uint, byte> MouseButtonDown;
    public delegate* unmanaged<float*, float*, void> CursorPos;
    public delegate* unmanaged<byte*, byte*, Transform*, Entity> SpawnRenderable;
    public delegate* unmanaged<Entity, byte> Despawn;
}

public static unsafe class Native
{
    private static FerronApi _api;

    internal static void Initialize(FerronApi* api) => _api = *api;

    public static void Log(string message)
    {
        var bytes = Encoding.UTF8.GetBytes(message);
        Span<byte> buffer = stackalloc byte[bytes.Length + 1];
        bytes.CopyTo(buffer);
        buffer[bytes.Length] = 0;
        fixed (byte* p = buffer)
            _api.Log(p);
    }

    public static Entity Spawn() => _api.Spawn();

    public static Transform GetTransform(Entity entity)
    {
        Transform transform = default;
        _api.GetTransform(entity, &transform);
        return transform;
    }

    public static void SetTransform(Entity entity, Transform value) =>
        _api.SetTransform(entity, &value);

    public static bool KeyDown(uint code) => _api.KeyDown(code) != 0;

    public static bool KeyPressed(uint code) => _api.KeyPressed(code) != 0;

    public static bool KeyReleased(uint code) => _api.KeyReleased(code) != 0;

    public static bool MouseButtonDown(uint button) => _api.MouseButtonDown(button) != 0;

    public static (float X, float Y) CursorPos()
    {
        float x = 0, y = 0;
        _api.CursorPos(&x, &y);
        return (x, y);
    }

    public static Entity SpawnRenderable(string mesh, string material, Transform transform)
    {
        var meshBytes = NulTerminated(mesh);
        var materialBytes = NulTerminated(material);
        fixed (byte* meshPtr = meshBytes)
        fixed (byte* materialPtr = materialBytes)
            return _api.SpawnRenderable(meshPtr, materialPtr, &transform);
    }

    public static bool Despawn(Entity entity) => _api.Despawn(entity) != 0;

    private static byte[] NulTerminated(string value)
    {
        var bytes = new byte[Encoding.UTF8.GetByteCount(value) + 1];
        Encoding.UTF8.GetBytes(value, 0, value.Length, bytes, 0);
        return bytes;
    }
}
