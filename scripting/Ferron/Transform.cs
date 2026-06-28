using System.Numerics;
using System.Runtime.InteropServices;

namespace Ferron;

// Layout must match Rust CTransform.
[StructLayout(LayoutKind.Sequential)]
public struct Transform
{
    public Vector3 Position;
    public Quaternion Rotation;
    public Vector3 Scale;

    public static Transform Identity => new()
    {
        Position = Vector3.Zero,
        Rotation = Quaternion.Identity,
        Scale = Vector3.One,
    };
}
