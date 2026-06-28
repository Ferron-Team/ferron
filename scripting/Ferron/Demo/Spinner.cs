using System.Numerics;

namespace Ferron.Demo;

public class Spinner : Behaviour
{
    public float DegreesPerSecond = 90f;

    public override void OnStart() => Native.Log($"Spinner attached to {Entity}");

    public override void OnUpdate(float deltaTime)
    {
        var t = Transform;
        var radians = MathF.PI / 180f * DegreesPerSecond * deltaTime;
        t.Rotation = Quaternion.Normalize(
            Quaternion.Concatenate(t.Rotation, Quaternion.CreateFromAxisAngle(Vector3.UnitY, radians)));
        Transform = t;
    }
}
