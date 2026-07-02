namespace Ferron;

/// <summary>Frame timing, read live from the engine's Time resource.</summary>
public static class Time
{
    /// <summary>Seconds since the last frame.</summary>
    public static float DeltaTime
    {
        // TODO: Native.TimeDelta()
        get => 0f;
    }

    /// <summary>Total elapsed seconds since engine start.</summary>
    public static float Total
    {
        // TODO: Native.TimeTotal()
        get => 0f;
    }

    /// <summary>Frames ticked since engine start; handy for debugging frame-dependent bugs.</summary>
    public static ulong FrameCount
    {
        // TODO: Native.TimeFrameCount()
        get => 0;
    }
}
