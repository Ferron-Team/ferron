using System.Runtime.InteropServices;

namespace Ferron;

public static unsafe class Bootstrap
{
    [UnmanagedCallersOnly]
    public static int Init(FerronApi* api)
    {
        Native.Initialize(api);
        Native.Log("hello from C#");
        return 0;
    }

    [UnmanagedCallersOnly]
    public static void Free(nint handle)
    {
        if (handle != 0)
            GCHandle.FromIntPtr(handle).Free();
    }
}
