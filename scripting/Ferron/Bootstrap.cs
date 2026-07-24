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
}
