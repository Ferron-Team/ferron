using System.Runtime.InteropServices;

namespace Ferron;

public static unsafe class Bootstrap
{
    const int Ok = 0;
    const int AbiMismatch = 1;

    [UnmanagedCallersOnly]
    public static int Init(FerronApi* api, int apiSize)
    {
        // ABI handshake. The engine passes the byte size of *its* FerronApi; if
        // it differs from ours, the two were built against different definitions
        // of the struct. `Native.Initialize` would copy `sizeof(our FerronApi)`
        // bytes out of the engine's table and later call through those offsets —
        // reading past a smaller table or landing on the wrong function pointer,
        // i.e. undefined behaviour on the first script call. Refuse first.
        //
        // sizeof grows automatically as fields are appended, so this needs no
        // manual version bump. Reported via stderr, not `Native.Log`, because the
        // log callback lives in the very table that may be malformed.
        int expected = sizeof(FerronApi);
        if (apiSize != expected)
        {
            Console.Error.WriteLine(
                $"[Ferron] ABI mismatch: engine FerronApi is {apiSize} bytes, this "
                + $"assembly expects {expected}. Rebuild the Ferron assembly against "
                + "the current engine (dotnet build scripting/Ferron).");
            return AbiMismatch;
        }

        Native.Initialize(api);
        Native.Log("hello from C#");
        return Ok;
    }
}
