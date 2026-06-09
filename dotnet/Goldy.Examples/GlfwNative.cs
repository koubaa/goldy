using System.Runtime.InteropServices;

namespace Goldy.Examples;

/// <summary>
/// GLFW native window handle accessors (not wrapped by Silk.NET.GLFW).
/// Loads the same native library Silk.NET ships (e.g. libglfw.3.dylib on macOS).
/// </summary>
internal static class GlfwNative
{
    private static readonly nint Lib = LoadGlfw();

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate nint GetWindowDelegate(nint window);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate nint GetWaylandDisplayDelegate();

    internal static nint GetWin32Window(nint window) =>
        GetExport<GetWindowDelegate>("glfwGetWin32Window")(window);

    internal static nint GetCocoaWindow(nint window) =>
        GetExport<GetWindowDelegate>("glfwGetCocoaWindow")(window);

    internal static nint GetWaylandDisplay() =>
        GetExport<GetWaylandDisplayDelegate>("glfwGetWaylandDisplay")();

    internal static nint GetWaylandWindow(nint window) =>
        GetExport<GetWindowDelegate>("glfwGetWaylandWindow")(window);

    private static T GetExport<T>(string name) where T : Delegate
    {
        var ptr = NativeLibrary.GetExport(Lib, name);
        return Marshal.GetDelegateForFunctionPointer<T>(ptr);
    }

    private static nint LoadGlfw()
    {
        // Silk.NET.GLFW ships platform-specific names under runtimes/{rid}/native/.
        ReadOnlySpan<string> names = OperatingSystem.IsWindows()
            ? ["glfw3"]
            : OperatingSystem.IsMacOS()
                ? ["glfw.3", "glfw3"]
                : ["glfw.so.3", "glfw3"];

        foreach (var name in names)
        {
            if (NativeLibrary.TryLoad(name, typeof(GlfwNative).Assembly, null, out var handle))
                return handle;
        }

        throw new DllNotFoundException(
            "Could not load GLFW native library (expected glfw3 on Windows, libglfw.3 on macOS, libglfw.so.3 on Linux).");
    }
}
