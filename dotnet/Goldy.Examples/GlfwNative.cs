using System.Runtime.InteropServices;

namespace Goldy.Examples;

/// <summary>
/// GLFW native window handle accessors (not wrapped by Silk.NET.GLFW).
/// </summary>
internal static partial class GlfwNative
{
    private const string LibName = "glfw3";

    [LibraryImport(LibName, EntryPoint = "glfwGetWin32Window")]
    internal static partial nint GetWin32Window(nint window);

    [LibraryImport(LibName, EntryPoint = "glfwGetCocoaWindow")]
    internal static partial nint GetCocoaWindow(nint window);

    [LibraryImport(LibName, EntryPoint = "glfwGetWaylandDisplay")]
    internal static partial nint GetWaylandDisplay();

    [LibraryImport(LibName, EntryPoint = "glfwGetWaylandWindow")]
    internal static partial nint GetWaylandWindow(nint window);
}
