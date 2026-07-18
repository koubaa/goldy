using System.Runtime.InteropServices;
using Goldy;
using Silk.NET.GLFW;

namespace Goldy.Examples;

/// <summary>
/// Create a Goldy <see cref="SurfaceExchange"/> from a GLFW window (cross-platform).
/// </summary>
internal static class GlfwSurfaceExchange
{
    public static unsafe SurfaceExchange Create(Context ctx, WindowHandle* window)
    {
        var handle = (nint)window;

        if (OperatingSystem.IsWindows())
        {
            var hwnd = GlfwNative.GetWin32Window(handle);
            if (hwnd == nint.Zero)
                throw new InvalidOperationException("glfwGetWin32Window returned null");
            return SurfaceExchange.CreateWin32(ctx, hwnd);
        }

        if (OperatingSystem.IsMacOS())
        {
            var nsWindow = GlfwNative.GetCocoaWindow(handle);
            if (nsWindow == nint.Zero)
                throw new InvalidOperationException("glfwGetCocoaWindow returned null");
            var nsView = ObjC.ContentView(nsWindow);
            if (nsView == nint.Zero)
                throw new InvalidOperationException("NSWindow contentView is null");
            return SurfaceExchange.CreateAppKit(ctx, nsView);
        }

        if (OperatingSystem.IsLinux())
        {
            var display = GlfwNative.GetWaylandDisplay();
            var surface = GlfwNative.GetWaylandWindow(handle);
            if (display == nint.Zero || surface == nint.Zero)
            {
                throw new InvalidOperationException(
                    "Wayland handles unavailable — run under a Wayland session (Vulkan backend requires Wayland on Linux)");
            }
            return SurfaceExchange.CreateWayland(ctx, display, surface);
        }

        throw new PlatformNotSupportedException("Unsupported OS for GLFW surface exchange creation");
    }

    private static class ObjC
    {
        [DllImport("libobjc.A.dylib")]
        private static extern nint objc_msgSend(nint receiver, nint selector);

        [DllImport("libobjc.A.dylib", EntryPoint = "sel_registerName")]
        private static extern nint SelRegisterName(string name);

        public static nint ContentView(nint nsWindow)
        {
            var sel = SelRegisterName("contentView");
            return objc_msgSend(nsWindow, sel);
        }
    }
}
