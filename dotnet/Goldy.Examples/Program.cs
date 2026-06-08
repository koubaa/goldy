class Program
{
    static void Main(string[] args)
    {
        var example = args.FirstOrDefault()?.ToLowerInvariant();

        switch (example)
        {
            case "triangle":
            case "triangle-window":
            case "":
            case null:
                Goldy.Examples.TriangleWindow.Run();
                break;
            case "triangle-headless":
            case "headless":
                TriangleHeadless.Run();
                break;
            default:
                PrintUsage();
                break;
        }
    }

    static void PrintUsage()
    {
        Console.WriteLine("Goldy.Examples - GPU demos via TaskGraph FFI");
        Console.WriteLine();
        Console.WriteLine("Usage: dotnet run --project Goldy.Examples -- [example]");
        Console.WriteLine();
        Console.WriteLine("Available examples:");
        Console.WriteLine("  triangle           - Windowed animated triangle (default)");
        Console.WriteLine("  triangle-headless  - Headless triangle + readback (CI)");
    }
}
