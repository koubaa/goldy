class Program
{
    static void Main(string[] args)
    {
        var example = args.FirstOrDefault()?.ToLowerInvariant();
        
        switch (example)
        {
            case "triangle":
                Triangle.Run();
                break;
            case "plasma":
                Plasma.Run();
                break;
            case "gameoflife":
            case "gol":
                GameOfLife.Run();
                break;
            default:
                PrintUsage();
                break;
        }
    }

    static void PrintUsage()
    {
        Console.WriteLine("Goldy.Examples - GPU graphics demos");
        Console.WriteLine();
        Console.WriteLine("Usage: dotnet run -- <example>");
        Console.WriteLine();
        Console.WriteLine("Available examples:");
        Console.WriteLine("  triangle         - Animated triangle in a window (Surface API)");
        Console.WriteLine("  gameoflife, gol  - Conway's Game of Life (compute + graphics)");
        Console.WriteLine("  plasma           - Classic demoscene plasma effect (bindless)");
    }
}
