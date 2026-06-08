class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("Goldy .NET bindings no longer expose CommandEncoder.");
        Console.WriteLine("Graphics rendering uses the Rust TaskGraph API (see goldy/examples/).");
        Console.WriteLine("Compute workloads remain available via ComputeEncoder.");
        Console.WriteLine();
        Console.WriteLine("FFI example: goldy/ffi-client/examples/compute_simple.rs");
    }
}
