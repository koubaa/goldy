namespace Goldy;

/// <summary>
/// Exception thrown when a Goldy operation fails.
/// </summary>
public class GoldyException : Exception
{
    public GoldyException() : base() { }
    public GoldyException(string message) : base(message) { }
    public GoldyException(string message, Exception innerException) : base(message, innerException) { }

    internal static GoldyException FromLastError(string operation)
    {
        var error = Native.NativeMethods.GetLastErrorString();
        return new GoldyException(error ?? $"{operation} failed");
    }
}

