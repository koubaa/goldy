using Goldy.Native;

namespace Goldy;

/// <summary>
/// Destination of a memory-exchange deposit (buffer range or texture region).
/// </summary>
public readonly struct DepositTarget
{
    internal NativeDepositTarget Raw { get; }

    private DepositTarget(NativeDepositTarget raw) => Raw = raw;

    public static DepositTarget Buffer(Parcel destination, ulong capacity, ulong dstOffset = 0)
    {
        ArgumentNullException.ThrowIfNull(destination);
        return new DepositTarget(new NativeDepositTarget
        {
            Kind = 0,
            Buffer = destination.Handle,
            DstOffset = dstOffset,
            Capacity = capacity,
        });
    }

    public static DepositTarget Texture(
        Texture destination,
        uint x,
        uint y,
        uint width,
        uint height,
        ulong capacity,
        uint srcRowPitch = 0)
    {
        ArgumentNullException.ThrowIfNull(destination);
        return new DepositTarget(new NativeDepositTarget
        {
            Kind = 1,
            Texture = destination.Handle,
            Capacity = capacity,
            X = x,
            Y = y,
            Width = width,
            Height = height,
            SrcRowPitch = srcRowPitch,
        });
    }
}
