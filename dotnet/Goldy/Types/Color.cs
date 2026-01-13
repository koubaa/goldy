using System.Runtime.InteropServices;

namespace Goldy;

/// <summary>
/// RGBA color with floating point components (0.0 - 1.0).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct Color : IEquatable<Color>
{
    public readonly float R;
    public readonly float G;
    public readonly float B;
    public readonly float A;

    public Color(float r, float g, float b, float a = 1.0f)
    {
        R = r;
        G = g;
        B = b;
        A = a;
    }

    /// <summary>
    /// Create a color from RGB byte values (0-255).
    /// </summary>
    public static Color FromRgb(byte r, byte g, byte b) =>
        new(r / 255.0f, g / 255.0f, b / 255.0f, 1.0f);

    /// <summary>
    /// Create a color from RGBA byte values (0-255).
    /// </summary>
    public static Color FromRgba(byte r, byte g, byte b, byte a) =>
        new(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);

    /// <summary>
    /// Convert to RGBA byte array.
    /// </summary>
    public byte[] ToRgba8() =>
    [
        (byte)(R * 255.0f),
        (byte)(G * 255.0f),
        (byte)(B * 255.0f),
        (byte)(A * 255.0f),
    ];

    // Predefined colors
    public static Color Black => new(0.0f, 0.0f, 0.0f, 1.0f);
    public static Color White => new(1.0f, 1.0f, 1.0f, 1.0f);
    public static Color Red => new(1.0f, 0.0f, 0.0f, 1.0f);
    public static Color Green => new(0.0f, 1.0f, 0.0f, 1.0f);
    public static Color Blue => new(0.0f, 0.0f, 1.0f, 1.0f);
    public static Color CornflowerBlue => new(0.392f, 0.584f, 0.929f, 1.0f);
    public static Color Transparent => new(0.0f, 0.0f, 0.0f, 0.0f);

    public bool Equals(Color other) =>
        R == other.R && G == other.G && B == other.B && A == other.A;

    public override bool Equals(object? obj) =>
        obj is Color other && Equals(other);

    public override int GetHashCode() =>
        HashCode.Combine(R, G, B, A);

    public static bool operator ==(Color left, Color right) => left.Equals(right);
    public static bool operator !=(Color left, Color right) => !left.Equals(right);

    public override string ToString() =>
        $"Color({R:F3}, {G:F3}, {B:F3}, {A:F3})";
}

