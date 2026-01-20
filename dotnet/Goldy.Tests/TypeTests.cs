namespace Goldy.Tests;

public class TypeTests
{
    [Fact]
    public void Color_FromRgb_ConvertsCorrectly()
    {
        var color = Color.FromRgb(255, 128, 0);
        
        Assert.Equal(1.0f, color.R, 0.01f);
        Assert.Equal(0.502f, color.G, 0.01f);
        Assert.Equal(0.0f, color.B, 0.01f);
        Assert.Equal(1.0f, color.A, 0.01f);
    }

    [Fact]
    public void Color_FromRgba_ConvertsCorrectly()
    {
        var color = Color.FromRgba(255, 128, 64, 128);
        
        Assert.Equal(1.0f, color.R, 0.01f);
        Assert.Equal(0.502f, color.G, 0.01f);
        Assert.Equal(0.251f, color.B, 0.01f);
        Assert.Equal(0.502f, color.A, 0.01f);
    }

    [Fact]
    public void Color_ToRgba8_ConvertsCorrectly()
    {
        var color = new Color(1.0f, 0.5f, 0.0f, 1.0f);
        var bytes = color.ToRgba8();
        
        Assert.Equal(255, bytes[0]);
        Assert.Equal(127, bytes[1]);
        Assert.Equal(0, bytes[2]);
        Assert.Equal(255, bytes[3]);
    }

    [Fact]
    public void Color_Constants_AreCorrect()
    {
        Assert.Equal(new Color(0, 0, 0, 1), Color.Black);
        Assert.Equal(new Color(1, 1, 1, 1), Color.White);
        Assert.Equal(new Color(1, 0, 0, 1), Color.Red);
        Assert.Equal(new Color(0, 1, 0, 1), Color.Green);
        Assert.Equal(new Color(0, 0, 1, 1), Color.Blue);
    }

    [Fact]
    public void Color_Equality_Works()
    {
        var a = new Color(0.5f, 0.5f, 0.5f, 1.0f);
        var b = new Color(0.5f, 0.5f, 0.5f, 1.0f);
        var c = new Color(0.5f, 0.5f, 0.5f, 0.5f);
        
        Assert.Equal(a, b);
        Assert.NotEqual(a, c);
        Assert.True(a == b);
        Assert.True(a != c);
    }

    [Fact]
    public void DataAccess_Enum_HasCorrectValues()
    {
        // Verify enum values match C FFI
        Assert.Equal(0, (int)DataAccess.Scattered);
        Assert.Equal(1, (int)DataAccess.Broadcast);
    }

    [Fact]
    public void SpatialAccess_Enum_HasCorrectValues()
    {
        // Verify enum values match C FFI
        Assert.Equal(0, (int)SpatialAccess.Interpolated);
        Assert.Equal(1, (int)SpatialAccess.Direct);
    }

    [Fact]
    public void TextureFlags_Work()
    {
        var flags = TextureFlags.CopySrc | TextureFlags.CopyDst;
        
        Assert.True(flags.HasFlag(TextureFlags.CopySrc));
        Assert.True(flags.HasFlag(TextureFlags.CopyDst));
        Assert.False(flags.HasFlag(TextureFlags.RenderTarget));
    }

    [Fact]
    public void ShaderStages_Flags_Work()
    {
        var stages = ShaderStages.Vertex | ShaderStages.Fragment;
        
        Assert.True(stages.HasFlag(ShaderStages.Vertex));
        Assert.True(stages.HasFlag(ShaderStages.Fragment));
        Assert.False(stages.HasFlag(ShaderStages.Compute));
        Assert.Equal(ShaderStages.All, ShaderStages.Vertex | ShaderStages.Fragment | ShaderStages.Compute);
    }

    [Fact]
    public void SamplerDesc_Default_HasCorrectValues()
    {
        var desc = SamplerDesc.Default;
        
        Assert.Equal(AddressMode.ClampToEdge, desc.AddressModeU);
        Assert.Equal(AddressMode.ClampToEdge, desc.AddressModeV);
        Assert.Equal(FilterMode.Nearest, desc.MagFilter);
        Assert.Equal(FilterMode.Nearest, desc.MinFilter);
        Assert.Equal(1.0f, desc.MaxAnisotropy);
    }

    [Fact]
    public void SamplerDesc_Linear_HasCorrectValues()
    {
        var desc = SamplerDesc.Linear;
        
        Assert.Equal(FilterMode.Linear, desc.MagFilter);
        Assert.Equal(FilterMode.Linear, desc.MinFilter);
        Assert.Equal(FilterMode.Linear, desc.MipmapFilter);
    }

    [Fact]
    public void DepthStencilState_DefaultValues_AreCorrect()
    {
        var state = new DepthStencilState(DepthFormat.Depth24Plus);
        
        Assert.Equal(DepthFormat.Depth24Plus, state.Format);
        Assert.True(state.DepthWriteEnabled);
        Assert.Equal(CompareFunction.Less, state.DepthCompare);
    }
}

