namespace Goldy.Examples;

static class ShaderPaths
{
    public static string Load(string fileName)
    {
        var dir = FindShadersDirectory()
            ?? throw new FileNotFoundException(
                $"Could not find goldy/shaders (looking for {fileName}). Run from the goldy repo.");
        return File.ReadAllText(Path.Combine(dir, fileName));
    }

    static string? FindShadersDirectory()
    {
        DirectoryInfo? dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            var shaders = Path.Combine(dir.FullName, "shaders");
            if (Directory.Exists(shaders))
                return shaders;
            dir = dir.Parent;
        }

        dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir != null)
        {
            var shaders = Path.Combine(dir.FullName, "shaders");
            if (Directory.Exists(shaders))
                return shaders;
            dir = dir.Parent;
        }

        return null;
    }
}
