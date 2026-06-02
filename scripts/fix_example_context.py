#!/usr/bin/env python3
"""Fix goldy examples: Context must exist before Surface::new."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

NEEDS_STORED_CONTEXT = {
    "bouncing_lines",
    "starfield",
    "instancing",
    "game_of_life",
    "particles",
    "compute_particles",
}

CTX_LINE = "        let ctx = device.create_context()?;\n"


def strip_ctx_lines(text: str) -> str:
    return re.sub(
        r"\n\s*let ctx = device\.create_context\(\)(\?|\.expect\(\"context\"\));\n",
        "\n",
        text,
    )


def insert_ctx_after_device(text: str) -> str:
    patterns = [
        (
            r"(let device = Arc::new\(\s*\n(?:.*\n)*?\s*\)\?;)\n",
            r"\1\n" + CTX_LINE,
        ),
        (
            r"(let device = Arc::new\([^;]+\?;)\n",
            r"\1\n" + CTX_LINE,
        ),
    ]
    for pat, repl in patterns:
        if "let ctx = device.create_context" not in text[: text.find("Surface::new") if "Surface::new" in text else len(text)]:
            if re.search(pat, text):
                text = re.sub(pat, repl, text, count=1)
                break
    return text


def ensure_context_field(text: str, stem: str) -> str:
    if stem not in NEEDS_STORED_CONTEXT:
        return text
    if "context: goldy::Context" not in text:
        text = re.sub(
            r"(device: Arc<goldy::Device>,)\n",
            r"\1\n    context: goldy::Context,\n",
            text,
            count=1,
        )
    return text


def ensure_context_in_init(text: str, stem: str) -> str:
    if stem not in NEEDS_STORED_CONTEXT:
        return text
    if re.search(r"Ok\(Self \{[^}]*\bcontext:", text, re.S):
        return text
    return re.sub(
        r"(Ok\(Self \{\n\s+window,\n\s+device,)\n",
        r"\1\n            context: ctx,\n",
        text,
        count=1,
    )


def fix_dispatch(text: str) -> str:
    return text.replace(
        ".dispatch(&self.device)?", ".dispatch(&self.context)?"
    )


def fix_file(path: Path) -> bool:
    stem = path.stem
    text = path.read_text(encoding="utf-8")
    if "&ctx" not in text and ".dispatch(&self.device)" not in text:
        return False
    orig = text
    text = strip_ctx_lines(text)
    text = insert_ctx_after_device(text)
    text = ensure_context_field(text, stem)
    text = ensure_context_in_init(text, stem)
    text = fix_dispatch(text)
    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main():
    changed = []
    for path in sorted(EXAMPLES.glob("*.rs")):
        if fix_file(path):
            changed.append(path.name)
    for name in changed:
        print(name)
    print(f"fixed {len(changed)} files")


if __name__ == "__main__":
    main()
