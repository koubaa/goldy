#!/usr/bin/env python3
"""Insert submission_context after device binding when tests use ctx without defining it."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

CTX_HELPER = "submission_context(&device)"
CTX_LINE = "    let ctx = submission_context(&device);\n"

DEVICE_PATTERNS = [
    r"let device = make_device\(\);",
    r"let device = make_device_for_stride_tests\(\);",
    r"let device = request_default_device\([^)]+\);",
    r"let device = request_device_preferring\([^)]+\);",
]


def uses_ctx(block: str) -> bool:
    return bool(re.search(r"\bctx\.|&ctx\b", block))


def has_ctx_binding(block: str) -> bool:
    return bool(re.search(r"\blet ctx\b|submission_context\(", block))


def fix_block(block: str) -> str:
    if not uses_ctx(block) or has_ctx_binding(block):
        return block
    for pat in DEVICE_PATTERNS:
        m = re.search(pat, block)
        if m:
            insert_at = m.end()
            # skip trailing comment on same line
            line_end = block.find("\n", insert_at)
            if line_end == -1:
                line_end = len(block)
            return block[:line_end] + "\n" + CTX_LINE + block[line_end + 1 :]
    return block


def fix_submission_diagnostics(text: str) -> str:
    return re.sub(
        r"\bdevice\.(deferred_deletion_pending_count|in_flight_command_buffer_count)\b",
        r"ctx.\1",
        text,
    )


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text
    text = fix_submission_diagnostics(text)

    parts = re.split(r"(?=#\[test\])", text)
    if len(parts) == 1:
        # whole-file pass for non-test modules
        if uses_ctx(text) and not has_ctx_binding(text):
            text = fix_block(text)
    else:
        out = [parts[0]]
        for part in parts[1:]:
            out.append(fix_block(part))
        text = "".join(out)

    # orphan ctx.flush without ctx in noop test
    text = re.sub(
        r"(fn flush_deferred_deletions_noop_on_idle_device\(\) \{\n    let device = make_device\(\);)\n    ctx\.",
        r"\1\n    let ctx = submission_context(&device);\n    ctx.",
        text,
    )

    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main():
    changed = []
    for path in TESTS.rglob("*.rs"):
        if fix_file(path):
            changed.append(path)
    for p in changed:
        print(p.relative_to(ROOT))
    print(f"fixed {len(changed)} files")


if __name__ == "__main__":
    main()
