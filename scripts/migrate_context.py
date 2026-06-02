#!/usr/bin/env python3
"""Migrate goldy call sites from Device submission API to Context."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP = {"refs", "target", "scripts", "placement_heap.rs"}

DEVICE_METHODS = [
    "submit_pipelined_and_retain",
    "submit_pipelined",
    "try_resubmit_retained",
    "flush_deferred_deletions",
    "poll_signals_and_service",
    "poll_signals",
    "has_deferred_payloads",
    "oldest_deferred_epoch",
    "high_water_timeline",
    "peek_oldest_in_flight",
    "gpu_progress",
    "wait_until_timeout",
    "wait_until",
    "defer_release",
    "defer_until",
    "dispatch",
    "submit",
]

SURFACE_NEW = re.compile(r"Surface::new(_with_(?:depth|config))?\(&device\b")
FRAME_ORCH = re.compile(r"FrameOrchestrator::new\(&device\b")
GRAPH_SUBMIT = re.compile(r"\.submit\(&device\b")
GRAPH_DISPATCH = re.compile(r"\.dispatch\(&device\b")
ENCODER_DISPATCH = re.compile(r"\.dispatch\(&device\b")
ENCODER_SUBMIT = re.compile(r"encoder\.submit\(&device\b|ComputeEncoder.*\.submit\(&device\b")

def should_process(path: Path) -> bool:
    if path.name == "placement_heap.rs":
        return False
    parts = path.parts
    if any(p in SKIP for p in parts):
        return False
    return path.suffix == ".rs"

def add_context_after_device(text: str) -> str:
    if "create_context()" in text:
        return text
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = re.search(
            r"let\s+(\w+)\s*=\s*(?:.*?(?:request_device|create_device|test_device|from_backend)[^\n;]*;)",
            line,
        )
        if m and m.group(1) == "device":
            indent = re.match(r"^(\s*)", line).group(1)
            ctx_line = f"{indent}let ctx = device.create_context().expect(\"context\");\n"
            # peek ahead: skip if next lines already define ctx
            rest = "".join(lines[i + 1 : i + 4])
            if "let ctx" not in rest and "create_context" not in rest:
                out.append(ctx_line)
        i += 1
    return "".join(out)

def migrate_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text

    for method in DEVICE_METHODS:
        text = re.sub(rf"\bdevice\.{method}\b", f"ctx.{method}", text)

    text = SURFACE_NEW.sub(lambda m: f"Surface::new{m.group(1) or ''}(&ctx", text)
    text = FRAME_ORCH.sub("FrameOrchestrator::new(&ctx", text)
    text = GRAPH_SUBMIT.sub(".submit(&ctx", text)
    text = GRAPH_DISPATCH.sub(".dispatch(&ctx", text)

    text = re.sub(r"GpuGuard::new\(&device\b", "GpuGuard::new(&ctx", text)

    if path.parent.name in ("examples", "tests") or "tests" in path.parts:
        text = add_context_after_device(text)

    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False

def main():
    changed = []
    for path in ROOT.rglob("*.rs"):
        if should_process(path):
            if migrate_file(path):
                changed.append(path)
    for p in changed:
        print(p.relative_to(ROOT))
    print(f"updated {len(changed)} files")

if __name__ == "__main__":
    main()
