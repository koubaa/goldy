"""Function-scoped bind_withdraw → host-claim migrator.

Does not strip WithdrawTransaction from type names.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BIND = re.compile(
    r"""let\s+(?:mut\s+)?(?P<name>\w+)\s*=\s*
        (?:[^;]*?)
        \.bind_withdraw(?:_texture)?\(\s*&mut\s+\w+\s*,\s*(?P<parcel>[^)]+?)\s*\)\s*
        (?:\.(?:expect|unwrap)\([^;]*\)|\?)?\s*;""",
    re.VERBOSE | re.DOTALL,
)

CLAIM = re.compile(
    r"""(?P<name>\w+)\s*\.claim\(\s*(?P<sub>(?:&mut\s+)?[^)]+)\s*\)\s*
        (?:\?|\.expect\([^)]*\)|\.unwrap\(\))\s*
        \.consume\(\)\s*
        (?:\?|\.expect\([^)]*\)|\.unwrap\(\))""",
    re.VERBOSE,
)

READ_GRANT = re.compile(
    r"read_grant_u32\(\s*&?(?P<name>\w+)\s*,\s*(?P<sub>[^,]+),\s*(?P<count>[^)]+)\)"
)


def parcel_expr(raw: str) -> str:
    p = " ".join(raw.strip().split())
    if p.startswith("&"):
        return p
    if p.endswith(".whole()"):
        return p if p.startswith("&") else p
    return f"&*{p}" if p.isidentifier() else (p if p.startswith("&") else f"&{p}" if p[0].isidentifier() else p)


def take_expr(sub: str, parcel: str, q: bool) -> str:
    sub = sub.strip()
    inner = sub[len("&mut ") :] if sub.startswith("&mut ") else sub
    recv = f"&mut {inner}"
    if q:
        return f"({recv} >> {parcel}).take::<u8>()?"
    return f"({recv} >> {parcel}).take::<u8>().expect(\"host take\")"


def transform_fn(text: str) -> str:
    binds: dict[str, str] = {}
    for m in BIND.finditer(text):
        binds[m.group("name")] = parcel_expr(m.group("parcel"))
    if binds:
        text = BIND.sub("", text)

    def claim_sub(m: re.Match[str]) -> str:
        name = m.group("name")
        if name not in binds:
            return m.group(0)
        q = "?" in m.group(0) and ".expect" not in m.group(0) and ".unwrap" not in m.group(0)
        return take_expr(m.group("sub"), binds[name], q)

    text = CLAIM.sub(claim_sub, text)

    def rg_sub(m: re.Match[str]) -> str:
        name = m.group("name")
        if name not in binds:
            return m.group(0)
        return f"read_grant_u32({m.group('sub').strip()}, {binds[name]}, {m.group('count').strip()})"

    text = READ_GRANT.sub(rg_sub, text)
    return text


FN = re.compile(
    r"(?P<pre>(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+\w+[^{]*\{)",
)


def split_functions(src: str) -> list[str]:
    """Split a Rust file into chunks that start at `fn` (best-effort)."""
    idxs = [m.start() for m in re.finditer(r"(?m)^(?:    )?(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+", src)]
    if not idxs:
        return [src]
    chunks = [src[: idxs[0]]]
    for i, start in enumerate(idxs):
        end = idxs[i + 1] if i + 1 < len(idxs) else len(src)
        chunks.append(src[start:end])
    return chunks


def ensure_shr(src: str) -> str:
    if " >> " not in src or "use std::ops::Shr" in src or "ops::Shr" in src:
        return src
    if "use std::ops::{" in src and "Shr" not in src:
        return src.replace("use std::ops::{", "use std::ops::{Shr, ", 1)
    # insert after first use block
    m = re.search(r"(?m)^use [^\n]+;\n", src)
    if m:
        return src[: m.end()] + "use std::ops::Shr;\n" + src[m.end() :]
    return "use std::ops::Shr;\n" + src


def fix_undefined_parcels(fn: str) -> str:
    """If take/read_grant uses &output but output is not defined, pick a local parcel."""
    lets = set(re.findall(r"\blet\s+(?:mut\s+)?(\w+)", fn))
    params = set(re.findall(r"\b(\w+)\s*:\s*&(?:mut\s+)?(?:goldy::)?(?:Parcel|Buffer|Texture)\b", fn))
    defined = lets | params | {"self"}
    writes = re.findall(r"with_parcel\(&(\w+),\s*NodeAccess::(?:Write|ReadWrite|Overwrite)", fn)
    writes += re.findall(r"with_parcel\((\w+),\s*NodeAccess::(?:Write|ReadWrite|Overwrite)", fn)

    def replace_name(old: str, new: str, text: str) -> str:
        text = text.replace(f">> &{old}", f">> &{new}")
        text = text.replace(f">> &*{old}", f">> &*{new}")
        text = re.sub(rf"read_grant_u32\(([^,]+), &{old},", rf"read_grant_u32(\1, &{new},", text)
        return text

    for m in re.finditer(r">> &(?:\*)?(\w+)", fn):
        name = m.group(1)
        if name in defined or name in {"self"}:
            continue
        if "output" == name and "out" in defined:
            fn = replace_name("output", "out", fn)
        elif name == "y" and "data" in defined and "y" not in defined:
            fn = replace_name("y", "data", fn)
        elif name == "output" and "texture" in defined:
            fn = replace_name("output", "texture", fn)
        elif name == "output" and "shared" in defined:
            fn = replace_name("output", "shared", fn)
        elif name == "output" and "parcel" in defined:
            fn = replace_name("output", "parcel", fn)
        elif name == "output" and writes:
            fn = replace_name("output", writes[-1], fn)
        elif writes:
            fn = replace_name(name, writes[-1], fn)
    for m in re.finditer(r"read_grant_u32\([^,]+, &(\w+),", fn):
        name = m.group(1)
        if name in defined:
            continue
        if name == "output" and "out" in defined:
            fn = replace_name("output", "out", fn)
        elif name == "output" and writes:
            fn = replace_name("output", writes[-1], fn)
        elif writes:
            fn = replace_name(name, writes[-1], fn)
    return fn


def transform_file(src: str) -> str:
    chunks = split_functions(src)
    out = "".join(fix_undefined_parcels(transform_fn(c)) for c in chunks)
    if " >> " in out:
        out = ensure_shr(out)
    return out


def main() -> None:
    globs = (
        "src/**/*.rs",
        "tests/**/*.rs",
        "examples/**/*.rs",
    )
    changed = 0
    for glob in globs:
        for path in ROOT.glob(glob):
            if "ffi" in path.parts or "python" in path.parts or "dotnet" in path.parts:
                continue
            original = path.read_text(encoding="utf-8")
            updated = transform_file(original)
            if updated != original:
                path.write_text(updated, encoding="utf-8")
                changed += 1
                print(f"updated {path.relative_to(ROOT)}")
    print(f"changed {changed} files")


if __name__ == "__main__":
    main()
