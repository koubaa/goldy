#!/usr/bin/env bash
# Build the mdBook and reject silent breakage.
#
# mdBook exits 0 when an `{{#include}}` target is missing — it only logs the
# failure — so renaming an example would quietly gut the Examples chapter. This
# also catches internal links that point at pages the book does not build.
#
# Usage: scripts/check_docs_book.sh   (requires mdbook and python3 on PATH)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

build_log="$(mktemp)"
trap 'rm -f "$build_log"' EXIT

rm -rf docs/book
mdbook build docs 2>&1 | tee "$build_log"

if grep -q '\[ERROR\]' "$build_log"; then
    echo "mdbook reported errors (see above)" >&2
    exit 1
fi

python3 - <<'PY'
import pathlib
import re
import sys

root = pathlib.Path("docs/book")
failures = []

for page in root.rglob("*.html"):
    text = page.read_text(encoding="utf-8", errors="replace")
    if "Unable to include file" in text or "{{#include" in text:
        failures.append(f"{page}: unresolved include directive")
    for match in re.finditer(r'href="([^"#?:]+\.html)([#?][^"]*)?"', text):
        target = (page.parent / match.group(1)).resolve()
        if not target.exists():
            failures.append(f"{page}: broken link to {match.group(1)}")
    # print.html concatenates every chapter at the book root, so chapter-relative
    # media paths never resolve there.
    media = [] if page.name == "print.html" else re.finditer(r'<(?:video|img)[^>]*\ssrc="([^"#?:]+)"', text)
    for match in media:
        target = (page.parent / match.group(1)).resolve()
        if not target.exists():
            failures.append(f"{page}: missing media {match.group(1)}")

if failures:
    print("\n".join(sorted(set(failures))), file=sys.stderr)
    print(f"\n{len(set(failures))} book problem(s)", file=sys.stderr)
    sys.exit(1)

print("book OK")
PY
