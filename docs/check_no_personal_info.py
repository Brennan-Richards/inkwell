"""Fail the build if the published knowledge base regains personal information.

The documents under docs/knowledge-base/ are Page One's own operating documents,
published with permission and redacted so that individuals are referred to by
role rather than by name, handle, or contact details.

This check is deliberately shape-based. It looks for the *forms* personal data
takes -- phone numbers, email addresses, and Discord handles that are not known
role or brand accounts -- so that it never has to contain the very information
it exists to keep out of the repository.

Usage:
    python docs/check_no_personal_info.py docs/knowledge-base/*.md
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Handles that legitimately appear: role mentions, brand accounts, Discord role
# tags, and the placeholder names the documents already use in examples.
ALLOWED_HANDLES = {
    "@abc",
    "@communityhelper",
    "@fantasy-writers",
    "@lead-editor",
    "@lead-moderator",
    "@moderator",
    "@pageonehq",
    "@pageoneleadmoderator",
    "@pageonemember1",
    "@pageonemember2",
    "@pageonemember3",
    "@pageonemember4",
    "@polls",
    "@romance-writers",
    "@server-owner",
    "@serverowner",
    "@thriller-writers",
    "@writingprompts",
    "@xyz",
}

PATTERNS = {
    "phone number": re.compile(r"(?:\+?1[ .\-]?)?\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}"),
    "email address": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "personal social link": re.compile(
        r"https?://(?:www\.)?(?:instagram|twitter|x|facebook|tiktok|linkedin)\.com/[A-Za-z0-9_.\-]+",
        re.IGNORECASE,
    ),
}

HANDLE = re.compile(r"@[A-Za-z0-9._\-]{2,32}")


def check(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []

    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            problems.append(f"{path}:{line}: {label}: {match.group(0)!r}")

    for match in HANDLE.finditer(text):
        handle = match.group(0)
        if handle.lower() not in ALLOWED_HANDLES:
            line = text.count("\n", 0, match.start()) + 1
            problems.append(
                f"{path}:{line}: unrecognised handle: {handle!r} "
                f"(add it to ALLOWED_HANDLES if it is a role or brand account)"
            )

    return problems


def main(argv: list[str]) -> int:
    paths = [Path(arg) for arg in argv]
    if not paths:
        print("usage: check_no_personal_info.py <file>...", file=sys.stderr)
        return 2

    problems: list[str] = []
    for path in paths:
        problems.extend(check(path))

    if problems:
        print("Personal information found in published documents:\n")
        for problem in problems:
            print(f"  {problem}")
        print(f"\n{len(problems)} problem(s) found.")
        return 1

    print(f"OK: no personal information found in {len(paths)} document(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
