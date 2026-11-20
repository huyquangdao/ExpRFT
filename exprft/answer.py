from __future__ import annotations

import re

ANSWER_PATTERN = re.compile(
    r"the\s+answer\s+is\s*:?\s*[\*_`\"'(\[]*\s*([A-Z])[\*_`\"')\]]*(?=$|[^A-Za-z0-9_])", re.I
)


def parse_answer(text: str, valid_letters=("A", "B", "C", "D")) -> str | None:
    if not text:
        return None
    valid = {v.upper() for v in valid_letters}
    for m in ANSWER_PATTERN.finditer(text):
        if m.group(1) in valid:
            return m.group(1)
    return None
