"""Plain text from a CLI result, as a reader sees it.

Typer prints usage errors in a Rich panel. Under GitHub Actions it forces
colour codes, and the panel wraps at the terminal width, so a phrase can be
split by escape codes, border characters and line breaks. Compare against
this instead of the raw output.
"""

import re

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BOX = re.compile(r"[│╭╮╰╯─]")


def plain(output: str) -> str:
    return " ".join(_BOX.sub(" ", _ANSI.sub("", output or "")).split())
