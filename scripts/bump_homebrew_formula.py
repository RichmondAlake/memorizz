"""Update the tap source without retaining binaries from an older release."""

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse


def update_formula(source: str, version: str, url: str, sha256: str) -> str:
    """Validate metadata before editing; same-version retries preserve bottles."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Expected a stable X.Y.Z release version")
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "files.pythonhosted.org"
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith(f"/memorizz-{version}.tar.gz")
        or any(character.isspace() or character in '\\"' for character in url)
    ):
        raise ValueError("Expected the matching PyPI source archive URL")
    if not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("Expected a SHA-256 checksum")
    urls = list(re.finditer(r'^  url "([^"\n]+)"$', source, re.MULTILINE))
    hashes = list(re.finditer(r'^  sha256 "[a-f0-9]{64}"$', source, re.MULTILINE))
    if len(urls) != 1 or len(hashes) != 1:
        raise ValueError("Expected exactly one formula source URL and checksum")
    previous = re.search(r"/memorizz-(\d+\.\d+\.\d+)\.tar\.gz$", urls[0][1])
    if previous is None:
        raise ValueError("Cannot identify the previous formula version")
    result = source
    for match, replacement in reversed(
        sorted(
            [(urls[0], f'  url "{url}"'), (hashes[0], f'  sha256 "{sha256}"')],
            key=lambda edit: edit[0].start(),
        )
    ):
        result = result[: match.start()] + replacement + result[match.end() :]
    if previous[1] != version:
        result = re.sub(
            r"^  bottle do\n.*?^  end\n(?:\n)?",
            "",
            result,
            flags=re.MULTILINE | re.DOTALL,
        )
        if re.search(r"^  bottle\b", result, re.MULTILINE):
            raise ValueError("Cannot safely remove the previous bottle block")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("formula", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    original = args.formula.read_text()
    updated = update_formula(original, args.version, args.url, args.sha256)
    if updated != original:
        args.formula.write_text(updated)


if __name__ == "__main__":
    main()
