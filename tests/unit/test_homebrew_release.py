"""A source bump must never serve the previous release's Homebrew binaries."""

import runpy
from pathlib import Path

import pytest

update_formula = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/bump_homebrew_formula.py")
)["update_formula"]

SOURCE = """class Memorizz < Formula
  url "https://files.pythonhosted.org/packages/old/memorizz-0.10.0.tar.gz"
  sha256 "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

  bottle do
    root_url "https://github.com/RichmondAlake/homebrew-memorizz/releases/download/memorizz-0.10.0"
    sha256 cellar: :any, arm64_tahoe: "old-binary"
  end

  resource "packaging" do
    url "https://files.pythonhosted.org/packages/packaging.tar.gz"
    sha256 "dependency-checksum"
  end
end
"""
URL = "https://files.pythonhosted.org/packages/new/memorizz-0.11.0.tar.gz"
SHA = "b" * 64


def test_new_release_removes_stale_bottles_and_preserves_dependencies():
    updated = update_formula(SOURCE, "0.11.0", URL, SHA)
    assert f'  url "{URL}"' in updated
    assert f'  sha256 "{SHA}"' in updated
    assert "bottle do" not in updated
    assert "0.10.0" not in updated
    assert (
        updated.split('  resource "packaging"')[1]
        == SOURCE.split('  resource "packaging"')[1]
    )
    assert update_formula(updated, "0.11.0", URL, SHA) == updated


def test_same_release_preserves_current_bottles():
    current = (
        SOURCE.replace("0.10.0", "0.11.0")
        .replace(
            "https://files.pythonhosted.org/packages/old/memorizz-0.11.0.tar.gz", URL
        )
        .replace("a" * 64, SHA)
    )
    assert update_formula(current, "0.11.0", URL, SHA) == current


@pytest.mark.parametrize(
    "source,version,url,checksum",
    [
        (SOURCE, "0.11.0", URL, "invalid"),
        (SOURCE, "0.11.0", URL.replace("0.11.0", "0.10.0"), SHA),
        (SOURCE, "0.11.0", URL.replace("files.pythonhosted.org", "example.com"), SHA),
        (SOURCE.replace("  bottle do", "  bottle do # unexpected"), "0.11.0", URL, SHA),
        (SOURCE.replace('  sha256 "' + "a" * 64 + '"', ""), "0.11.0", URL, SHA),
    ],
)
def test_invalid_release_or_formula_fails_before_writing(
    source, version, url, checksum
):
    with pytest.raises(ValueError):
        update_formula(source, version, url, checksum)
