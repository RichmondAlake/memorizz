# Homebrew formula for the memorizz CLI.
#
# Ships via a PERSONAL TAP (github.com/RichmondAlake/homebrew-memorizz), not
# homebrew-core: the PolyForm-Noncommercial license is not OSI-approved, so core
# would reject it. Uses uv to avoid hand-pinning dozens of Python resources.
#
# Publish: copy this file to Formula/memorizz.rb in the tap repo, fill in the
# sdist sha256 for the release, and `brew install RichmondAlake/memorizz/memorizz`.
class Memorizz < Formula
  desc "Memory management library and local agent CLI for AI agents"
  homepage "https://github.com/RichmondAlake/memorizz"
  url "https://files.pythonhosted.org/packages/source/m/memorizz/memorizz-0.1.0.tar.gz"
  sha256 "REPLACE_WITH_SDIST_SHA256"
  license :cannot_represent # PolyForm-Noncommercial-1.0.0 (not an SPDX/OSI id)
  version "0.1.0"

  depends_on "python@3.12"
  depends_on "uv"

  def install
    # Install into a uv-managed tool env rooted under the formula prefix so brew
    # owns and cleans it up.
    ENV["UV_TOOL_DIR"] = libexec/"tools"
    ENV["UV_TOOL_BIN_DIR"] = libexec/"bin"
    system "uv", "tool", "install",
           "--python", Formula["python@3.12"].opt_bin/"python3.12",
           "memorizz==#{version}"
    bin.install_symlink libexec/"bin/memorizz"
  end

  test do
    assert_match "memorizz", shell_output("#{bin}/memorizz --version")
  end
end
