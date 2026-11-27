# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""A self-aware agent's own copy of the MemoRizz source.

The copy lives under ``<MEMORIZZ_HOME>/self_aware/<agent>/source`` and is a
git repository: the first commit is MemoRizz as installed, and every change
the agent keeps is one more commit, a *version*. Nothing here touches the
installed package, so the running process is never edited from under itself;
a version is checked in a separate interpreter that imports the copy.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PACKAGE = "memorizz"
BASE_TAG = "installed"
_IGNORED = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".DS_Store", ".git", "node_modules"
)
_AUTHOR = ["-c", "user.name=MemoRizz agent", "-c", "user.email=agent@memorizz.local"]
# Nothing of the person's own git setup runs in the agent's copy.
_SAFE = ["-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"]
MAX_READ_CHARS = 60_000
MAX_DIFF_CHARS = 200_000
MAX_VIEW_CHARS = 2_000_000


class SelfSourceError(RuntimeError):
    """The agent's copy could not be made, read or changed."""


def _safe_id(agent_id: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(agent_id or "")).strip("-.")
    if not text:
        raise SelfSourceError("An agent needs an ID before it can have its own source")
    return text[:120]


def installed_package_dir() -> Path:
    """Where the running MemoRizz package is on disk."""
    import memorizz

    return Path(memorizz.__file__).resolve().parent


def installed_version() -> str:
    import memorizz

    return str(getattr(memorizz, "__version__", "unknown"))


def parse_diff(text: str) -> List[Dict[str, Any]]:
    """A git diff as files, hunks and numbered lines, for a page to draw."""
    files: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    hunk: Optional[Dict[str, Any]] = None
    old_no = new_no = 0
    for line in str(text or "").split("\n"):
        if line.startswith("diff --git "):
            found = re.match(r"diff --git a/(.*) b/(.*)$", line)
            path = found.group(2) if found else line[11:]
            current = {
                "path": path,
                "old_path": found.group(1) if found else path,
                "status": "changed",
                "insertions": 0,
                "deletions": 0,
                "binary": False,
                "hunks": [],
            }
            files.append(current)
            hunk = None
            continue
        if current is None:
            continue
        if hunk is None or not line[:1] in {" ", "+", "-", "\\"}:
            if line.startswith("new file mode"):
                current["status"] = "added"
            elif line.startswith("deleted file mode"):
                current["status"] = "deleted"
            elif line.startswith("rename from "):
                current["status"] = "renamed"
                current["old_path"] = line[12:]
            elif line.startswith("Binary files "):
                current["binary"] = True
            found = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@ ?(.*)$", line)
            if found:
                old_no, new_no = int(found.group(1)), int(found.group(2))
                hunk = {"header": line, "context": found.group(3), "lines": []}
                current["hunks"].append(hunk)
            continue
        mark, body = line[:1], line[1:]
        if mark == "+":
            hunk["lines"].append(
                {"kind": "add", "old": None, "new": new_no, "text": body}
            )
            new_no += 1
            current["insertions"] += 1
        elif mark == "-":
            hunk["lines"].append(
                {"kind": "del", "old": old_no, "new": None, "text": body}
            )
            old_no += 1
            current["deletions"] += 1
        elif mark == " ":
            hunk["lines"].append(
                {"kind": "same", "old": old_no, "new": new_no, "text": body}
            )
            old_no += 1
            new_no += 1
        # A "\ No newline at end of file" note is not a line of the file.
    return files


class SelfSource:
    """One agent's editable, versioned copy of the MemoRizz package."""

    def __init__(
        self,
        agent_id: str,
        *,
        home: Optional[Path] = None,
        package_dir: Optional[Path] = None,
    ):
        if home is None:
            from .._env_io import memorizz_home

            home = memorizz_home()
        self.agent_id = str(agent_id)
        self.folder = Path(home).expanduser() / "self_aware" / _safe_id(agent_id)
        self.root = self.folder / "source"
        self.package = self.root / PACKAGE
        self._package_dir = Path(package_dir) if package_dir else None

    # --- git ---

    def _git(self, *args: str, check: bool = True) -> str:
        git = shutil.which("git")
        if git is None:
            raise SelfSourceError("git is needed to keep versions of the source")
        env = {
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": "cat",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        done = subprocess.run(
            [git, *_SAFE, *args],
            cwd=str(self.root),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if check and done.returncode != 0:
            raise SelfSourceError(
                f"git {args[0]} failed: {(done.stderr or done.stdout).strip()[:400]}"
            )
        return done.stdout

    # --- making the copy ---

    def exists(self) -> bool:
        return (self.root / ".git").exists() and self.package.is_dir()

    def ensure(self) -> Dict[str, Any]:
        """Make the copy if there is none yet, and say what is there."""
        if not self.exists():
            source = self._package_dir or installed_package_dir()
            if self.root.exists():
                shutil.rmtree(self.root)
            self.root.mkdir(parents=True)
            shutil.copytree(source, self.package, ignore=_IGNORED)
            version = installed_version()
            (self.root / "BASE_VERSION").write_text(version + "\n", encoding="utf-8")
            self._git("init", "-q")
            self._git("add", "-A")
            self._git(
                *_AUTHOR, "commit", "-q", "-m", f"MemoRizz {version} as installed"
            )
            self._git("tag", BASE_TAG)
        return self.status()

    def _require(self) -> None:
        if not self.exists():
            raise SelfSourceError("This agent has no copy of the source yet")

    # --- reading the state ---

    def base_version(self) -> str:
        try:
            return (self.root / "BASE_VERSION").read_text(encoding="utf-8").strip()
        except OSError:
            return "unknown"

    def changed_files(self) -> List[str]:
        """Files edited since the last kept version."""
        self._require()
        lines = self._git("status", "--porcelain").splitlines()
        return sorted(line[3:].strip().strip('"') for line in lines if line.strip())

    def versions(self) -> List[Dict[str, Any]]:
        """Every kept version, newest first; the last one is the install."""
        self._require()
        out = self._git(
            "log", "--pretty=format:\x1e%h\x1f%H\x1f%aI\x1f%s", "--shortstat"
        )
        base = self._git("rev-parse", BASE_TAG).strip()
        versions: List[Dict[str, Any]] = []
        for block in out.split("\x1e")[1:]:
            head, _, stat = block.partition("\n")
            short, full, when, message = head.split("\x1f", 3)

            def count(word: str) -> int:
                found = re.search(rf"(\d+) {word}", stat)
                return int(found.group(1)) if found else 0

            versions.append(
                {
                    "id": short,
                    "commit": full,
                    "at": when,
                    "message": message.strip(),
                    "files": count("file"),
                    "insertions": count("insertion"),
                    "deletions": count("deletion"),
                    "installed": full == base,
                }
            )
        total = len(versions)
        for index, version in enumerate(versions):
            # The install is version 0; each kept change counts up from it.
            version["number"] = total - 1 - index
        return versions

    def status(self) -> Dict[str, Any]:
        if not self.exists():
            return {
                "exists": False,
                "path": str(self.root),
                "installed_version": installed_version(),
            }
        versions = self.versions()
        return {
            "exists": True,
            "path": str(self.root),
            "base_version": self.base_version(),
            "installed_version": installed_version(),
            "head": versions[0]["id"],
            "version_number": versions[0]["number"],
            "versions": len(versions) - 1,
            "changed_files": self.changed_files(),
        }

    def diff(self, version: Optional[str] = None) -> str:
        """Changes not yet kept (default), one version's change, or ``"all"``
        for everything that differs from the install."""
        self._require()
        if version in (None, "", "working"):
            # Intent-to-add, so a file the agent created shows in the diff.
            self._git("add", "-A", "-N")
            text = self._git("diff", "HEAD")
        elif version == "all":
            self._git("add", "-A", "-N")
            text = self._git("diff", BASE_TAG)
        else:
            if not re.fullmatch(r"[0-9a-fA-F]{4,40}", str(version)):
                raise SelfSourceError("A version is named by its ID")
            text = self._git("show", "--format=", str(version))
        if len(text) > MAX_DIFF_CHARS:
            text = text[:MAX_DIFF_CHARS] + "\n… (diff cut short)\n"
        return text

    # --- changing the state ---

    def save_version(self, message: str) -> Optional[Dict[str, Any]]:
        """Keep the current edits as a new version. None when nothing changed."""
        self._require()
        if not self.changed_files():
            return None
        summary = re.sub(r"\s+", " ", str(message or "")).strip() or "Change"
        self._git("add", "-A")
        self._git(*_AUTHOR, "commit", "-q", "-m", summary[:200])
        return self.versions()[0]

    def discard_changes(self) -> List[str]:
        """Drop edits made since the last kept version."""
        self._require()
        dropped = self.changed_files()
        self._git("reset", "-q", "--hard", "HEAD")
        self._git("clean", "-fdq")
        return dropped

    def restore(self, version: str) -> Optional[Dict[str, Any]]:
        """Go back to how the source was at ``version``, as a new version, so
        nothing that was kept is ever lost."""
        self._require()
        if version == BASE_TAG:
            target = BASE_TAG
        elif re.fullmatch(r"[0-9a-fA-F]{4,40}", str(version)):
            target = str(version)
        else:
            raise SelfSourceError("A version is named by its ID")
        self.discard_changes()
        self._git("read-tree", "-u", "--reset", target)
        label = "the installed source" if target == BASE_TAG else f"version {target}"
        self._git("add", "-A")
        if not self._git("status", "--porcelain").strip():
            return None
        self._git(*_AUTHOR, "commit", "-q", "-m", f"Back to {label}")
        return self.versions()[0]

    def patch(self) -> str:
        """Everything kept so far, as one patch against the install."""
        self._require()
        return self._git("diff", BASE_TAG, "HEAD")

    # --- files ---

    def resolve(self, path: str, *, must_exist: bool = True) -> Path:
        """A path inside the copy. Anything that leaves it is refused."""
        text = str(path or "").strip().lstrip("/")
        if text == PACKAGE or text.startswith(PACKAGE + "/"):
            text = text[len(PACKAGE) :].lstrip("/")
        candidate = (self.package / text).resolve()
        package = self.package.resolve()
        if candidate != package and package not in candidate.parents:
            raise SelfSourceError("That path is outside the agent's copy of the source")
        if must_exist and not candidate.exists():
            raise SelfSourceError(f"No such file in the source: {PACKAGE}/{text}")
        return candidate

    def display(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.root.resolve()))

    def list_files(self, path: str = "") -> List[str]:
        self._require()
        folder = self.resolve(path)
        if folder.is_file():
            return [self.display(folder)]
        entries = []
        for child in sorted(folder.iterdir()):
            if child.name == "__pycache__" or child.name.startswith("."):
                continue
            entries.append(self.display(child) + ("/" if child.is_dir() else ""))
        return entries

    def read_file(self, path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Lines of a file, each with its number."""
        self._require()
        target = self.resolve(path)
        if not target.is_file():
            raise SelfSourceError(f"{self.display(target)} is a folder")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        first = max(1, int(start_line or 1))
        last = int(end_line or 0) or len(lines)
        last = min(len(lines), max(first, last))
        out: List[str] = []
        size = 0
        for number in range(first, last + 1):
            line = f"{number}\t{lines[number - 1]}"
            size += len(line) + 1
            if size > MAX_READ_CHARS:
                out.append(f"… cut short at line {number}; ask for a later start_line")
                break
            out.append(line)
        header = f"{self.display(target)} (lines {first}-{last} of {len(lines)})"
        return header + "\n" + "\n".join(out)

    # --- how a version came about ---

    def _commit(self, version: str) -> str:
        if not re.fullmatch(r"[0-9a-fA-F]{4,40}", str(version or "")):
            raise SelfSourceError("A version is named by its ID")
        return self._git("rev-parse", "--verify", f"{version}^{{commit}}").strip()

    def save_story(self, version: str, story: Dict[str, Any]) -> None:
        """Keep the account of how a version was made: the errors that were
        read, the prompt the coding model got, its steps and the check."""
        self._require()
        folder = self.folder / "fixes"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self._commit(version)}.json"
        path.write_text(
            json.dumps(story, ensure_ascii=False, default=str, indent=1),
            encoding="utf-8",
        )

    def story(self, version: str) -> Optional[Dict[str, Any]]:
        """The account kept with a version, or None when it was made by hand."""
        self._require()
        try:
            path = self.folder / "fixes" / f"{self._commit(version)}.json"
            found = json.loads(path.read_text(encoding="utf-8"))
        except (SelfSourceError, OSError, ValueError):
            return None
        return found if isinstance(found, dict) else None

    def file_view(
        self, path: str, version: Optional[str] = None, old: bool = False
    ) -> Dict[str, Any]:
        """A whole file as one change left it (or, with ``old``, as it was
        before), and the lines that change touched.

        ``version`` is read as in :meth:`diff`: edits not yet kept (default),
        one version, or ``"all"`` for everything since the install.
        """
        self._require()
        name = self.display(self.resolve(path, must_exist=False))
        diff_of = version or "working"
        if diff_of in ("working", "all"):
            revision = (BASE_TAG if diff_of == "all" else "HEAD") if old else None
        elif re.fullmatch(r"[0-9a-fA-F]{4,40}", str(diff_of)):
            revision = f"{diff_of}~1" if old else str(diff_of)
        else:
            raise SelfSourceError("A version is named by its ID")
        if revision is None:
            target = self.resolve(path)
            if not target.is_file():
                raise SelfSourceError(f"{name} is a folder")
            text = target.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                text = self._git("show", f"{revision}:{name}")
            except SelfSourceError as exc:
                raise SelfSourceError(f"{name} did not exist at that point") from exc
        cut = len(text) > MAX_VIEW_CHARS
        if cut:
            text = text[:MAX_VIEW_CHARS]

        added: List[int] = []
        removed: List[int] = []
        # Where lines were taken out, by the line that now follows them.
        removed_before: List[int] = []
        for changed in parse_diff(self.diff(diff_of)):
            if name not in (changed["path"], changed["old_path"]):
                continue
            for hunk in changed["hunks"]:
                following = None
                for line in reversed(hunk["lines"]):
                    if line["kind"] == "add":
                        added.append(line["new"])
                        following = line["new"]
                    elif line["kind"] == "del":
                        removed.append(line["old"])
                        if following is not None:
                            removed_before.append(following)
                    else:
                        following = line["new"]
        return {
            "path": name,
            "version": diff_of,
            "side": "old" if old else "new",
            "text": text,
            "cut_short": cut,
            "changed": sorted(set(removed if old else added)),
            "removed_before": [] if old else sorted(set(removed_before) - set(added)),
        }

    def search(self, text: str, path: str = "", limit: int = 60) -> List[str]:
        """Lines containing ``text`` (not a pattern), as ``file:line: text``."""
        self._require()
        needle = str(text or "")
        if not needle.strip():
            raise SelfSourceError("Nothing to search for")
        folder = self.resolve(path)
        files = [folder] if folder.is_file() else sorted(folder.rglob("*"))
        hits: List[str] = []
        for file in files:
            if not file.is_file() or "__pycache__" in file.parts:
                continue
            if file.suffix not in {
                ".py",
                ".html",
                ".js",
                ".css",
                ".md",
                ".sql",
                ".json",
            }:
                continue
            try:
                content = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if needle not in content:
                continue
            for number, line in enumerate(content.splitlines(), 1):
                if needle in line:
                    hits.append(f"{self.display(file)}:{number}: {line.strip()[:200]}")
                    if len(hits) >= limit:
                        return hits + ["… more matches; narrow the search"]
        return hits

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        """Replace one exact piece of a file. ``old_text`` must occur once."""
        self._require()
        target = self.resolve(path)
        content = target.read_text(encoding="utf-8")
        found = content.count(old_text) if old_text else 0
        if found == 0:
            raise SelfSourceError(
                "old_text is not in the file. Read the lines again and copy "
                "them exactly, with their indentation."
            )
        if found > 1:
            raise SelfSourceError(
                f"old_text occurs {found} times. Include more of the lines "
                "around it so it names one place."
            )
        target.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {self.display(target)}"

    def create_file(self, path: str, content: str) -> str:
        self._require()
        target = self.resolve(path, must_exist=False)
        if target.exists():
            raise SelfSourceError("That file exists; edit it instead")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
        return f"Created {self.display(target)}"

    # --- checking ---

    def check(self) -> Dict[str, Any]:
        """Does the copy still compile and import? Run apart from this process."""
        self._require()
        changed = self._git("diff", "--name-only", BASE_TAG).splitlines()
        changed += self._git("ls-files", "--others", "--exclude-standard").splitlines()
        python_files = sorted(
            {name for name in changed if name.endswith(".py")}
            & {name for name in changed if (self.root / name).is_file()}
        )
        env = {
            **os.environ,
            "PYTHONPATH": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        steps: List[Dict[str, Any]] = []

        def run(name: str, code: str, *args: str) -> None:
            try:
                done = subprocess.run(
                    [sys.executable, "-c", code, *args],
                    cwd=str(self.root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                output = (done.stdout + done.stderr).strip()
                ok = done.returncode == 0
            except subprocess.TimeoutExpired:
                output, ok = "Timed out after 180 seconds", False
            steps.append({"name": name, "ok": ok, "output": output[-4000:]})

        if python_files:
            run(
                "The changed files compile",
                "import sys\n"
                "bad = 0\n"
                "for name in sys.argv[1:]:\n"
                "    try:\n"
                "        compile(open(name, encoding='utf-8').read(), name, 'exec')\n"
                "    except SyntaxError as exc:\n"
                "        bad = 1; print(f'{name}:{exc.lineno}: {exc.msg}')\n"
                "sys.exit(bad)\n",
                *python_files,
            )
        modules = [
            name[:-3].replace("/", ".").removesuffix(".__init__")
            for name in python_files
            if name.startswith(PACKAGE + "/")
        ]
        run(
            "MemoRizz imports from the copy",
            "import sys, importlib, pathlib, memorizz\n"
            "here = pathlib.Path(memorizz.__file__).resolve()\n"
            "assert pathlib.Path.cwd().resolve() in here.parents, "
            "f'imported {here}, not the copy'\n"
            "for name in sys.argv[1:]:\n"
            "    importlib.import_module(name)\n"
            "print('memorizz', memorizz.__version__, 'and', len(sys.argv) - 1, "
            "'changed modules import')\n",
            *modules,
        )
        # A change to the console's pages or scripts never touches Python, so
        # the steps above would pass a page that cannot be drawn.
        pages = sorted(
            name
            for name in set(changed)
            if name.endswith(".html")
            and "/templates/" in name
            and (self.root / name).is_file()
        )
        if pages:
            run(
                "The console's changed pages compile",
                "import sys, pathlib\n"
                "from jinja2 import Environment, FileSystemLoader\n"
                "class Any(dict):\n"
                "    # The console adds its own filters and tests when it starts;\n"
                "    # here only the page's syntax is in question.\n"
                "    def __contains__(self, key): return True\n"
                "    def __missing__(self, key): return lambda value, *a, **k: value\n"
                "    def get(self, key, default=None): return self[key]\n"
                "bad = 0\n"
                "for name in sys.argv[1:]:\n"
                "    folder, _, page = name.partition('/templates/')\n"
                "    env = Environment(loader=FileSystemLoader(folder + '/templates'))\n"
                "    env.filters, env.tests = Any(env.filters), Any(env.tests)\n"
                "    try:\n"
                "        env.get_template(page)\n"
                "    except Exception as exc:\n"
                "        bad = 1\n"
                "        line = getattr(exc, 'lineno', None)\n"
                "        print(f\"{name}{':' + str(line) if line else ''}: {exc}\")\n"
                "sys.exit(bad)\n",
                *pages,
            )
        scripts = sorted(
            name
            for name in set(changed)
            if name.endswith(".js") and (self.root / name).is_file()
        )
        node = shutil.which("node")
        if scripts and node:
            failed = []
            for name in scripts:
                try:
                    done = subprocess.run(
                        [node, "--check", name],
                        cwd=str(self.root),
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if done.returncode != 0:
                        failed.append((done.stderr or done.stdout).strip()[-800:])
                except (OSError, subprocess.TimeoutExpired) as exc:
                    failed.append(f"{name}: {exc}")
            steps.append(
                {
                    "name": "The console's changed scripts parse",
                    "ok": not failed,
                    "output": "\n".join(failed)[-4000:],
                }
            )
        return {"ok": all(step["ok"] for step in steps), "steps": steps}

    # --- a version on its own, to run and look at ---

    def checkout(self, version: str) -> Path:
        """A folder holding the source exactly as one version left it, apart
        from the copy the agent edits. Made once per version and reused."""
        self._require()
        commit = self._commit(version)
        target = self.folder / "previews" / commit[:12]
        if (target / PACKAGE).is_dir():
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        self._git("worktree", "prune")
        self._git("worktree", "add", "--detach", "--force", str(target), commit)
        return target

    def remove_checkout(self, version: str) -> None:
        try:
            commit = self._commit(version)
        except SelfSourceError:
            return
        target = self.folder / "previews" / commit[:12]
        if target.exists():
            self._git("worktree", "remove", "--force", str(target), check=False)
            shutil.rmtree(target, ignore_errors=True)
            self._git("worktree", "prune", check=False)
