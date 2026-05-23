# Copyright (c) 2024 Richmond Alake. All rights reserved.
# Licensed under the PolyForm Noncommercial License 1.0.0.
# See LICENSE file in the project root for full license information.

"""Text extraction for knowledge-base ingestion.

This module is the single source of truth for turning a file (bytes, path,
or file-like object) into a plain-text corpus that ``KnowledgeBase`` can
ingest. Both the Python SDK (``KnowledgeBase.ingest_file`` /
``ingest_directory``) and the local UI's upload endpoint route through
these helpers, so we never maintain two copies of the format-handling
logic.

Adding a new file type is a one-liner::

    from memorizz.long_term.semantic.extractors import register_extractor

    def extract_docx(raw: bytes) -> str:
        import docx, io
        return "\\n\\n".join(p.text for p in docx.Document(io.BytesIO(raw)).paragraphs)

    register_extractor(".docx", extract_docx)

Error hierarchy::

    ExtractorError
      ├─ UnsupportedFileType            (no extractor registered)
      ├─ MissingExtractorDependency     (extractor exists but optional dep is missing)
      ├─ EmptyDocumentError             (extracted text was empty)
      └─ ExtractionError                (anything else went wrong inside the extractor)
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

#: An extractor takes raw file bytes and returns plain text.
Extractor = Callable[[bytes], str]

#: What callers can pass to ``extract_text`` / ``ingest_file``.
FileSource = Union[str, os.PathLike, bytes, bytearray, memoryview]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ExtractorError(Exception):
    """Base class for all text-extraction errors."""


class UnsupportedFileType(ExtractorError):
    """No extractor is registered for this file extension."""


class MissingExtractorDependency(ExtractorError):
    """Extractor is registered but its optional dependency isn't installed.

    The message includes the exact ``pip install`` command to fix it.
    """


class EmptyDocumentError(ExtractorError):
    """Extraction ran but produced no text (e.g. scanned-image PDF, empty file)."""


class ExtractionError(ExtractorError):
    """Extraction failed for reasons specific to the file (corrupt, password, etc.)."""


# ---------------------------------------------------------------------------
# Built-in extractors
# ---------------------------------------------------------------------------


def extract_plaintext(raw: bytes) -> str:
    """Decode as UTF-8 text, falling back to latin-1 for legacy files."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # latin-1 never errors; good enough for legacy text-y files.
        return raw.decode("latin-1")


def extract_pdf(raw: bytes) -> str:
    """Extract text from a PDF using ``pypdf``.

    Optional dependency: ``pip install 'memorizz[ingest-pdf]'``. Raises
    :class:`MissingExtractorDependency` if pypdf isn't importable.
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise MissingExtractorDependency(
            "PDF ingestion requires `pypdf`. Install it with: "
            "pip install 'memorizz[ingest-pdf]'"
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as exc:
        raise ExtractionError(f"pypdf could not open the PDF: {exc}") from exc

    pages: List[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:
            logger.debug("pypdf extract_text failed on a page: %s", exc)
            pages.append("")
    return "\n\n".join(p.strip() for p in pages if p.strip())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Mapping of lower-case extension (including the leading dot) to an extractor.
# Kept module-global so ``register_extractor`` mutates the same table used by
# the SDK and UI. Ordering doesn't matter.
_TEXT_EXTENSIONS: Tuple[str, ...] = (
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".json",
    ".jsonl",
    ".csv",
    ".tsv",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",
    ".html",
    ".htm",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".tex",
)

EXTRACTORS: Dict[str, Extractor] = {
    **{ext: extract_plaintext for ext in _TEXT_EXTENSIONS},
    ".pdf": extract_pdf,
}


def register_extractor(extension: str, extractor: Extractor) -> None:
    """Register or override an extractor for a file extension.

    Parameters
    ----------
    extension:
        File extension, with or without a leading dot. Case-insensitive.
    extractor:
        Callable mapping raw bytes to extracted text. May raise
        :class:`MissingExtractorDependency`, :class:`ExtractionError`, or
        return an empty string (caller decides how to handle empty).
    """
    normalized = extension.lower()
    if not normalized.startswith("."):
        normalized = "." + normalized
    EXTRACTORS[normalized] = extractor


def supported_extensions() -> List[str]:
    """Return the sorted list of currently-registered extensions."""
    return sorted(EXTRACTORS.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_text(
    source: FileSource,
    *,
    filename: Optional[str] = None,
) -> str:
    """Extract plain text from a file path, raw bytes, or file-like object.

    Parameters
    ----------
    source:
        One of:
          - ``str`` / ``PathLike`` path to a file on disk
          - raw ``bytes`` / ``bytearray`` / ``memoryview``
          - an open binary file-like object (``.read()`` returns bytes)
    filename:
        Required when ``source`` is bytes or a file-like without a ``.name``
        attribute, because the extension is how we pick the extractor.

    Raises
    ------
    UnsupportedFileType:
        If no extractor is registered for the file's extension.
    MissingExtractorDependency:
        If the matching extractor needs an optional package that isn't installed.
    ExtractionError:
        If extraction failed (corrupt file, password-protected, etc.).
    EmptyDocumentError:
        If the extractor ran but returned no text — typically a scanned PDF
        that needs OCR, or a genuinely empty file.
    """
    raw, name = _read_source(source, filename)
    extension = _extension_for(name)
    extractor = EXTRACTORS.get(extension)
    if extractor is None:
        raise UnsupportedFileType(
            f"No extractor registered for '{extension}' (from {name!r}). "
            f"Registered: {', '.join(supported_extensions())}. "
            "Use memorizz.long_term.semantic.extractors.register_extractor "
            "to add support for a new format."
        )
    text = extractor(raw)
    if not text or not text.strip():
        raise EmptyDocumentError(
            f"Extractor for '{extension}' returned no text for {name!r}. "
            "This often indicates a scanned/image PDF that requires OCR."
        )
    return text


def iter_ingestable_files(
    root: Union[str, os.PathLike],
    *,
    recursive: bool = True,
    pattern: Optional[str] = None,
    extensions: Optional[Iterable[str]] = None,
) -> Iterator[Path]:
    """Walk ``root`` and yield files that have a registered extractor.

    Parameters
    ----------
    root:
        Directory to scan. Must exist and be a directory.
    recursive:
        If True, walk the entire tree; if False, only immediate children.
    pattern:
        Optional glob pattern (e.g. ``"*.pdf"``) applied before the
        extension filter. Combine with ``extensions=None`` to use *only*
        the glob.
    extensions:
        Iterable of extensions to keep (``".pdf"`` or ``"pdf"``). Defaults
        to ``supported_extensions()``; pass an explicit list to narrow the
        scan (e.g. only PDFs).
    """
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Directory not found: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    if extensions is None:
        allowed = set(supported_extensions())
    else:
        allowed = {
            e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions
        }

    if pattern:
        iterator = root_path.rglob(pattern) if recursive else root_path.glob(pattern)
    else:
        iterator = root_path.rglob("*") if recursive else root_path.iterdir()

    for path in iterator:
        if not path.is_file():
            continue
        if _extension_for(path.name) not in allowed:
            continue
        yield path


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _extension_for(name: str) -> str:
    """Return the lower-cased final extension of ``name``, including the dot."""
    return os.path.splitext(name)[1].lower()


def _read_source(source: FileSource, filename: Optional[str]) -> Tuple[bytes, str]:
    """Normalize any accepted ``FileSource`` into ``(raw_bytes, logical_name)``."""
    # Path-ish
    if isinstance(source, (str, os.PathLike)) and not isinstance(
        source, (bytes, bytearray)
    ):
        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.is_dir():
            raise IsADirectoryError(
                f"{path} is a directory; use ingest_directory() or iter_ingestable_files()."
            )
        return path.read_bytes(), filename or path.name

    # Raw bytes
    if isinstance(source, (bytes, bytearray, memoryview)):
        raw = bytes(source)
        if not filename:
            raise ValueError(
                "When passing raw bytes, 'filename' is required so the correct "
                "extractor can be chosen by extension."
            )
        return raw, filename

    # File-like (has .read)
    reader = getattr(source, "read", None)
    if callable(reader):
        raw = reader()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        if not filename:
            filename = getattr(source, "name", None)
        if not filename:
            raise ValueError(
                "File-like source has no .name; pass filename=... so the extension "
                "can be detected."
            )
        return bytes(raw), os.path.basename(str(filename))

    raise TypeError(
        "Unsupported source type. Pass a path (str/PathLike), raw bytes, "
        "or a binary file-like object."
    )
