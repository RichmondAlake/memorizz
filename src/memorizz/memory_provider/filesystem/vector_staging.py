"""Re-embed a filesystem store beside its live vectors, then switch over.

Staged vectors are written to sidecar files, never into the documents, while
search keeps reading each document's ``embedding``:

    <root>/.vector-staging/next/<store>/<id>.json  {"embedding", "sig"}

The switch writes each staged vector into its document and keeps the vector
it replaces under ``prev/``, so swapping back moves ``prev/`` into the staged
slot and only re-embeds what changed. A filesystem store holds any vector
size (each size gets its own FAISS index), so changing size needs nothing
more.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ...enums.memory_type import MemoryType

_SLOTS = ".vector-staging"


class FileSystemVectorStaging:
    """Staged re-embedding for ``FileSystemProvider``."""

    VECTOR_STAGING = {"fixed_size": False, "resize": True}

    def _slot(self, name: str) -> Path:
        return Path(self.root_path) / _SLOTS / name

    @staticmethod
    def _write_json(path: Path, value: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        tmp.replace(path)

    def vector_stage_begin(
        self, dimensions: int, *, reuse_previous: bool = False
    ) -> None:
        nxt, prev = self._slot("next"), self._slot("prev")
        shutil.rmtree(nxt, ignore_errors=True)
        if reuse_previous:
            if not prev.is_dir():
                raise RuntimeError("No previous vectors are kept in this store")
            prev.rename(nxt)
        else:
            nxt.mkdir(parents=True, exist_ok=True)

    def _document_id(
        self, memory_type: MemoryType, candidates: Sequence[str]
    ) -> Optional[str]:
        for value in candidates:
            try:
                if self._document_path(memory_type, str(value)).exists():
                    return str(value)
            except ValueError:
                continue
        return None

    def vector_stage_keys(
        self, memory_type: MemoryType, candidates: Sequence[str]
    ) -> List[str]:
        return [str(value) for value in candidates]

    def vector_stage_signatures(self, memory_type: MemoryType) -> Dict[str, str]:
        folder = self._slot("next") / memory_type.value
        out: Dict[str, str] = {}
        if folder.is_dir():
            for path in folder.glob("*.json"):
                try:
                    out[path.stem] = (
                        json.loads(path.read_text(encoding="utf-8")).get("sig") or ""
                    )
                except (OSError, ValueError):
                    continue
        return out

    def vector_stage_put(
        self,
        memory_type: MemoryType,
        items: Sequence[Tuple[Sequence[str], Sequence[float], str]],
    ) -> List[bool]:
        done = []
        folder = self._slot("next") / memory_type.value
        for candidates, vector, sig in items:
            doc_id = self._document_id(memory_type, candidates)
            if doc_id is None:
                done.append(False)
                continue
            self._write_json(
                folder / f"{doc_id}.json",
                {"embedding": [float(x) for x in vector], "sig": sig},
            )
            done.append(True)
        return done

    def vector_stage_switch(self, staged: Iterable[MemoryType]) -> Dict[str, Any]:
        nxt, prev = self._slot("next"), self._slot("prev")
        if not nxt.is_dir():
            raise RuntimeError("Nothing is staged in this store")
        shutil.rmtree(prev, ignore_errors=True)
        switched: Dict[str, int] = {}
        for folder in sorted(p for p in nxt.iterdir() if p.is_dir()):
            try:
                memory_type = MemoryType(folder.name)
            except ValueError:
                continue
            count = 0
            with self._locks[memory_type]:
                for path in folder.glob("*.json"):
                    staged_vector = json.loads(path.read_text(encoding="utf-8"))
                    document = self._read_document(memory_type, path.stem)
                    if not document:
                        continue
                    if document.get("embedding"):
                        self._write_json(
                            prev / memory_type.value / path.name,
                            {
                                "embedding": document["embedding"],
                                "sig": staged_vector.get("sig"),
                            },
                        )
                    document["embedding"] = staged_vector["embedding"]
                    self._write_document(
                        memory_type, path.stem, document, persist_index=False
                    )
                    count += 1
                self._save_index(memory_type)
                self._mark_vector_index_dirty(memory_type)
            switched[memory_type.value] = count
        shutil.rmtree(nxt, ignore_errors=True)
        return {"switched": switched, "carried": {}, "dropped": {}, "indexes": []}

    def vector_stage_drop(self) -> None:
        shutil.rmtree(self._slot("next"), ignore_errors=True)

    def vector_previous(self) -> Optional[Dict[str, Any]]:
        prev = self._slot("prev")
        files = list(prev.glob("*/*.json")) if prev.is_dir() else []
        if not files:
            return None
        try:
            size = len(json.loads(files[0].read_text(encoding="utf-8"))["embedding"])
        except (OSError, ValueError, KeyError, TypeError):
            size = None
        return {"dimensions": size, "rows": len(files)}

    def vector_previous_drop(self) -> None:
        shutil.rmtree(self._slot("prev"), ignore_errors=True)


__all__ = ["FileSystemVectorStaging"]
