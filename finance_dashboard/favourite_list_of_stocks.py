from __future__ import annotations

import json
from pathlib import Path
from typing import List


class FavouriteListOfStocks:
    def __init__(self, storage_path: str | Path):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self.storage_path.write_text("[]", encoding="utf-8")

    def _load(self) -> List[str]:
        try:
            with self.storage_path.open("r", encoding="utf-8") as handle:
                content = json.load(handle)
                if isinstance(content, list):
                    return [str(item).upper() for item in content]
        except (json.JSONDecodeError, OSError):
            return []
        return []

    def _save(self, items: List[str]) -> None:
        with self.storage_path.open("w", encoding="utf-8") as handle:
            json.dump(items, handle)

    def add(self, ticker: str) -> List[str]:
        normalized = ticker.strip().upper()
        if not normalized:
            return self.list()
        items = self._load()
        if normalized not in items:
            items.append(normalized)
            self._save(items)
        return items

    def remove(self, ticker: str) -> List[str]:
        normalized = ticker.strip().upper()
        items = self._load()
        updated = [item for item in items if item != normalized]
        self._save(updated)
        return updated

    def list(self) -> List[str]:
        return self._load()
