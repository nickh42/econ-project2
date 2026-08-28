from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Dashboard:
    tickers: List[str]
    start_date: str
    end_date: str
    data: Dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    returns: Dict[str, pd.Series] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    selected_view: str = "Returns"
