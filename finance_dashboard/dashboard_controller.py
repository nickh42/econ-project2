from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd
import yfinance as yf

from .stock_comparator import StockComparator
from .dashboard import Dashboard


class DashboardController:
    def __init__(self, favorites=None):
        self.favorites = favorites

    def normalize_tickers(self, tickers: Iterable[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for ticker in tickers:
            cleaned = str(ticker).strip().upper()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
        return result

    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            actions=False,
        )

        if data.empty:
            raise ValueError(f"No market data available for {ticker}.")

        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data.xs(ticker, axis=1, level=1)
            except KeyError:
                try:
                    data = data.xs(ticker, axis=1, level=0)
                except KeyError as exc:
                    raise ValueError(f"Ticker {ticker} returned malformed market data.") from exc

        if "Close" not in data.columns:
            raise ValueError(f"Close prices not available for {ticker}.")

        return data

    def build_dashboard(self, tickers: Iterable[str], start_date: str, end_date: str) -> Dashboard:
        normalized = self.normalize_tickers(tickers)
        if len(normalized) > 3:
            raise ValueError("You can compare up to three tickers at a time.")

        data: Dict[str, pd.DataFrame] = {}
        summary: Dict[str, dict] = {}
        returns: Dict[str, pd.Series] = {}
        errors: Dict[str, str] = {}

        for ticker in normalized:
            try:
                frame = self.fetch_stock_data(ticker, start_date, end_date)
                data[ticker] = frame
                summary[ticker] = StockComparator.compute_summary({ticker: frame})[ticker]
                returns[ticker] = summary[ticker]["returns"]
            except Exception as exc:  # pragma: no cover - user-facing error path
                errors[ticker] = f"Unable to pull data for {ticker}: {exc}"

        return Dashboard(
            tickers=normalized,
            start_date=start_date,
            end_date=end_date,
            data=data,
            summary=summary,
            returns=returns,
            errors=errors,
        )
