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

    def get_stock_snapshot(self, ticker: str, start_date: str, end_date: str) -> dict:
        normalized = self.normalize_tickers([ticker])[0]
        frame = self.fetch_stock_data(normalized, start_date, end_date)
        close_values = frame["Close"].dropna()
        if close_values.empty:
            raise ValueError(f"No close values available for {normalized}.")

        current_price = float(close_values.iloc[-1])
        previous_price = float(close_values.iloc[-2]) if len(close_values) >= 2 else current_price
        day_change = current_price - previous_price
        day_return = (day_change / previous_price) if previous_price else 0.0

        return {
            "ticker": normalized,
            "current_price": current_price,
            "previous_price": previous_price,
            "day_change": day_change,
            "day_return": day_return,
            "last_updated": close_values.index[-1],
        }

    def build_portfolio_snapshot(
        self,
        holdings: Dict[str, Dict[str, float]],
        prices: Dict[str, float],
        cash_balance: float,
    ) -> Dict[str, float | Dict[str, float]]:
        invested = 0.0
        for ticker, position in holdings.items():
            shares = float(position.get("shares", 0.0))
            avg_cost = float(position.get("avg_cost", 0.0))
            market_price = float(prices.get(ticker, avg_cost))
            invested += shares * market_price

        total_value = float(cash_balance) + invested
        return {
            "cash": float(cash_balance),
            "invested": invested,
            "total_value": total_value,
            "holdings": holdings,
        }

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
