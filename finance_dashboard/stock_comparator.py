from __future__ import annotations

from typing import Dict

import pandas as pd


class StockComparator:
    @staticmethod
    def daily_returns(prices: pd.Series) -> pd.Series:
        if not isinstance(prices, pd.Series):
            raise TypeError("daily_returns expects a pandas Series of prices.")

        cleaned = prices.sort_index().dropna()
        if cleaned.empty:
            return pd.Series(dtype=float)

        returns = cleaned.pct_change().dropna().round(3)
        return returns

    @staticmethod
    def resample_data(data: Dict[str, pd.DataFrame], frequency: str) -> Dict[str, pd.DataFrame]:
        mapping = {"daily": "D", "weekly": "W", "monthly": "ME", "yearly": "YE"}
        rule = mapping.get(str(frequency).lower(), "D")

        resampled: Dict[str, pd.DataFrame] = {}
        for ticker, frame in data.items():
            if frame.empty:
                resampled[ticker] = frame.copy()
                continue

            if isinstance(frame.columns, pd.MultiIndex):
                if "Close" in frame.columns.get_level_values(0):
                    close_frame = frame.xs("Close", axis=1, level=0)
                elif "Close" in frame.columns.get_level_values(1):
                    close_frame = frame.xs("Close", axis=1, level=1)
                else:
                    close_frame = frame
            else:
                close_frame = frame

            if isinstance(close_frame, pd.DataFrame) and "Close" in close_frame.columns:
                resampled[ticker] = close_frame[["Close"]].resample(rule).last().dropna()
            elif isinstance(close_frame, pd.DataFrame) and close_frame.shape[1] == 1:
                resampled[ticker] = close_frame.resample(rule).last().dropna()
            else:
                resampled[ticker] = pd.DataFrame(index=frame.index)

        return resampled

    @staticmethod
    def compute_summary(data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        outputs: Dict[str, Dict[str, float]] = {}

        for ticker, frame in data.items():
            if frame.empty:
                outputs[ticker] = {
                    "returns": pd.Series(dtype=float),
                    "volatility": 0.0,
                    "avg_return": 0.0,
                    "min_return": 0.0,
                    "max_return": 0.0,
                }
                continue

            close_values = None
            if isinstance(frame.columns, pd.MultiIndex):
                if "Close" in frame.columns.get_level_values(0):
                    close_values = frame.xs("Close", axis=1, level=0)
                elif "Close" in frame.columns.get_level_values(1):
                    close_values = frame.xs("Close", axis=1, level=1)
            else:
                close_values = frame

            if close_values is None or (isinstance(close_values, pd.DataFrame) and "Close" not in close_values.columns and not close_values.shape[1]):
                outputs[ticker] = {
                    "returns": pd.Series(dtype=float),
                    "volatility": 0.0,
                    "avg_return": 0.0,
                    "min_return": 0.0,
                    "max_return": 0.0,
                }
                continue

            if isinstance(close_values, pd.DataFrame):
                if "Close" in close_values.columns:
                    series = close_values["Close"].dropna()
                elif close_values.shape[1] == 1:
                    series = close_values.iloc[:, 0].dropna()
                else:
                    series = pd.Series(dtype=float)
            else:
                series = close_values.dropna()

            returns = StockComparator.daily_returns(series)
            volatility = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
            outputs[ticker] = {
                "returns": returns,
                "volatility": volatility,
                "avg_return": float(returns.mean()) if not returns.empty else 0.0,
                "min_return": float(returns.min()) if not returns.empty else 0.0,
                "max_return": float(returns.max()) if not returns.empty else 0.0,
            }

        return outputs

    @staticmethod
    def volatility_over_time(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        results: Dict[str, pd.Series] = {}

        for ticker, frame in data.items():
            if frame.empty:
                results[ticker] = pd.Series(dtype=float)
                continue

            close_values = None
            if isinstance(frame.columns, pd.MultiIndex):
                if "Close" in frame.columns.get_level_values(0):
                    close_values = frame.xs("Close", axis=1, level=0)
                elif "Close" in frame.columns.get_level_values(1):
                    close_values = frame.xs("Close", axis=1, level=1)
            else:
                close_values = frame

            if close_values is None:
                results[ticker] = pd.Series(dtype=float)
                continue

            if isinstance(close_values, pd.DataFrame):
                if "Close" in close_values.columns:
                    series = close_values["Close"].dropna()
                elif close_values.shape[1] == 1:
                    series = close_values.iloc[:, 0].dropna()
                else:
                    series = pd.Series(dtype=float)
            else:
                series = close_values.dropna()

            if series.empty:
                results[ticker] = pd.Series(dtype=float)
                continue

            returns = StockComparator.daily_returns(series)
            volatility = returns.rolling(window=5, min_periods=2).std().fillna(0.0).round(6)
            aligned = pd.Series(0.0, index=series.index, dtype=float)
            if not volatility.empty:
                aligned.loc[volatility.index] = volatility.values
            results[ticker] = aligned

        return results

    @staticmethod
    def correlation_table(data: Dict[str, pd.Series]) -> pd.DataFrame:
        frame = pd.DataFrame(data)
        if frame.empty:
            return pd.DataFrame()
        return frame.corr()

    @staticmethod
    def compare_three_stocks(data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        return StockComparator.compute_summary(data)
