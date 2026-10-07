from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TradeEngine:
    """Execute and deduplicate stock buy/sell orders."""

    processed_trade_keys: set[tuple[str, str, float, float]] = field(default_factory=set)

    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        return str(ticker).strip().upper()

    @classmethod
    def _make_order_key(cls, side: str, ticker: str, shares: float, price: float) -> tuple[str, str, float, float]:
        return (side.lower(), cls.normalize_ticker(ticker), float(shares), float(price))

    def execute_order(
        self,
        *,
        side: str,
        ticker: str,
        shares: float,
        price: float,
        cash_balance: float,
        holdings: dict[str, dict[str, float]],
    ) -> tuple[float, dict[str, dict[str, float]]]:
        normalized_side = str(side).strip().lower()
        if normalized_side not in {"buy", "sell"}:
            raise ValueError("Trade side must be 'buy' or 'sell'.")

        normalized_ticker = self.normalize_ticker(ticker)
        qty = float(shares)
        unit_price = float(price)

        if qty <= 0:
            raise ValueError("Number of shares must be greater than zero.")

        order_key = self._make_order_key(normalized_side, normalized_ticker, qty, unit_price)
        if order_key in self.processed_trade_keys:
            raise ValueError("Duplicate trade rejected.")

        new_holdings: dict[str, dict[str, float]] = {k: dict(v) for k, v in holdings.items()}
        updated_cash = float(cash_balance)

        if normalized_side == "buy":
            total_cost = qty * unit_price
            if updated_cash < total_cost:
                raise ValueError("Insufficient funds for this buy order.")

            current = new_holdings.get(normalized_ticker, {"shares": 0.0, "avg_cost": 0.0})
            existing_shares = float(current.get("shares", 0.0))
            existing_cost_basis = float(current.get("avg_cost", 0.0))
            new_shares = existing_shares + qty
            new_average_cost = (
                ((existing_shares * existing_cost_basis) + (qty * unit_price)) / new_shares
                if new_shares > 0
                else unit_price
            )
            updated_cash -= total_cost
            new_holdings[normalized_ticker] = {
                "shares": new_shares,
                "avg_cost": new_average_cost,
            }
        else:
            current = new_holdings.get(normalized_ticker, {"shares": 0.0, "avg_cost": 0.0})
            owned_shares = float(current.get("shares", 0.0))
            if qty > owned_shares:
                raise ValueError(f"Not enough shares available to sell {normalized_ticker}.")

            proceeds = qty * unit_price
            updated_cash += proceeds
            remaining_shares = owned_shares - qty
            if remaining_shares > 0:
                new_holdings[normalized_ticker] = {
                    "shares": remaining_shares,
                    "avg_cost": float(current.get("avg_cost", unit_price)),
                }
            else:
                new_holdings.pop(normalized_ticker, None)

        self.processed_trade_keys.add(order_key)
        return updated_cash, new_holdings
