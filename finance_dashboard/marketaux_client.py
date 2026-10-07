from __future__ import annotations

from typing import Any, Dict, List
import requests
from datetime import datetime


class MarketauxClient:
    BASE_URL = "https://api.marketaux.com/v1/news/all"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_news_for_ticker(self, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
        # Request a slightly larger page from the API so we can filter down
        api_limit = max(limit, 20)
        params = {
            "symbols": ticker,
            "language": "en",
            "filter_entities": "true",
            "limit": limit,
            "api_token": self.api_key,
        }

        resp = requests.get(self.BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("data", []) if isinstance(data, dict) else []

        # Normalize and sort by newest first
        def parse_date(item: Dict[str, Any]) -> datetime:
            date_str = item.get("published_at") or item.get("published_at_iso") or item.get("published_at_date")
            if not date_str:
                return datetime.min
            try:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except Exception:
                try:
                    return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S%z")
                except Exception:
                    return datetime.min

        articles = sorted(articles, key=parse_date, reverse=True)

        # Only keep articles where the ticker appears in entities
        filtered: List[Dict[str, Any]] = []
        for a in articles:
            entities = a.get("entities", [])
            symbols = [e.get("symbol") for e in entities if isinstance(e, dict) and e.get("symbol")]
            if ticker.upper() in [s.upper() for s in symbols]:
                filtered.append(a)
            if len(filtered) >= limit:
                break

        # Normalize sentiment fields onto each article for easier consumption by the UI
        for art in filtered:
            score = None
            label = None

            # First prefer per-entity sentiment for the matching ticker (if present)
            entities = art.get("entities", []) or []
            matched_entity = None
            for e in entities:
                if not isinstance(e, dict):
                    continue
                symbol = e.get("symbol")
                if symbol and symbol.upper() == ticker.upper():
                    matched_entity = e
                    break

            if matched_entity:
                # Marketaux sometimes uses `sentiment_score` or `sentiment`
                score = matched_entity.get("sentiment_score") or matched_entity.get("sentiment")
                # Some providers include per-highlight sentiment only; we'll fall back to highlights below
                # label may not be present; map from numeric score later
            else:
                # Fallback to top-level sentiment fields
                raw = art.get("sentiment")
                if isinstance(raw, dict):
                    score = raw.get("score") or raw.get("sentiment_score")
                    label = raw.get("label") or raw.get("sentiment")
                else:
                    if art.get("sentiment_score") is not None:
                        score = art.get("sentiment_score")
                    if art.get("sentiment_label") is not None:
                        label = art.get("sentiment_label")

            # Normalize numeric types
            try:
                if isinstance(score, str):
                    score = float(score)
            except Exception:
                score = None

            # If we still don't have a per-entity score, try to average highlight sentiments
            if score is None:
                highlights = art.get("highlights", []) or []
                vals = []
                for h in highlights:
                    if isinstance(h, dict) and (h.get("sentiment") is not None):
                        try:
                            vals.append(float(h.get("sentiment")))
                        except Exception:
                            continue
                if vals:
                    score = sum(vals) / len(vals)

            # Derive a human label if none provided
            if label is None and isinstance(score, (int, float)):
                # Use thresholds appropriate for scores in [-1,1] or [0,1]
                if score >= 0.55:
                    label = "positive"
                elif score <= 0.45:
                    label = "negative"
                else:
                    label = "neutral"

            art["sentiment_score"] = score
            art["sentiment_label"] = label

        return filtered
