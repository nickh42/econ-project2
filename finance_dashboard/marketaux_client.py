from __future__ import annotations

from typing import Any, Dict, List
import requests
from datetime import datetime, timezone


class MarketauxClient:
    BASE_URL = "https://api.marketaux.com/v1/news/all"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_news_for_ticker(self, ticker: str, limit: int = 5) -> List[Dict[str, Any]]:
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
        articles = sorted(articles, key=self._parse_date, reverse=True)

        # Only keep articles where the ticker appears in entities
        filtered: List[Dict[str, Any]] = []
        for a in articles:
            if self._matches_ticker(a, ticker):
                self._normalize_sentiment(a, ticker)
                filtered.append(a)
            if len(filtered) >= limit:
                break

        return filtered

    def fetch_historical_news_for_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> List[Dict[str, Any]]:
        """Fetch all available ticker news in a date range, oldest articles first."""
        articles: List[Dict[str, Any]] = []
        page = 1
        page_size = 100
        while True:
            params = {
                "symbols": ticker,
                "language": "en",
                "filter_entities": "true",
                "published_after": start_date,
                "published_before": end_date,
                "sort": "published_at",
                "limit": page_size,
                "page": page,
                "api_token": self.api_key,
            }
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("data", []) if isinstance(payload, dict) else []
            if not isinstance(batch, list) or not batch:
                break

            for article in batch:
                if self._matches_ticker(article, ticker):
                    self._normalize_sentiment(article, ticker)
                    articles.append(article)
            if len(batch) < page_size:
                break
            page += 1

        articles.sort(key=self._parse_date)
        return articles

    @staticmethod
    def _matches_ticker(article: Dict[str, Any], ticker: str) -> bool:
        entities = article.get("entities", [])
        symbols = [
            entity.get("symbol")
            for entity in entities
            if isinstance(entity, dict) and entity.get("symbol")
        ]
        return ticker.upper() in [symbol.upper() for symbol in symbols]

    @staticmethod
    def _parse_date(article: Dict[str, Any]) -> datetime:
        date_str = (
            article.get("published_at")
            or article.get("published_at_iso")
            or article.get("published_at_date")
        )
        if not date_str:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            parsed = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
        except (TypeError, ValueError):
            return datetime.min.replace(tzinfo=timezone.utc)

    @classmethod
    def _normalize_sentiment(cls, article: Dict[str, Any], ticker: str) -> None:
        score = None
        label = None
        matched_entity = next(
            (
                entity
                for entity in article.get("entities", []) or []
                if isinstance(entity, dict)
                and entity.get("symbol")
                and str(entity["symbol"]).upper() == ticker.upper()
            ),
            None,
        )

        if matched_entity:
            score = matched_entity.get("sentiment_score")
            if score is None:
                score = matched_entity.get("sentiment")
        else:
            raw = article.get("sentiment")
            if isinstance(raw, dict):
                score = raw.get("score")
                if score is None:
                    score = raw.get("sentiment_score")
                label = raw.get("label") or raw.get("sentiment")
            else:
                score = article.get("sentiment_score")
                label = article.get("sentiment_label")

        try:
            if score is not None:
                score = float(score)
        except (TypeError, ValueError):
            score = None

        if score is None:
            values = []
            for highlight in article.get("highlights", []) or []:
                if isinstance(highlight, dict) and highlight.get("sentiment") is not None:
                    try:
                        values.append(float(highlight["sentiment"]))
                    except (TypeError, ValueError):
                        continue
            if values:
                score = sum(values) / len(values)

        if label is None and isinstance(score, (int, float)):
            label = "positive" if score >= 0.55 else "negative" if score <= 0.45 else "neutral"

        article["sentiment_score"] = score
        article["sentiment_label"] = label
