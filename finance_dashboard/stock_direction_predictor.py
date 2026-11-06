from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MIN_TRAINING_ROWS = 60
SENTIMENT_FEATURES = ("Daily_sentiment_mean", "News_article_count")


class StockDirectionPredictor:
    """Train a chronological Random Forest model for next-session direction."""

    @staticmethod
    def build_features(
        prices: pd.DataFrame,
        articles: Sequence[dict[str, Any]] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, bool]:
        if "Close" not in prices.columns:
            raise ValueError("Close prices are required to predict stock direction.")

        data = prices.copy()
        data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
        data = data.sort_index()
        data = data[~data.index.duplicated(keep="last")]
        close = pd.to_numeric(data["Close"], errors="coerce")
        if close.dropna().empty:
            raise ValueError("No valid close prices are available for this ticker.")

        daily_returns = close.pct_change()
        features = pd.DataFrame(index=data.index)
        features["Daily_returns"] = daily_returns
        for days in (5, 10, 20):
            features[f"{days}_day_return"] = close.pct_change(days)
            features[f"{days}_day_moving_average"] = close.rolling(days).mean()
            features[f"{days}_day_volatility"] = daily_returns.rolling(days).std()

        volume = (
            pd.to_numeric(data["Volume"], errors="coerce").fillna(0.0)
            if "Volume" in data.columns
            else pd.Series(0.0, index=data.index)
        )
        features["Volume_change"] = (
            volume.pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        features["5_day_average_volume"] = volume.rolling(5).mean()

        sentiment_mean, article_count, sentiment_available = StockDirectionPredictor._daily_sentiment(
            data.index,
            articles or [],
        )
        features[SENTIMENT_FEATURES[0]] = sentiment_mean
        features[SENTIMENT_FEATURES[1]] = article_count

        next_close = close.shift(-1)
        target = pd.Series(
            np.where(next_close.notna(), (next_close > close).astype(float), np.nan),
            index=data.index,
            name="target",
        )
        features = features.replace([np.inf, -np.inf], np.nan)
        return features, target, sentiment_available

    @staticmethod
    def _daily_sentiment(
        trading_dates: pd.DatetimeIndex,
        articles: Sequence[dict[str, Any]],
    ) -> tuple[pd.Series, pd.Series, bool]:
        cutoffs = pd.DatetimeIndex(
            [
                (day.tz_localize("America/New_York") + pd.Timedelta(hours=16)).tz_convert("UTC")
                for day in trading_dates
            ]
        )
        scores: list[list[float]] = [[] for _ in trading_dates]
        counts = np.zeros(len(trading_dates), dtype=float)

        for article in articles:
            published_at = (
                article.get("published_at")
                or article.get("published_at_iso")
                or article.get("published_at_date")
            )
            if not published_at:
                continue
            timestamp = pd.to_datetime(published_at, utc=True, errors="coerce")
            if pd.isna(timestamp):
                continue

            # Place each article on the first trading close at or after publication.
            row = int(cutoffs.searchsorted(timestamp, side="left"))
            if row >= len(trading_dates):
                continue
            counts[row] += 1
            try:
                score = float(article.get("sentiment_score"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(score):
                scores[row].append(score)

        daily_means = [
            float(np.mean(day_scores)) if day_scores else np.nan
            for day_scores in scores
        ]
        sentiment = pd.Series(daily_means, index=trading_dates, dtype=float).ffill().fillna(0.0)
        article_count = pd.Series(counts, index=trading_dates, dtype=float)
        return sentiment, article_count, bool(article_count.gt(0).any())

    @staticmethod
    def _new_model() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )

    def predict(
        self,
        prices: pd.DataFrame,
        articles: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        features, target, sentiment_available = self.build_features(prices, articles)
        labeled = features.join(target).dropna()
        if len(labeled) < MIN_TRAINING_ROWS:
            raise ValueError(
                f"At least {MIN_TRAINING_ROWS} usable trading days are required; "
                f"only {len(labeled)} are available."
            )

        split_at = int(len(labeled) * 0.8)
        if split_at < 2 or split_at >= len(labeled):
            raise ValueError("There is not enough historical data for a chronological validation split.")

        x = labeled.drop(columns="target")
        y = labeled["target"].astype(int)
        chronological_model = self._new_model()
        chronological_model.fit(x.iloc[:split_at], y.iloc[:split_at])
        holdout_accuracy = float(
            chronological_model.score(x.iloc[split_at:], y.iloc[split_at:])
        )

        # Refit on all known labels after the chronological holdout evaluation.
        final_model = self._new_model()
        final_model.fit(x, y)
        latest_features = features.dropna().iloc[[-1]]
        prediction = int(final_model.predict(latest_features)[0])
        classifier = final_model.named_steps["classifier"]
        class_probabilities = final_model.predict_proba(latest_features)[0]
        probability_by_class = dict(zip(classifier.classes_, class_probabilities))

        return {
            "direction": "UP" if prediction == 1 else "DOWN",
            "probability": float(probability_by_class[prediction]),
            "sentiment_available": sentiment_available,
            "training_observations": len(labeled),
            "validation_accuracy": holdout_accuracy,
            "validation_start": str(labeled.index[split_at].date()),
            "feature_date": str(features.dropna().index[-1].date()),
        }
