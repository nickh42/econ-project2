from __future__ import annotations

import json
from typing import Any, Dict

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class LLMClient:
    """Thin Groq Chat Completions client."""

    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        if not api_key or not api_key.strip():
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to .streamlit/secrets.toml."
            )

        try:
            from groq import Groq
        except ImportError as error:
            raise ValueError(
                "The groq package is not installed. Run: pip install groq"
            ) from error

        self._client = Groq(api_key=api_key.strip())
        self.model = (model or self.DEFAULT_MODEL).strip() or self.DEFAULT_MODEL

    def complete(self, messages: list[dict[str, str]]) -> str:
        response: Any = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=700,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("The LLM returned an empty response.")
        return str(content).strip()


def portfolio_has_holdings(observed_data: Dict[str, Any]) -> bool:
    holdings = observed_data.get("holdings")
    if isinstance(holdings, list):
        return bool(holdings)
    if isinstance(holdings, dict):
        return bool(holdings)
    return False


def is_question_in_scope(user_question: str) -> bool:
    """Return True only for dashboard-relevant finance questions."""
    if not user_question or not user_question.strip():
        return False

    lowered = user_question.lower()
    excluded_terms = (
        "poem",
        "joke",
        "recipe",
        "travel",
        "weather",
        "movie",
        "song",
        "politics",
        "math",
        "code",
        "debug",
    )
    if any(term in lowered for term in excluded_terms):
        return False

    allowed_terms = (
        "stock",
        "ticker",
        "portfolio",
        "holding",
        "holdings",
        "news",
        "sec",
        "filing",
        "financial",
        "earnings",
        "balance sheet",
        "cash flow",
        "market cap",
        "price",
        "return",
        "volatility",
        "sector",
        "industry",
        "dividend",
        "watchlist",
        "trade education",
        "buy",
        "sell",
        "risk",
    )
    return any(term in lowered for term in allowed_terms)


def contains_prediction_request(user_question: str) -> bool:
    lowered = (user_question or "").lower()
    prediction_terms = (
        "will it go up",
        "will it go down",
        "predict",
        "forecast",
        "price target",
        "target price",
        "tomorrow",
        "next week",
        "next month",
        "future price",
        "go up",
        "go down",
        "outperform",
        "underperform",
    )
    return any(term in lowered for term in prediction_terms)


def build_ai_prompt(page_name: str, current_ticker: str | None, user_query: str, observed_data: Dict[str, Any]) -> str:
    """Build a strict prompt that limits the assistant to read-only finance answers."""
    observed_snapshot = json.dumps(observed_data, ensure_ascii=False, default=str)
    ticker_text = current_ticker or "No specific ticker selected"
    return (
        "You are a read-only finance assistant for a stock and portfolio dashboard. "
        "Use only the data shown on this page. Do not make up any additional facts.\n\n"
        f"Page: {page_name}\n"
        f"Current ticker: {ticker_text}\n"
        f"Observed data: {observed_snapshot}\n\n"
        "Rules:\n"
        "- Answer only questions about stocks, SEC data, market news, portfolio holdings, or trade education.\n"
        "- Read-only mode: you cannot place trades, submit orders, or trigger any action.\n"
        "- You cannot make predictions, forecasts, price targets, or tell the user whether a stock will go up or down.\n"
        "- Do not provide trade recommendations or investment advice.\n"
        "- If the question is unrelated to stock, SEC, news, or portfolio analysis, respond with: 'I can only help with stock, SEC, news, and portfolio questions. Please ask about the current ticker, filings, market news, or your portfolio.'\n"
        "- If the user asks about trade concepts, explain them educationally without recommending a specific buy or sell.\n"
        "- Use the actual values from the provided data and keep the explanation concise and actionable.\n\n"
        f"User question: {user_query}"
    )


def ask_groq_assistant(
    page_name: str,
    current_ticker: str | None,
    user_query: str,
    observed_data: Dict[str, Any],
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Send a read-only question to Groq and return the answer text."""
    if not api_key or not api_key.strip():
        raise ValueError("Add groq_api_key to .streamlit/secrets.toml to enable Ask AI.")

    if not is_question_in_scope(user_query):
        raise ValueError("I can only help with stock, SEC, news, and portfolio questions. Please ask about the current ticker, filings, market news, or your portfolio.")

    if contains_prediction_request(user_query):
        raise ValueError("I can explain the current information on this page, but I cannot make predictions or price forecasts.")

    prompt = build_ai_prompt(page_name, current_ticker, user_query, observed_data)
    client = LLMClient(api_key=api_key, model=model)

    try:
        return client.complete([{"role": "user", "content": prompt}])
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - network-specific branch
        raise ValueError(
            f"The AI assistant could not reach Groq with model '{model}'. Check the API key and network connection, then try again later."
        ) from exc
