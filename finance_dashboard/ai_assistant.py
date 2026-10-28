from __future__ import annotations

import json
from typing import Any, Dict

DEFAULT_MODEL = "openai/gpt-oss-120b"


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


def is_question_in_scope(
    user_question: str,
    *,
    page_name: str | None = None,
    current_ticker: str | None = None,
) -> bool:
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

    if page_name == "stock_detail" and current_ticker:
        return True

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


def _summarize_prompt_value(value: Any, *, max_items: int = 8, max_string_length: int = 220) -> Any:
    """Reduce large payloads to a compact, still-useful summary for LLM prompts."""
    if isinstance(value, dict):
        items = list(value.items())
        limited_items = items[:max_items]
        summarized: Dict[str, Any] = {}
        for key, item in limited_items:
            summarized[str(key)] = _summarize_prompt_value(
                item,
                max_items=max_items,
                max_string_length=max_string_length,
            )
        if len(items) > max_items:
            summarized["truncated"] = True
            summarized["additional fields omitted"] = len(items) - max_items
        return summarized

    if isinstance(value, (list, tuple)):
        items = list(value)
        limited = items[:max_items]
        summarized_items = [
            _summarize_prompt_value(
                item,
                max_items=max_items,
                max_string_length=max_string_length,
            )
            for item in limited
        ]
        if len(items) > max_items:
            summarized_items.append({"truncated": True, "additional items omitted": len(items) - max_items})
        return summarized_items

    if isinstance(value, str):
        if len(value) > max_string_length:
            return f"{value[: max_string_length - 3]}..."
        return value

    if isinstance(value, float):
        if abs(value) > 1_000_000_000:
            return round(value / 1_000_000_000, 3)
        return value

    if isinstance(value, (int, bool)) or value is None:
        return value

    text = str(value)
    if len(text) > max_string_length:
        return f"{text[: max_string_length - 3]}..."
    return text


def _collect_truncation_notes(value: Any, *, prefix: str = "") -> list[str]:
    """Collect readable truncation markers from nested prompt summaries."""
    notes: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, dict) and item.get("truncated") is True:
                notes.append(f"{path}: additional fields omitted")
            elif isinstance(item, list):
                for nested in item:
                    if isinstance(nested, dict) and nested.get("truncated") is True:
                        notes.append(f"{path}: additional fields omitted")
            notes.extend(_collect_truncation_notes(item, prefix=path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            if isinstance(item, dict) and item.get("truncated") is True:
                notes.append(f"{path}: additional fields omitted")
            notes.extend(_collect_truncation_notes(item, prefix=path))
    return notes


def _prioritize_query_matches(value: Any, user_query: str) -> Any:
    """Keep matching financial rows near the front before prompt-size limits apply."""
    ignored_terms = {
        "about",
        "can",
        "does",
        "for",
        "from",
        "give",
        "me",
        "tell",
        "that",
        "the",
        "this",
        "what",
        "with",
        "you",
    }
    query_terms = {
        term.strip(".,?!:;()[]{}").lower()
        for term in user_query.split()
    } - ignored_terms
    query_terms = {term for term in query_terms if len(term) > 2}

    if isinstance(value, dict):
        return {
            key: _prioritize_query_matches(item, user_query)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        items = list(value)
        if query_terms:
            matching = [
                item
                for item in items
                if any(
                    term in json.dumps(item, ensure_ascii=False, default=str).lower()
                    for term in query_terms
                )
            ]
            if matching:
                prioritized = matching + [item for item in items if item not in matching]
                items = prioritized[:8]
        return [
            _prioritize_query_matches(item, user_query)
            for item in items
        ]

    return value


def build_ai_prompt(page_name: str, current_ticker: str | None, user_query: str, observed_data: Dict[str, Any]) -> str:
    """Build a strict prompt that limits the assistant to read-only finance answers."""
    relevant_data = _prioritize_query_matches(observed_data, user_query)
    summarized_data = _summarize_prompt_value(relevant_data)
    truncation_notes = _collect_truncation_notes(summarized_data)
    if truncation_notes:
        summarized_data["truncation_summary"] = "; ".join(truncation_notes)
    observed_snapshot = json.dumps({"summary": summarized_data, "truncated": True}, ensure_ascii=False, default=str)
    ticker_text = current_ticker or "No specific ticker selected"
    return (
        "You are a read-only finance assistant for a stock and portfolio dashboard. "
        "Use only the data shown on this page. Do not make up any additional facts.\n\n"
        f"Page: {page_name}\n"
        f"Current ticker: {ticker_text}\n"
        f"Observed data summary: {observed_snapshot}\n\n"
        "Rules:\n"
        "- Answer only questions about stocks, SEC data, market news, portfolio holdings, or trade education.\n"
        "- Read-only mode: you cannot place trades, submit orders, or trigger any action.\n"
        "- If the observed data includes EDGAR filing links or snippets, treat those as the authoritative direct SEC source and use them for SEC questions.\n"
        "- You may explain the general meaning of financial terms using standard financial knowledge, but do not claim a company-specific value unless it appears in the observed data.\n"
        "- You cannot make predictions, forecasts, price targets, or tell the user whether a stock will go up or down.\n"
        "- Do not provide trade recommendations or investment advice.\n"
        "- If the question is unrelated to stock, SEC, news, or portfolio analysis, respond with: 'I can only help with stock, SEC, news, and portfolio questions. Please ask about the current ticker, filings, market news, or your portfolio.'\n"
        "- If the user asks about trade concepts, explain them educationally without recommending a specific buy or sell.\n"
        "- Use matching observed financial data when available. If it is unavailable, say so and distinguish a general definition from company-specific facts.\n\n"
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

    if not is_question_in_scope(
        user_query,
        page_name=page_name,
        current_ticker=current_ticker,
    ):
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
