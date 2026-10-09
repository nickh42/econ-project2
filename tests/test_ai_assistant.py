from finance_dashboard.ai_assistant import build_ai_prompt, is_question_in_scope


def test_build_ai_prompt_includes_stock_context_and_guardrails():
    prompt = build_ai_prompt(
        page_name="Stock detail",
        current_ticker="AAPL",
        user_query="What does the recent SEC filing say about liquidity?",
        observed_data={
            "company_name": "Apple Inc.",
            "current_price": 210.5,
            "sector": "Technology",
            "news": [{"title": "Apple launches new chip", "source": "Reuters"}],
            "sec_financial_data": {"financials": {"2024": [1000.0]}},
        },
    )

    assert "AAPL" in prompt
    assert "Apple Inc." in prompt
    assert "Technology" in prompt
    assert "read-only" in prompt.lower()
    assert "cannot make predictions" in prompt.lower()
    assert "liquidity" in prompt.lower()


def test_is_question_in_scope_rejects_unrelated_queries():
    assert is_question_in_scope("Can you write me a poem about the ocean?") is False
    assert is_question_in_scope("Explain the current portfolio and recent SEC filings for AAPL") is True
