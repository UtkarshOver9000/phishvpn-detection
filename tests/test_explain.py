"""
Unit tests for the pure helper functions in explain.py.

main() itself calls the OpenAI API and is intentionally left untested here —
exercising it would require network access / a live API key, which CI doesn't
have and shouldn't need for a unit suite.
"""

import json

from phishvpn.explain import _iter_candidates, _session_to_prompt


def test_session_to_prompt_embeds_row_and_risk_score():
    row = {"country": "US", "login_failures_24h": 3}
    prompt = _session_to_prompt(row, risk_score=0.8231)
    assert "security analyst" in prompt.lower()
    payload = json.loads(prompt.rsplit("\n\n", 1)[1])
    assert payload["session"] == row
    assert payload["risk_score"] == 0.8231


def test_iter_candidates_filters_by_threshold():
    import pandas as pd

    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    scores = [0.1, 0.9, 0.5, 0.71]
    candidates = list(_iter_candidates(df, scores, threshold=0.7))
    assert candidates == [(1, 0.9), (3, 0.71)]


def test_iter_candidates_empty_when_nothing_clears_threshold():
    import pandas as pd

    df = pd.DataFrame({"x": [1, 2]})
    candidates = list(_iter_candidates(df, [0.1, 0.2], threshold=0.9))
    assert candidates == []
