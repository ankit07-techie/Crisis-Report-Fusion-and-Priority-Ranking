"""Unit tests for P2 Operational Priority Prediction (src/priority.py)."""

import pytest
from src.priority import predict_priority

def test_predict_priority_bounds():
    p1 = predict_priority("Trapped on rooftop, drowning emergency life threatening!")
    p2 = predict_priority("Where can we get updates on power restoration?")

    assert isinstance(p1, float)
    assert isinstance(p2, float)

    assert 1.0 <= p1 <= 5.0
    assert 1.0 <= p2 <= 5.0

    assert p1 > p2

def test_predict_priority_relative_ranking():
    critical = predict_priority("Explosion at gas plant, 5 people trapped in rubble, severe bleeding!")
    moderate = predict_priority("Need transportation buses to evacuate elderly citizens from flood threat zone.")
    low = predict_priority("Where can we get updates on power restoration?")

    assert critical >= moderate
    assert moderate >= low

def test_predict_priority_edge_cases():
    assert predict_priority("") == 1.0
    assert predict_priority("    ") == 1.0
