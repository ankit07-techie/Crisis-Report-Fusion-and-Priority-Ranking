"""Unit tests for P2 Variant Robustness (src/category.py and src/priority.py)."""

import pytest
from src.category import predict_category
from src.priority import predict_priority

def test_robustness_lower_punct():
    raw = "Trapped on rooftop of 2-story building due to sudden flash flood!"
    transformed = "trapped on rooftop of 2story building due to sudden flash flood"

    assert predict_category(transformed) == predict_category(raw)
    assert abs(predict_priority(transformed) - predict_priority(raw)) <= 0.5

def test_robustness_wrapper():
    raw = "Casualties with severe bleeding after highway crash."
    transformed = "UPDATE: CONFIRMED REPORT - Casualties with severe bleeding after highway crash."

    assert predict_category(transformed) == predict_category(raw)
    assert abs(predict_priority(transformed) - predict_priority(raw)) <= 0.5

def test_robustness_truncation():
    raw = "Toxic chemical gas odor leaking from industrial storage tank in sector 9."
    transformed = "Toxic chemical gas odor leaking from industrial storage"

    assert predict_category(transformed) == "EmergingThreats"
    assert predict_priority(transformed) >= 3.5

def test_robustness_char_swap():
    raw = "Collapse at parking structure, workers trapped under concrete."
    transformed = "Colapse at prking structure, wrkers trpped under concrete."

    assert predict_category(transformed) == "SearchAndRescue"
    assert predict_priority(transformed) >= 4.0
