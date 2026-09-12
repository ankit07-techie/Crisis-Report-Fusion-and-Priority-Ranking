"""Unit tests for deterministic variant generation (AI-03)."""

from src.variants import (
    compute_sha256_bytes,
    should_generate_variant,
    select_transform_type,
    transform_lowercase_no_punct,
    transform_prepend_append,
    transform_remove_final_tokens,
    transform_swap_alphanumeric,
    generate_development_variant
)


def test_sha256_bytes_determinism():
    """Verify hash bytes are completely deterministic for fixed message IDs."""
    h0, h1 = compute_sha256_bytes("MSG_000")
    assert h0 == 12
    assert h1 == 96
    assert h0 % 10 == 2
    assert h1 % 4 == 0

    h0_neg, h1_neg = compute_sha256_bytes("MSG_002")
    assert h0_neg == 58
    assert h0_neg % 10 == 8  # Not in {0..5}


def test_transform_0_lowercase_no_punct():
    """Transform 0: convert text to lowercase and remove ASCII punctuation."""
    raw = "Flash FLOOD on 4th Street! Help, please; hurry... #Emergency"
    transformed = transform_lowercase_no_punct(raw)
    assert transformed == "flash flood on 4th street help please hurry emergency"
    assert "!" not in transformed
    assert "," not in transformed
    assert ";" not in transformed
    assert "#" not in transformed


def test_transform_1_prepend_append():
    """Transform 1: prepend 'Update: ' and append ' Please verify.'."""
    raw = "Bridge collapsed on Route 9."
    transformed = transform_prepend_append(raw)
    assert transformed == "Update: Bridge collapsed on Route 9. Please verify."


def test_transform_2_remove_final_tokens():
    """Transform 2: remove final ceil(0.15 * n) tokens while retaining >= 1 token."""
    # 10 tokens: ceil(0.15 * 10) = 2 tokens removed -> 8 tokens remain
    raw_10 = "one two three four five six seven eight nine ten"
    t_10 = transform_remove_final_tokens(raw_10)
    assert t_10 == "one two three four five six seven eight"

    # 1 token: must retain at least 1 token
    assert transform_remove_final_tokens("Evacuate") == "Evacuate"


def test_transform_3_swap_alphanumeric():
    """Transform 3: swap character with next at position 40, 80... if both alphanumeric."""
    # Build string with known characters at 1-based pos 40 and 41 (indices 39 and 40)
    prefix = "A" * 39      # indices 0..38
    mid = "XY"             # index 39 is 'X', index 40 is 'Y'
    suffix = "B" * 50
    text = prefix + mid + suffix
    assert text[39] == "X"
    assert text[40] == "Y"

    transformed = transform_swap_alphanumeric(text)
    assert transformed[39] == "Y"
    assert transformed[40] == "X"

    # If position 40 or 41 is non-alphanumeric, no swap occurs
    text_non_alnum = prefix + "X " + suffix
    assert transform_swap_alphanumeric(text_non_alnum)[39:41] == "X "


def test_generate_development_variant_fixed_ids():
    """Test full variant generation pipeline on verified fixed IDs."""
    # MSG_000 -> Transform 0
    res_0 = generate_development_variant("MSG_000", "Hello, World! Need Food.")
    assert res_0.is_variant is True
    assert res_0.transform_type == 0
    assert res_0.variant_text == "hello world need food"
    assert res_0.variant_id == "MSG_000_var_0"

    # MSG_004 -> Transform 1
    res_1 = generate_development_variant("MSG_004", "Shelter needed urgently.")
    assert res_1.is_variant is True
    assert res_1.transform_type == 1
    assert res_1.variant_text == "Update: Shelter needed urgently. Please verify."

    # MSG_001 -> Transform 2
    res_2 = generate_development_variant("MSG_001", "one two three four five six seven eight nine ten")
    assert res_2.is_variant is True
    assert res_2.transform_type == 2
    assert res_2.variant_text == "one two three four five six seven eight"

    # MSG_005 -> Transform 3
    text_long = "A" * 39 + "XY" + "B" * 50
    res_3 = generate_development_variant("MSG_005", text_long)
    assert res_3.is_variant is True
    assert res_3.transform_type == 3
    assert res_3.variant_text[39:41] == "YX"

    # MSG_002 -> Not generated (h0 % 10 = 8)
    res_neg = generate_development_variant("MSG_002", "Some crisis report text.")
    assert res_neg.is_variant is False
    assert res_neg.variant_text is None
