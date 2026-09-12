"""Official Deterministic Variant Generator for AI-03 (Crisis Report Fusion).
Strictly implements the specification from the problem statement:
- h = SHA256("20260911:" + message_id)
- Variant created if: h[0] % 10 in {0, 1, 2, 3, 4, 5}
- Transformation selected by: h[1] % 4:
    0: convert text to lowercase and remove ASCII punctuation
    1: prepend 'Update: ' and append ' Please verify.'
    2: remove final ceil(0.15 * n) tokens while retaining at least one token
    3: at character positions 40, 80, 120, ..., swap character with next when both are alphanumeric
"""

import hashlib
import math
import string
from typing import Optional, NamedTuple


class VariantResult(NamedTuple):
    is_variant: bool
    variant_id: Optional[str]
    variant_text: Optional[str]
    transform_type: Optional[int]
    transform_name: Optional[str]


def compute_sha256_bytes(message_id: str) -> tuple[int, int]:
    """Compute the first two bytes of SHA256('20260911:' + message_id)."""
    raw_key = f"20260911:{message_id}".encode("utf-8")
    digest = hashlib.sha256(raw_key).digest()
    return digest[0], digest[1]


def should_generate_variant(message_id: str) -> bool:
    """Check if message_id qualifies for variant generation: h0 % 10 in {0, 1, 2, 3, 4, 5}."""
    h0, _ = compute_sha256_bytes(message_id)
    return (h0 % 10) in {0, 1, 2, 3, 4, 5}


def select_transform_type(message_id: str) -> int:
    """Select transformation type 0..3 using: h1 % 4."""
    _, h1 = compute_sha256_bytes(message_id)
    return h1 % 4


def transform_lowercase_no_punct(text: str) -> str:
    """Transformation 0: Convert text to lowercase and remove ASCII punctuation."""
    lowered = text.lower()
    return lowered.translate(str.maketrans("", "", string.punctuation))


def transform_prepend_append(text: str) -> str:
    """Transformation 1: Prepend 'Update: ' and append ' Please verify.'."""
    cleaned = text.strip()
    return f"Update: {cleaned} Please verify."


def transform_remove_final_tokens(text: str) -> str:
    """Transformation 2: Remove final ceil(0.15 * n) tokens while retaining >= 1 token."""
    tokens = text.split()
    n = len(tokens)
    if n <= 1:
        return text
    k = math.ceil(0.15 * n)
    # Retain at least one token
    k = min(k, n - 1)
    return " ".join(tokens[:-k])


def transform_swap_alphanumeric(text: str) -> str:
    """
    Transformation 3: At 1-based character positions 40, 80, 120, ... (indices 39, 79, 119),
    swap character with next character when both are alphanumeric.
    """
    chars = list(text)
    length = len(chars)
    pos = 40
    while pos < length:
        idx = pos - 1  # 0-based index for 1-based character position
        if idx + 1 < length:
            if chars[idx].isalnum() and chars[idx + 1].isalnum():
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        pos += 40
    return "".join(chars)


TRANSFORM_NAMES = {
    0: "lowercase_remove_punctuation",
    1: "prepend_update_append_verify",
    2: "remove_final_15pct_tokens",
    3: "swap_alphanumeric_at_40_intervals"
}

TRANSFORM_FUNCS = {
    0: transform_lowercase_no_punct,
    1: transform_prepend_append,
    2: transform_remove_final_tokens,
    3: transform_swap_alphanumeric
}


def generate_development_variant(message_id: str, text: str) -> VariantResult:
    """
    Generate deterministic development variant for a crisis report.
    Depends solely on message_id and text. No labels or metadata used.
    """
    h0, h1 = compute_sha256_bytes(message_id)
    if (h0 % 10) not in {0, 1, 2, 3, 4, 5}:
        return VariantResult(
            is_variant=False,
            variant_id=None,
            variant_text=None,
            transform_type=None,
            transform_name=None
        )

    t_type = h1 % 4
    func = TRANSFORM_FUNCS[t_type]
    v_text = func(text)
    v_id = f"{message_id}_var_{t_type}"

    return VariantResult(
        is_variant=True,
        variant_id=v_id,
        variant_text=v_text,
        transform_type=t_type,
        transform_name=TRANSFORM_NAMES[t_type]
    )
