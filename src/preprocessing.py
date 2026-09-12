"""P1 Text Preprocessing Module for Crisis Reports.
Performs clean, causal text normalization without metadata leakage.
"""

import re
import html

# Regex patterns for social media crisis text
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")
MENTION_REGEX = re.compile(r"@\w+")
WHITESPACE_REGEX = re.compile(r"\s+")
# Strip leading hash from hashtags, leaving the word token intact
HASHTAG_REGEX = re.compile(r"#(\w+)")


def preprocess_crisis_text(text: str, remove_urls: bool = True, clean_mentions: bool = True) -> str:
    """
    Clean and normalize raw crisis text:
    - Decodes HTML entities (&amp; -> &)
    - Strips URLs (which are transient or shortened t.co links)
    - Replaces @mentions with generic token or removes them
    - Preserves hashtag words (#rescue -> rescue)
    - Collapses multiple whitespaces and trims
    """
    if not text:
        return ""

    # Unescape HTML entities
    cleaned = html.unescape(text)

    # Remove URLs
    if remove_urls:
        cleaned = URL_REGEX.sub(" ", cleaned)

    # Strip or clean user mentions
    if clean_mentions:
        cleaned = MENTION_REGEX.sub(" ", cleaned)

    # Convert hashtags to plain words
    cleaned = HASHTAG_REGEX.sub(r"\1", cleaned)

    # Normalize whitespace
    cleaned = WHITESPACE_REGEX.sub(" ", cleaned).strip()

    return cleaned
