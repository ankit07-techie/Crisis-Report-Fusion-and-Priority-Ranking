"""Unit tests for P1 text preprocessing."""

from src.preprocessing import preprocess_crisis_text


def test_preprocess_url_removal():
    raw = "Bridge collapsed on Route 9 https://t.co/xyz123 traffic halted."
    cleaned = preprocess_crisis_text(raw)
    assert "https://" not in cleaned
    assert "t.co" not in cleaned
    assert "Bridge collapsed on Route 9 traffic halted." == cleaned


def test_preprocess_mention_removal():
    raw = "@RedCross @GovOffice immediate boat rescue needed on 4th street."
    cleaned = preprocess_crisis_text(raw)
    assert "@RedCross" not in cleaned
    assert "@GovOffice" not in cleaned
    assert "immediate boat rescue needed on 4th street." == cleaned


def test_preprocess_hashtag_preservation():
    raw = "#FlashFlood rescue team deployed to #Sector7 #emergency"
    cleaned = preprocess_crisis_text(raw)
    assert "#" not in cleaned
    assert "FlashFlood" in cleaned
    assert "Sector7" in cleaned
    assert "emergency" in cleaned


def test_preprocess_html_entities():
    raw = "Heavy rains &amp; strong winds in downtown."
    cleaned = preprocess_crisis_text(raw)
    assert "&amp;" not in cleaned
    assert "Heavy rains & strong winds in downtown." == cleaned
