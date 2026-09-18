"""Verify the dashboard renders real HTML with SVG icons and NO emoji."""
import re
import tempfile

import pytest

import dashboard

EMOJI = re.compile(
    r"[\U0001F300-\U0001FAFF\u2600-\u27BF\u2b00-\u2bff\u2190-\u21FF"
    r"\u2300-\u23FF]")


def test_dashboard_renders_html():
    html = dashboard.render()
    assert html.startswith("<!doctype html>")
    assert "</html>" in html


def test_dashboard_uses_svg_not_emoji():
    html = dashboard.render()
    assert "<svg" in html, "icons must be inline SVG"
    assert EMOJI.findall(html) == [], "no emoji/icons from unicode ranges"


def test_dashboard_shows_data_source():
    html = dashboard.render()
    assert "Source:" in html or "data source" in html.lower()
