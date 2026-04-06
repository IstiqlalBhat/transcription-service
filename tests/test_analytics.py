import pytest


def test_record_and_get_stats():
    from gateway.analytics import AnalyticsTracker

    tracker = AnalyticsTracker()
    tracker.record("parakeet_tdt")
    tracker.record("parakeet_tdt")
    tracker.record("canary_qwen")

    stats = tracker.get_stats()
    assert stats["total_requests"] == 3
    assert stats["model_selections"]["parakeet_tdt"] == 2
    assert stats["model_selections"]["canary_qwen"] == 1
    assert stats["model_selection_pct"]["parakeet_tdt"] == pytest.approx(66.7, abs=0.1)
    assert stats["model_selection_pct"]["canary_qwen"] == pytest.approx(33.3, abs=0.1)


def test_empty_stats():
    from gateway.analytics import AnalyticsTracker

    tracker = AnalyticsTracker()
    stats = tracker.get_stats()
    assert stats["total_requests"] == 0
    assert stats["model_selections"] == {}
    assert stats["model_selection_pct"] == {}


def test_record_none_primary_model_ignored():
    from gateway.analytics import AnalyticsTracker

    tracker = AnalyticsTracker()
    tracker.record(None)
    tracker.record("parakeet_tdt")

    stats = tracker.get_stats()
    assert stats["total_requests"] == 1
    assert stats["model_selections"]["parakeet_tdt"] == 1
