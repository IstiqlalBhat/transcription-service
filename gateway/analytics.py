from collections import Counter


class AnalyticsTracker:
    """In-memory tracker for which model the judge selects most often."""

    def __init__(self):
        self._counts: Counter = Counter()

    def record(self, primary_model: str | None) -> None:
        """Record a judge decision. Ignores None."""
        if primary_model is not None:
            self._counts[primary_model] += 1

    def get_stats(self) -> dict:
        """Return current analytics."""
        total = sum(self._counts.values())
        selections = dict(self._counts)

        if total == 0:
            return {
                "total_requests": 0,
                "model_selections": {},
                "model_selection_pct": {},
            }

        pct = {
            model: round(count / total * 100, 1)
            for model, count in self._counts.items()
        }

        return {
            "total_requests": total,
            "model_selections": selections,
            "model_selection_pct": pct,
        }


# Singleton instance used by the gateway
tracker = AnalyticsTracker()
