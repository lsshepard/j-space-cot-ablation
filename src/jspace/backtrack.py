"""Regex candidate spans and length-normalized backtrack counts (§4.4)."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Reversal discourse markers (§4.4).
_MARKER_RE = re.compile(
    r"(?i)\b("
    r"wait|actually|alternatively|however|"
    r"let me reconsider|on second thought|hold on|"
    r"that'?s wrong|i made a mistake|scratch that|"
    r"let me try|try a different|different approach"
    r")\b"
)


@dataclass(frozen=True)
class BacktrackSpan:
    span_id: str
    trace_id: str
    start: int
    end: int
    marker: str
    text: str


def find_candidate_spans(
    trace: str,
    *,
    trace_id: str = "trace",
    window_chars: int = 280,
) -> list[BacktrackSpan]:
    """Emit marker-centered spans for judge / regex headline counting."""
    spans: list[BacktrackSpan] = []
    for i, match in enumerate(_MARKER_RE.finditer(trace)):
        start = max(0, match.start() - window_chars // 3)
        end = min(len(trace), match.end() + window_chars)
        spans.append(
            BacktrackSpan(
                span_id=f"{trace_id}-span-{i}",
                trace_id=trace_id,
                start=start,
                end=end,
                marker=match.group(0),
                text=trace[start:end],
            )
        )
    return spans


def regex_backtrack_count(trace: str) -> int:
    """
    Headline regex count: markers that are not immediately repeated marker loops.

    Degenerate 'wait, wait' chains count as one candidate region, not N.
    """
    spans = find_candidate_spans(trace, trace_id="tmp")
    if not spans:
        return 0
    # Merge overlapping spans.
    merged = 0
    current_end = -1
    for span in sorted(spans, key=lambda s: s.start):
        if span.start <= current_end:
            current_end = max(current_end, span.end)
            continue
        merged += 1
        current_end = span.end
    return merged


def normalized_backtrack_rate(count: int, trace_length_tokens: int) -> float:
    if trace_length_tokens <= 0:
        return 0.0
    return float(count) / float(trace_length_tokens)
