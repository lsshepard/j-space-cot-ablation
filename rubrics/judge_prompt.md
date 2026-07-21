# Backtrack span judge — frozen prompt

You are labeling whether a short span from a model’s chain-of-thought contains an **effective backtrack**.

Read the rubric below, then the span. Respond with **only** a single JSON object matching the schema. No prose outside JSON.

## Rubric

An effective backtrack requires:
1. A discourse marker of reversal (e.g. wait, actually, let me reconsider, alternatively, that’s wrong), and
2. An actual change in approach after the marker (different plan/method/setup), not a marker-only loop or continuation of the same approach.

## Span

{{SPAN}}

## Schema

```json
{"is_backtrack": <bool>, "approach_before": "<string>", "approach_after": "<string>"}
```
