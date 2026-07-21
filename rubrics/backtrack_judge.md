# Effective backtrack — judge rubric

**Freeze before inspecting result data.** Used identically for every candidate span.

## Definition

An **effective backtrack** is a discourse marker of reversal **followed by an actual change in approach**.

### Reversal markers (non-exhaustive)

Examples: “wait,” “actually,” “let me reconsider,” “alternatively,” “that’s wrong,” “on second thought,” “hold on,” “I made a mistake,” “scratch that,” “no,” when used to reject the prior approach.

### Approach change (required)

After the marker, the model must adopt a **different plan, method, equation setup, case split, or goal** than the one pursued immediately before the marker. Restating the same approach, repeating arithmetic, or looping markers without a new plan does **not** count.

### Non-examples (do not count)

- Marker-only loops: “wait, wait, wait” with no new approach.
- Emphasis / hedging without reversal: “actually, yes, that is correct.”
- Continuing the same derivation after a filler word.
- Switching notation while keeping the same strategy.

## Unit of judgment

Judge **one candidate span** at a time (marker ± local context), not a whole trace. Multiple effective backtracks in one trace are counted individually.

## Required structured output

```json
{
  "is_backtrack": true,
  "approach_before": "short paraphrase of the prior approach",
  "approach_after": "short paraphrase of the new approach"
}
```

Set `is_backtrack` to `true` only when both a reversal marker and a real approach change are present. If `is_backtrack` is `false`, still fill the approach fields when possible (use `""` if absent).
