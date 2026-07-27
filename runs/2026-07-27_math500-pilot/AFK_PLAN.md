# MATH500 AFK / migration plan (superseded ceiling)

**Authoritative instructions for the next GPU agent: see [README.md](./README.md).**

Historical notes:
- Started as A100 full run, n=15, band [27,31].
- OOM on eager long dual-KV → sdpa + expandable_segments.
- Plan was L1→L3→(L2|L5); ablated ceiling discussed at 16k then 10k.
- **Final cost decision: 8k global ceiling**, retro truncate+rescore, migrate to **A40**.
