"""Timestamped stdout + file progress for long local/GPU runs."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path


class RunProgress:
    """Log START/DONE lines with timestamps; survives silent multi-minute forwards."""

    def __init__(self, log_path: Path, *, section: str, total: int) -> None:
        self.log_path = log_path
        self.section = section
        self.total = total
        self.done = 0
        self._section_start = time.perf_counter()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.emit(f"BEGIN {section} runs={total}")

    def emit(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def start_run(self, description: str) -> _RunTimer:
        self.done += 1
        section_elapsed = time.perf_counter() - self._section_start
        self.emit(
            f"START {self.done}/{self.total} {description} "
            f"(section_elapsed={section_elapsed:.0f}s)"
        )
        return _RunTimer(self, description)


class _RunTimer:
    def __init__(self, progress: RunProgress, description: str) -> None:
        self._progress = progress
        self.description = description
        self._t0 = time.perf_counter()

    def finish(self, summary: str) -> None:
        elapsed = time.perf_counter() - self._t0
        self._progress.emit(
            f"DONE  {self._progress.done}/{self._progress.total} "
            f"{self.description} {summary} (run={elapsed:.0f}s)"
        )


def reset_jsonl(path: Path) -> None:
    """Truncate JSONL at section start so partial runs are not mixed with stale rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
