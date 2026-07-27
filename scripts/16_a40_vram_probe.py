#!/usr/bin/env python3
"""A40 smoke probe: dual-KV ablated CoT under the 8k ceiling.

Reports peak CUDA memory. Exit 0 if the run finishes without OOM.
Use before resuming the full MATH500 grid on a 48GB card.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8000, help="max_new_tokens")
    parser.add_argument(
        "--problem-index",
        type=int,
        default=0,
        help="index into MATH500 L1 sample (default first)",
    )
    args = parser.parse_args()

    os.environ.setdefault("JSPACE_MODEL", "Qwen/Qwen3-4B")
    os.environ.setdefault("JSPACE_DEVICE", "cuda")
    os.environ.setdefault("JSPACE_DTYPE", "bfloat16")
    os.environ.setdefault("JSPACE_ATTN_IMPLEMENTATION", "sdpa")
    os.environ.setdefault("JSPACE_TOKEN_BUDGET_CEILING", "8000")
    os.environ.setdefault("JSPACE_ABLATED_TOKEN_BUDGET_MULTIPLIER", "6")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    from jspace.ablation import AblationConfig
    from jspace.config import load_settings
    from jspace.data import load_math500
    from jspace.load import load_model_and_lens
    from jspace.run_one import run_problem

    settings = load_settings().with_overrides(
        early_stop_on_answer=True,
        token_budget_ceiling=8000,
        max_new_tokens_override=args.tokens,
    )
    print(
        f"probe tokens={args.tokens} attn={settings.attn_implementation} "
        f"device={settings.device or 'auto'}"
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for A40 probe")

    torch.cuda.reset_peak_memory_stats()
    loaded = load_model_and_lens(settings)
    problems = load_math500(level=1, limit=max(args.problem_index + 1, 1))
    problem = problems[args.problem_index]
    print(f"problem {problem.problem_id}")

    abl = AblationConfig(
        kind="jspace",
        band_start=27,
        band_end=31,
        k=10,
        seed=0,
        ablate_prompt_tokens=True,
    )
    rec = run_problem(
        loaded,
        problem,
        settings=settings,
        enable_thinking=True,
        ablation=abl,
        seed=0,
        max_new_tokens=args.tokens,
    )
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
    print(
        f"OK condition={rec.condition} len={rec.trace_length_tokens} "
        f"cap={rec.hit_token_cap} early={rec.extra.get('early_stopped')} "
        f"correct={rec.correct}"
    )
    print(f"peak_allocated_gb={peak_gb:.2f} peak_reserved_gb={reserved_gb:.2f}")
    if peak_gb > 44:
        print("WARNING: peak >44GB — A40 48GB has little headroom; watch fragmentation")


if __name__ == "__main__":
    main()
