#!/usr/bin/env python3
"""Band diagnostic + auto-select + number-token loading (§3.A.3–A.6)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.calibrate import (
    DEFAULT_BAND_SELECT_TOPK,
    DEFAULT_DIAGNOSTIC_TOPKS,
    layer_next_token_match_rates_by_topk,
    number_token_loading,
    plot_band_diagnostic,
    save_band_json,
    select_band,
)
from jspace.config import load_settings
from jspace.data import load_gsm8k
from jspace.load import load_model_and_lens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--band-start", type=int, default=None)
    parser.add_argument("--band-end", type=int, default=None)
    parser.add_argument("--n-prompts", type=int, default=8)
    parser.add_argument(
        "--select-topk",
        type=int,
        default=DEFAULT_BAND_SELECT_TOPK,
        help="lens readout top-k curve used for auto band selection",
    )
    parser.add_argument(
        "--diagnostic-topks",
        default=",".join(str(k) for k in DEFAULT_DIAGNOSTIC_TOPKS),
        help="comma list of lens readout top-k values to sweep (one shared forward)",
    )
    args = parser.parse_args()

    settings = load_settings()
    band_start = args.band_start if args.band_start is not None else settings.band_start
    band_end = args.band_end if args.band_end is not None else settings.band_end
    topks = tuple(int(x) for x in args.diagnostic_topks.split(",") if x.strip())
    if args.select_topk not in topks:
        topks = tuple(sorted(set(topks) | {args.select_topk}))

    loaded = load_model_and_lens(settings)
    problems = load_gsm8k(limit=args.n_prompts)
    prompts = [p.prompt for p in problems]

    rates_by_k = layer_next_token_match_rates_by_topk(loaded, prompts, topks=topks)
    select_rates = rates_by_k[args.select_topk]
    band = select_band(
        select_rates,
        loaded.n_layers,
        override_start=band_start,
        override_end=band_end,
        match_rates_by_topk=rates_by_k,
        select_topk=args.select_topk,
    )
    print(
        f"band=[{band.band_start}, {band.band_end}] "
        f"auto={band.auto_selected} select_topk={band.select_topk} "
        f"label={band.strength_label}"
    )
    for k in sorted(rates_by_k):
        mid = rates_by_k[k][band.band_start : band.band_end + 1]
        mean_mid = sum(mid) / max(len(mid), 1)
        print(f"  top-{k}: mean_in_band={mean_mid:.3f}")

    out_dir = settings.results_dir / "calibration"
    plot_path = out_dir / "band_diagnostic.png"
    plot_band_diagnostic(rates_by_k, band, plot_path)
    print(f"wrote {plot_path}")
    print("Review the plot before continuing to 04 (override with --band-start/--band-end).")

    loading = number_token_loading(
        loaded,
        prompts,
        band_start=band.band_start,
        band_end=band.band_end,
    )
    print(f"number_token_mean_cosine={loading['mean_cosine']:.4f}")

    save_band_json(
        out_dir / "band.json",
        band,
        extra={"number_token_loading": loading, "model_name": loaded.model_name},
    )
    (out_dir / "loading.json").write_text(
        json.dumps(loading, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir / 'band.json'}")


if __name__ == "__main__":
    main()
