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
    layer_next_token_match_rates,
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
    args = parser.parse_args()

    settings = load_settings()
    band_start = args.band_start if args.band_start is not None else settings.band_start
    band_end = args.band_end if args.band_end is not None else settings.band_end

    loaded = load_model_and_lens(settings)
    problems = load_gsm8k(limit=args.n_prompts)
    prompts = [p.prompt for p in problems]

    rates = layer_next_token_match_rates(loaded, prompts)
    band = select_band(
        rates,
        loaded.n_layers,
        override_start=band_start,
        override_end=band_end,
    )
    print(
        f"band=[{band.band_start}, {band.band_end}] "
        f"auto={band.auto_selected} label={band.strength_label}"
    )

    out_dir = settings.results_dir / "calibration"
    plot_path = out_dir / "band_diagnostic.png"
    plot_band_diagnostic(rates, band, plot_path)
    print(f"wrote {plot_path}")

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
