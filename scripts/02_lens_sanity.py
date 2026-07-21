#!/usr/bin/env python3
"""Load pre-fitted lens; calc intermediate sanity gate (§7.2 / §3.A.2)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jspace.calibrate import lens_sanity_calc
from jspace.config import DEV_MODEL, MAIN_MODEL, load_settings
from jspace.load import load_model_and_lens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="exit 1 if intermediates missing")
    args = parser.parse_args()

    settings = load_settings()
    loaded = load_model_and_lens(settings)
    print("lens_meta:", json.dumps(loaded.lens_meta, indent=2))

    # Try to surface CREDIT / fit-scale hints from hub side files if cached.
    try:
        from huggingface_hub import hf_hub_download

        for name in ("CREDIT.md", "README.md", "qwen3-4b/CREDIT.md"):
            try:
                path = hf_hub_download(
                    settings.lens_repo,
                    filename=name,
                    revision=settings.lens_revision,
                )
                print(f"--- {name} (first 40 lines) ---")
                lines = Path(path).read_text(encoding="utf-8").splitlines()[:40]
                print("\n".join(lines))
            except Exception:
                continue
    except Exception as exc:
        print(f"(metadata fetch skipped: {exc})")

    result = lens_sanity_calc(loaded)
    out = settings.results_dir / "calibration" / "lens_sanity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("any_hit", "skipped", "skip_reason") if k in result}, indent=2))
    print(f"wrote {out}")

    is_main = settings.model_name == MAIN_MODEL
    is_dev = settings.model_name == DEV_MODEL or "0.6B" in settings.model_name or "1.7B" in settings.model_name
    if result.get("skipped"):
        print(f"SOFT WARN: {result.get('skip_reason')}")
        return
    if not result["all_intermediates_seen"]:
        msg = "calc intermediates not all seen in J-lens top-10"
        if args.strict or is_main:
            raise SystemExit(f"GATE FAIL: {msg}")
        if is_dev:
            print(f"SOFT WARN (expected on small dev model): {msg}")
        else:
            print(f"WARN: {msg}")


if __name__ == "__main__":
    main()
