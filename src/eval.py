#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/eval.py
Evaluasi model pada test split berdasarkan manifest test_leadXX.csv.

Contoh (Kaggle):
python -m src.eval \
  --base_dir "/kaggle/input/your-dataset/NPZ" \
  --manifest_dir "/kaggle/input/your-dataset/manifests" \
  --weights "/kaggle/working/run_h60_BTDRadar/best.weights.h5" \
  --out_dir "/kaggle/working/run_h60_BTDRadar" \
  --mode BTDRadar --horizon 60 --hw 128 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.data.npz_sequence_tf import NPZSequence, NormStats
from src.modeling import build_model, compile_model
from src.metrics_tf import F1ScorePerChannel, ThreatScorePerChannel, BiasMetricPerChannel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--manifest_dir", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--mode", type=str, default="BTDRadar", choices=["BTDRadar", "onlyRadar", "BTDOnly"])
    ap.add_argument("--horizon", type=int, default=60, choices=[10, 30, 60])
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--window_steps", type=int, default=12)
    ap.add_argument("--cadence_min", type=int, default=10)
    ap.add_argument("--hw", type=int, nargs=2, default=None)
    ap.add_argument("--alpha_json", type=str, default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    man_dir = Path(args.manifest_dir)
    test_csv = man_dir / f"test_lead{args.horizon}.csv"
    if not test_csv.exists():
        raise FileNotFoundError(test_csv)

    expected_hw = tuple(args.hw) if args.hw is not None else None

    # Load norm if available
    norm_path = out_dir / "norm_stats.json"
    norm = None
    if norm_path.exists():
        norm = NormStats.from_json(json.loads(norm_path.read_text(encoding="utf-8")))

    test_seq = NPZSequence(
        base_dir=args.base_dir,
        manifest_csv=str(test_csv),
        mode=args.mode,
        horizon_min=args.horizon,
        batch_size=args.batch_size,
        window_steps=args.window_steps,
        cadence_min=args.cadence_min,
        shuffle=False,
        expected_hw=expected_hw,
        flatten_time_channels=True,
        norm_stats=norm,
    )

    # Alpha (optional) - for compile, but evaluation doesn't need loss to match training; still compile for metrics
    alpha = None
    if args.alpha_json:
        alpha = json.loads(Path(args.alpha_json).read_text(encoding="utf-8")).get("alpha")
    else:
        a_path = out_dir / "alpha.json"
        if a_path.exists():
            alpha = json.loads(a_path.read_text(encoding="utf-8")).get("alpha")

    H, W = test_seq._infer_hw()
    input_channels = args.window_steps * test_seq.c_base
    model = build_model(input_size=(H, W, input_channels), t_steps=args.window_steps, with_deep_supervision=True)

    # compile with alpha_override if available; else default internal
    model, _ = compile_model(
        model,
        y_train=tf.constant([0.0]),
        steps_per_epoch=1,
        total_epochs=1,
        alpha_override=alpha,
        extra_metrics=[
            F1ScorePerChannel(0, name="f1_T1"),
            F1ScorePerChannel(1, name="f1_T2"),
            F1ScorePerChannel(2, name="f1_T3"),
            ThreatScorePerChannel(0, name="ts_T1"),
            ThreatScorePerChannel(1, name="ts_T2"),
            ThreatScorePerChannel(2, name="ts_T3"),
            BiasMetricPerChannel(0, name="bias_T1"),
            BiasMetricPerChannel(1, name="bias_T2"),
            BiasMetricPerChannel(2, name="bias_T3"),
        ],
    )

    model.load_weights(args.weights)

    results = model.evaluate(test_seq, verbose=1, return_dict=True)
    (out_dir / f"eval_test_lead{args.horizon}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Also write CSV (one-row)
    df = pd.DataFrame([results])
    df.to_csv(out_dir / f"eval_test_lead{args.horizon}.csv", index=False)

    print("Saved:", out_dir / f"eval_test_lead{args.horizon}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
