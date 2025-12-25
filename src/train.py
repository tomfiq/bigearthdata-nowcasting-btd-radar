#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/train.py
CLI training untuk Kaggle/local: membaca manifest CSV + dataset mentah NPZ, lalu melatih model.

Contoh (Kaggle):
python -m src.train \
  --base_dir "/kaggle/input/your-dataset/NPZ" \
  --manifest_dir "/kaggle/input/your-dataset/manifests" \
  --out_dir "/kaggle/working/run_h60_BTDRadar" \
  --mode BTDRadar --horizon 60 --epochs 80 --batch_size 4 --hw 128 128

Catatan:
- Pastikan folder base_dir berisi: btd_13-08, btd_15-13, btd_16-13, radar, t1, t2, t3.
- Manifest: train_leadXX.csv & val_leadXX.csv harus tersedia.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import tensorflow as tf

from src.data.npz_sequence_tf import NPZSequence, compute_norm_stats, estimate_alpha_from_targets, NormStats
from src.modeling import build_model, compile_model
from src.metrics_tf import F1ScorePerChannel, ThreatScorePerChannel, BiasMetricPerChannel, LRSnapshot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--manifest_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--mode", type=str, default="BTDRadar", choices=["BTDRadar", "onlyRadar", "BTDOnly"])
    ap.add_argument("--horizon", type=int, default=60, choices=[10, 30, 60])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--window_steps", type=int, default=12)
    ap.add_argument("--cadence_min", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hw", type=int, nargs=2, default=None, help="H W, mis. 128 128")
    ap.add_argument("--base_lr", type=float, default=6e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--clip_norm", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Repro seed
    tf.keras.utils.set_random_seed(args.seed)

    man_dir = Path(args.manifest_dir)
    train_csv = man_dir / f"train_lead{args.horizon}.csv"
    val_csv = man_dir / f"val_lead{args.horizon}.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Missing manifest: {train_csv} or {val_csv}")

    expected_hw = tuple(args.hw) if args.hw is not None else None

    train_seq = NPZSequence(
        base_dir=args.base_dir,
        manifest_csv=str(train_csv),
        mode=args.mode,
        horizon_min=args.horizon,
        batch_size=args.batch_size,
        window_steps=args.window_steps,
        cadence_min=args.cadence_min,
        shuffle=True,
        seed=args.seed,
        expected_hw=expected_hw,
        flatten_time_channels=True,
        norm_stats=None,
    )
    val_seq = NPZSequence(
        base_dir=args.base_dir,
        manifest_csv=str(val_csv),
        mode=args.mode,
        horizon_min=args.horizon,
        batch_size=args.batch_size,
        window_steps=args.window_steps,
        cadence_min=args.cadence_min,
        shuffle=False,
        seed=args.seed,
        expected_hw=expected_hw,
        flatten_time_channels=True,
        norm_stats=None,
    )

    # Norm stats (save/load)
    norm_path = out_dir / "norm_stats.json"
    if norm_path.exists():
        norm = NormStats.from_json(json.loads(norm_path.read_text(encoding="utf-8")))
    else:
        norm = compute_norm_stats(train_seq, max_batches=None)
        norm_path.write_text(json.dumps(norm.to_json(), indent=2), encoding="utf-8")

    train_seq.norm_stats = norm
    val_seq.norm_stats = norm

    # Alpha (class-balance)
    alpha = estimate_alpha_from_targets(train_seq, max_batches=None)
    (out_dir / "alpha.json").write_text(json.dumps({"alpha": alpha}, indent=2), encoding="utf-8")

    # Build + compile model
    H, W = train_seq._infer_hw()
    input_channels = args.window_steps * train_seq.c_base
    model = build_model(input_size=(H, W, input_channels), t_steps=args.window_steps, with_deep_supervision=True)
    model, _ = compile_model(
        model,
        y_train=tf.constant([0.0]),  # dummy; alpha_override dipakai
        steps_per_epoch=len(train_seq),
        total_epochs=args.epochs,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        clip_norm=args.clip_norm,
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

    # Callbacks
    ckpt_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "best.weights.h5"),
        monitor="val_main_pr_auc_T3",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_main_pr_auc_T3",
        mode="max",
        patience=args.patience,
        restore_best_weights=True,
        verbose=1,
    )
    csvlog = tf.keras.callbacks.CSVLogger(str(out_dir / "train_log.csv"))
    lr_snap = LRSnapshot()
    reduce = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_main_pr_auc_T3",
        mode="max",
        factor=0.5,
        patience=max(3, args.patience // 4),
        min_lr=1e-6,
        verbose=1,
    )

    # Save config
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=[lr_snap, csvlog, ckpt_best, early, reduce],
        verbose=1,
        workers=2,
        use_multiprocessing=True,
    )

    # Save final
    model.save_weights(str(out_dir / "last.weights.h5"))
    (out_dir / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    print(f"Done. Outputs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
