#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/data/npz_sequence_tf.py
Keras Sequence untuk membaca dataset mentah (BTD folders + radar + t1/t2/t3)
berdasarkan manifest CSV hasil scripts/make_manifests.py.

Default output X: (B,H,W,T*C_base) agar kompatibel dengan temporal_stem_3d di src/modeling.py
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

TS_RE_DEFAULT = r"^\d{8}_\d{4}\.npz$"


def _load_npz_array(npz_path: Path, prefer_key: str, fallback_key: Optional[str] = None) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as d:
        if prefer_key in d:
            return d[prefer_key]
        if fallback_key and fallback_key in d:
            return d[fallback_key]
        for k in d.files:
            arr = d[k]
            if getattr(arr, "ndim", 0) >= 2:
                return arr
        raise KeyError(f"No suitable array in {npz_path.name}, keys={d.files}")


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def _list_timestamps(folder: Path, ts_re: re.Pattern) -> List[str]:
    ts = []
    for p in folder.iterdir():
        if p.is_file() and ts_re.match(p.name):
            ts.append(p.stem)
    ts.sort()
    return ts


def _group_by_day(timestamps: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for t in timestamps:
        out.setdefault(t[:8], []).append(t)
    for d in out:
        out[d].sort()
    return out


def _reshape_time_to_channels(x: np.ndarray) -> np.ndarray:
    # (T,H,W,C) -> (H,W,T*C)
    x = np.transpose(x, (1, 2, 0, 3))
    h, w, t, c = x.shape
    return x.reshape(h, w, t * c)


@dataclass
class NormStats:
    mean: np.ndarray  # (C_base,)
    std: np.ndarray   # (C_base,)

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_json(obj: dict) -> "NormStats":
        return NormStats(mean=np.array(obj["mean"], dtype=np.float32),
                         std=np.array(obj["std"], dtype=np.float32))


class NPZSequence(tf.keras.utils.Sequence):
    """
    mode:
      - "BTDRadar"  -> C_base=4 (3 BTD + radar)
      - "onlyRadar" -> C_base=1
      - "BTDOnly"   -> C_base=3
    """

    def __init__(
        self,
        base_dir: str,
        manifest_csv: str,
        mode: str,
        horizon_min: int,
        batch_size: int = 4,
        window_steps: int = 12,
        cadence_min: int = 10,
        shuffle: bool = True,
        seed: int = 42,
        ts_source_folder: str = "btd_15-13",
        btd_folders: Sequence[str] = ("btd_13-08", "btd_15-13", "btd_16-13"),
        radar_folder: str = "radar",
        target_folders: Sequence[str] = ("t1", "t2", "t3"),
        expected_hw: Optional[Tuple[int, int]] = None,
        flatten_time_channels: bool = True,
        norm_stats: Optional[NormStats] = None,
    ):
        self.base_dir = Path(base_dir)
        self.mode = str(mode)
        self.horizon_min = int(horizon_min)
        self.batch_size = int(batch_size)
        self.window_steps = int(window_steps)
        self.cadence_min = int(cadence_min)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.flatten_time_channels = bool(flatten_time_channels)
        self.norm_stats = norm_stats
        self.expected_hw = expected_hw

        self.btd_folders = list(btd_folders)
        self.radar_folder = radar_folder
        self.target_folders = list(target_folders)

        m = self.mode.lower()
        self.use_btd = ("btd" in m) and (m != "onlyradar")
        self.use_radar = ("radar" in m) and (m != "btdonly")
        if m == "onlyradar":
            self.use_btd, self.use_radar = False, True
        if m == "btdonly":
            self.use_btd, self.use_radar = True, False

        self.c_base = (3 if self.use_btd else 0) + (1 if self.use_radar else 0)
        if self.c_base <= 0:
            raise ValueError(f"Invalid mode: {mode}")

        # Lead steps
        if self.horizon_min % self.cadence_min != 0:
            raise ValueError("horizon_min harus habis dibagi cadence_min")
        self.lead_steps = self.horizon_min // self.cadence_min

        # Manifest
        df = pd.read_csv(manifest_csv)
        req = {"day", "base_index_in_day", "lead_min"}
        miss = req - set(df.columns)
        if miss:
            raise ValueError(f"Manifest missing columns: {miss}")
        df = df[df["lead_min"].astype(int) == self.horizon_min].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(f"Tidak ada baris manifest untuk lead_min={self.horizon_min}")
        self.df = df
        self.indices = list(range(len(df)))

        # Timestamp mapping per day
        ts_re = re.compile(TS_RE_DEFAULT)
        ts_src = self.base_dir / ts_source_folder
        all_ts = _list_timestamps(ts_src, ts_re)
        self.by_day = _group_by_day(all_ts)

        self.rng = np.random.default_rng(self.seed)
        if self.shuffle:
            self.rng.shuffle(self.indices)

        self._hw_inferred: Optional[Tuple[int, int]] = None

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _infer_hw(self) -> Tuple[int, int]:
        if self.expected_hw is not None:
            return self.expected_hw
        if self._hw_inferred is not None:
            return self._hw_inferred
        row = self.df.iloc[0]
        day = str(row["day"])
        i = int(row["base_index_in_day"])
        ts_list = self.by_day[day]
        tgt_ts = ts_list[i + self.lead_steps]
        arr = _load_npz_array(self.base_dir / self.target_folders[0] / f"{tgt_ts}.npz", "data")
        arr = _ensure_2d(arr)
        self._hw_inferred = (arr.shape[-2], arr.shape[-1])
        return self._hw_inferred

    def _get_seq(self, day: str, base_i: int) -> Tuple[List[str], str]:
        ts_list = self.by_day[day]
        start = base_i - (self.window_steps - 1)
        end = base_i + 1
        in_seq = ts_list[start:end]
        tgt = ts_list[base_i + self.lead_steps]
        return in_seq, tgt

    def _load_input_frame(self, ts: str) -> np.ndarray:
        chans = []
        if self.use_btd:
            for folder in self.btd_folders:
                p = self.base_dir / folder / f"{ts}.npz"
                a = _load_npz_array(p, "channel_data", "data")
                a = _ensure_2d(a).astype(np.float32, copy=False)
                chans.append(a[..., None])
        if self.use_radar:
            p = self.base_dir / self.radar_folder / f"{ts}.npz"
            a = _load_npz_array(p, "data")
            a = _ensure_2d(a).astype(np.float32, copy=False)
            chans.append(a[..., None])
        return np.concatenate(chans, axis=-1)

    def _load_target(self, ts: str) -> np.ndarray:
        ys = []
        for folder in self.target_folders:
            p = self.base_dir / folder / f"{ts}.npz"
            a = _load_npz_array(p, "data")
            a = _ensure_2d(a).astype(np.float32, copy=False)
            ys.append(a[..., None])
        return np.concatenate(ys, axis=-1)

    def _apply_norm(self, x: np.ndarray) -> np.ndarray:
        if self.norm_stats is None:
            return x
        mean = self.norm_stats.mean.reshape(1, 1, 1, -1)
        std = self.norm_stats.std.reshape(1, 1, 1, -1)
        std = np.where(std == 0, 1.0, std)
        return (x - mean) / std

    def __getitem__(self, idx: int):
        h, w = self._infer_hw()
        batch_ids = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        bsz = len(batch_ids)

        X = np.empty((bsz, self.window_steps, h, w, self.c_base), dtype=np.float32)
        Y = np.empty((bsz, h, w, 3), dtype=np.float32)

        for j, ridx in enumerate(batch_ids):
            row = self.df.iloc[ridx]
            day = str(row["day"])
            base_i = int(row["base_index_in_day"])
            in_seq, tgt_ts = self._get_seq(day, base_i)
            frames = [self._load_input_frame(t) for t in in_seq]
            x_t = np.stack(frames, axis=0)
            x_t = self._apply_norm(x_t)
            y = self._load_target(tgt_ts)
            X[j] = x_t
            Y[j] = y

        if self.flatten_time_channels:
            X_out = np.empty((bsz, h, w, self.window_steps * self.c_base), dtype=np.float32)
            for j in range(bsz):
                X_out[j] = _reshape_time_to_channels(X[j])
            return X_out, Y
        return X, Y


def compute_norm_stats(seq: NPZSequence, max_batches: Optional[int] = None, eps: float = 1e-6) -> NormStats:
    old_flat = seq.flatten_time_channels
    seq.flatten_time_channels = False
    c = seq.c_base
    sum_c = np.zeros((c,), dtype=np.float64)
    sumsq_c = np.zeros((c,), dtype=np.float64)
    count = 0
    nb = len(seq) if max_batches is None else min(len(seq), int(max_batches))
    for b in range(nb):
        X, _ = seq[b]
        X = X.astype(np.float64, copy=False)
        sum_c += X.sum(axis=(0, 1, 2, 3))
        sumsq_c += (X * X).sum(axis=(0, 1, 2, 3))
        count += np.prod(X.shape[:-1])
    mean = (sum_c / max(count, 1)).astype(np.float32)
    var = (sumsq_c / max(count, 1) - mean.astype(np.float64) ** 2).astype(np.float32)
    var = np.maximum(var, 0.0)
    std = np.sqrt(var + eps).astype(np.float32)
    seq.flatten_time_channels = old_flat
    return NormStats(mean=mean, std=std)


def estimate_alpha_from_targets(seq: NPZSequence, eps: float = 1e-6, normalize: bool = True, max_batches: Optional[int] = None) -> List[float]:
    pos = np.zeros((3,), dtype=np.float64)
    tot = 0
    nb = len(seq) if max_batches is None else min(len(seq), int(max_batches))
    for b in range(nb):
        _, Y = seq[b]
        Yb = Y.astype(np.float64, copy=False)
        pos += Yb.sum(axis=(0, 1, 2))
        tot += np.prod(Yb.shape[:-1])
    pos_rate = pos / max(tot, 1)
    alpha = 1.0 / (pos_rate + eps)
    if normalize:
        alpha = alpha / np.min(alpha)
    return [float(a) for a in alpha]
