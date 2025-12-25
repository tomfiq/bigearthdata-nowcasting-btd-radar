#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_npz.py
Validasi dataset NPZ per-timestamp (BTD + radar + T1/T2/T3).

Cek:
- jumlah file per folder (default 8640),
- keselarasan timestamp antar folder (YYYYMMDD_HHMM.npz),
- key NPZ:
  - BTD: prefer 'channel_data', fallback 'data'
  - radar & t1/t2/t3: 'data'
- shape HxW (opsional),
- NaN/Inf,
- target binary (t1/t2/t3) {0,1}.

Contoh:
python scripts/validate_npz.py --base_dir "D:/NPZ" --expected_count 8640 --expected_hw 128 128 \
  --split_json "D:/split_days_fixed.json" --report_json "D:/NPZ/validation_report.json"
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

TS_RE_DEFAULT = r"^\d{8}_\d{4}\.npz$"


@dataclass
class FolderPolicy:
    name: str
    prefer_key: str
    fallback_key: Optional[str] = None
    expect_binary: bool = False
    allow_ndim: Tuple[int, ...] = (2,)


def load_npz_flexible(fp: Path, prefer_key: str, fallback_key: Optional[str]) -> np.ndarray:
    with np.load(fp, allow_pickle=False) as data:
        if prefer_key in data:
            return data[prefer_key]
        if fallback_key and fallback_key in data:
            return data[fallback_key]
        for k in data.files:
            arr = data[k]
            if hasattr(arr, "ndim") and arr.ndim >= 2:
                return arr
        raise KeyError(f"Key tidak ditemukan di {fp}. Keys: {data.files}")


def list_timestamps(folder: Path, ts_re: re.Pattern) -> List[str]:
    out: List[str] = []
    for p in folder.iterdir():
        if p.is_file() and ts_re.match(p.name):
            out.append(p.stem)
    out.sort()
    return out


def ts_day(ts: str) -> str:
    return ts[:8]


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def check_binary(arr: np.ndarray) -> bool:
    if arr.dtype == np.bool_:
        return True
    arr = ensure_2d(arr)
    uniq = np.unique(arr)
    if uniq.size > 6:
        return False
    return set(map(float, uniq.tolist())).issubset({0.0, 1.0})


def finite_stats(arr: np.ndarray) -> Dict[str, float]:
    arrf = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arrf)
    if not finite.any():
        return {"finite_frac": 0.0, "min": math.nan, "max": math.nan, "mean": math.nan}
    vals = arrf[finite]
    return {"finite_frac": float(finite.mean()), "min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean())}


def validate_split_json(split_path: Path, all_days: Set[str]) -> Dict[str, object]:
    obj = json.loads(split_path.read_text(encoding="utf-8"))
    train = set(obj.get("train_days", []))
    val = set(obj.get("val_days", []))
    test = set(obj.get("test_days", []))
    return {
        "split_json": str(split_path),
        "policy": obj.get("policy"),
        "seed": obj.get("seed"),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "disjoint_ok": train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test),
        "missing_days_in_data": {
            "train": sorted(list(train - all_days)),
            "val": sorted(list(val - all_days)),
            "test": sorted(list(test - all_days)),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--folders", type=str, nargs="*", default=["btd_13-08", "btd_15-13", "btd_16-13", "t1", "t2", "t3"])
    ap.add_argument("--include_radar", action="store_true")
    ap.add_argument("--expected_count", type=int, default=8640)
    ap.add_argument("--ts_regex", type=str, default=TS_RE_DEFAULT)
    ap.add_argument("--sample_per_folder", type=int, default=50)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--expected_hw", type=int, nargs=2, default=None)
    ap.add_argument("--split_json", type=str, default="")
    ap.add_argument("--report_json", type=str, default="")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(base_dir)

    folders = list(args.folders)
    if args.include_radar and "radar" not in folders:
        folders.append("radar")

    ts_re = re.compile(args.ts_regex)

    policies: Dict[str, FolderPolicy] = {}
    for f in folders:
        if f.startswith("btd"):
            policies[f] = FolderPolicy(f, "channel_data", "data", expect_binary=False)
        elif f in ("t1", "t2", "t3"):
            policies[f] = FolderPolicy(f, "data", None, expect_binary=True)
        elif f == "radar":
            policies[f] = FolderPolicy(f, "data", None, expect_binary=False)
        else:
            policies[f] = FolderPolicy(f, "data", None, expect_binary=False)

    ts_by_folder: Dict[str, List[str]] = {}
    for f in folders:
        p = base_dir / f
        if not p.exists():
            raise FileNotFoundError(p)
        ts_by_folder[f] = list_timestamps(p, ts_re)

    count_report = {f: {"count": len(ts_by_folder[f]), "expected": args.expected_count, "ok": len(ts_by_folder[f]) == args.expected_count} for f in folders}

    sets = {f: set(ts_by_folder[f]) for f in folders}
    common = set.intersection(*sets.values())
    union = set.union(*sets.values())
    missing = {f: sorted(list(union - sets[f])) for f in folders}

    rng = random.Random(12345)
    deep_stats: Dict[str, Dict[str, object]] = {}

    for f in folders:
        pol = policies[f]
        folder_path = base_dir / f
        candidates = ts_by_folder[f]
        if not candidates:
            deep_stats[f] = {"checked": 0, "errors": ["No timestamp files found"], "examples": []}
            continue
        sample_ts = candidates if args.full else rng.sample(candidates, min(args.sample_per_folder, len(candidates)))

        errors: List[str] = []
        examples: List[Dict[str, object]] = []
        for ts in sample_ts:
            fp = folder_path / f"{ts}.npz"
            try:
                arr = load_npz_flexible(fp, pol.prefer_key, pol.fallback_key)
                info: Dict[str, object] = {"ts": ts, "shape": list(arr.shape), "dtype": str(arr.dtype)}
                if args.expected_hw is not None and arr.ndim >= 2:
                    H, W = args.expected_hw
                    a2 = ensure_2d(arr)
                    ah, aw = a2.shape[-2], a2.shape[-1]
                    if (ah, aw) != (H, W):
                        errors.append(f"{f}: {ts} HW=({ah},{aw}) != expected ({H},{W})")
                if pol.expect_binary:
                    if not check_binary(arr):
                        errors.append(f"{f}: {ts} not binary")
                else:
                    st = finite_stats(arr)
                    info.update(st)
                    if st["finite_frac"] < 1.0:
                        errors.append(f"{f}: {ts} finite_frac={st['finite_frac']:.6f} (<1.0)")
                examples.append(info)
            except Exception as e:
                errors.append(f"{f}: {ts} failed: {type(e).__name__}: {e}")

        deep_stats[f] = {"checked": len(sample_ts), "errors": errors[:200], "examples": examples[:20]}

    split_report = None
    if args.split_json:
        sp = Path(args.split_json)
        all_days = {ts_day(t) for t in common} if common else {ts_day(t) for t in union}
        split_report = validate_split_json(sp, all_days)

    report: Dict[str, object] = {
        "base_dir": str(base_dir),
        "folders": folders,
        "counts": count_report,
        "alignment": {"common_count": len(common), "union_count": len(union), "missing_by_folder": {k: v[:50] for k, v in missing.items()}},
        "deep_check": deep_stats,
        "split_check": split_report,
    }

    print("=== COUNT CHECK ===")
    for f in folders:
        c = count_report[f]
        print(f"- {f}: {c['count']} files (expected {c['expected']}) -> {'OK' if c['ok'] else 'MISMATCH'}")

    print("\n=== ALIGNMENT ===")
    print(f"- Common timestamps: {len(common)}")
    print(f"- Union timestamps : {len(union)}")
    if len(common) != len(union):
        print("WARNING: timestamp sets differ across folders.")

    print("\n=== DEEP CHECK ===")
    for f in folders:
        errs = deep_stats[f]["errors"]
        print(f"- {f}: checked={deep_stats[f]['checked']} errors={len(errs)}")
        if errs:
            print("  first_error:", errs[0])

    if split_report:
        print("\n=== SPLIT CHECK ===")
        print(f"- counts: {split_report['counts']} disjoint_ok={split_report['disjoint_ok']}")

    if args.report_json:
        outp = Path(args.report_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report: {outp}")

    fatal = any(not count_report[f]["ok"] for f in folders) or (len(common) != len(union))
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
