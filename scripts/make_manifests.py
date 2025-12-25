#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_manifests.py
Membangun manifest timestamp dan sampel (windowed sequences) per split dan per lead.

Output:
- out_dir/all_timestamps.csv
- out_dir/train_lead10.csv, val_lead10.csv, test_lead10.csv
- out_dir/train_lead30.csv, ...
- out_dir/train_lead60.csv, ...

Contoh:
python scripts/make_manifests.py --base_dir "D:/NPZ" --split_json "D:/split_days_fixed.json" --out_dir "D:/NPZ/manifests" --check_files
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

TS_RE_DEFAULT = r"^\d{8}_\d{4}\.npz$"


@dataclass(frozen=True)
class SampleRow:
    split: str
    lead_min: int
    input_start_ts: str
    input_end_ts: str
    target_ts: str
    day: str
    base_index_in_day: int


def ts_day(ts: str) -> str:
    return ts[:8]


def list_timestamps_from_folder(folder: Path, ts_re: re.Pattern) -> List[str]:
    ts: List[str] = []
    for p in folder.iterdir():
        if p.is_file() and ts_re.match(p.name):
            ts.append(p.stem)
    ts.sort()
    return ts


def group_by_day(timestamps: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for t in timestamps:
        d = ts_day(t)
        out.setdefault(d, []).append(t)
    for d in out:
        out[d].sort()
    return out


def read_split_days(split_json: Path) -> Dict[str, List[str]]:
    obj = json.loads(split_json.read_text(encoding="utf-8"))
    train_days = obj.get("train_days", [])
    val_days = obj.get("val_days", [])
    test_days = obj.get("test_days", [])
    if not (isinstance(train_days, list) and isinstance(val_days, list) and isinstance(test_days, list)):
        raise ValueError("split_days_fixed.json harus punya train_days/val_days/test_days.")
    return {"train": train_days, "val": val_days, "test": test_days}


def compute_lead_steps(leads_min: Sequence[int], cadence_min: int) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for lm in leads_min:
        if lm % cadence_min != 0:
            raise ValueError(f"Lead {lm} menit tidak habis dibagi cadence {cadence_min} menit.")
        out[lm] = lm // cadence_min
    return out


def maybe_check_required_files(base_dir: Path, folders: Sequence[str], timestamps: Sequence[str]) -> Tuple[Set[str], Dict[str, int]]:
    valid = set(timestamps)
    missing_counts = {f: 0 for f in folders}
    for f in folders:
        if not (base_dir / f).exists():
            raise FileNotFoundError(base_dir / f)
    for ts in timestamps:
        for f in folders:
            if not (base_dir / f / f"{ts}.npz").exists():
                missing_counts[f] += 1
                valid.discard(ts)
    return valid, missing_counts


def build_samples_for_day(day_ts: Sequence[str], split_name: str, lead_min: int, lead_steps: int, window_steps: int, stride: int) -> List[SampleRow]:
    n = len(day_ts)
    rows: List[SampleRow] = []
    start_i = window_steps - 1
    last_i = n - 1 - lead_steps
    if last_i < start_i:
        return rows
    for i in range(start_i, last_i + 1, stride):
        rows.append(SampleRow(
            split=split_name,
            lead_min=lead_min,
            input_start_ts=day_ts[i - (window_steps - 1)],
            input_end_ts=day_ts[i],
            target_ts=day_ts[i + lead_steps],
            day=ts_day(day_ts[i]),
            base_index_in_day=i
        ))
    return rows


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--ts_source_folder", type=str, default="btd_15-13")
    ap.add_argument("--ts_regex", type=str, default=TS_RE_DEFAULT)
    ap.add_argument("--cadence_min", type=int, default=10)
    ap.add_argument("--window_steps", type=int, default=12)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--leads_min", type=int, nargs="+", default=[10, 30, 60])
    ap.add_argument("--required_folders", type=str, nargs="*", default=["btd_13-08", "btd_15-13", "btd_16-13", "t1", "t2", "t3", "radar"])
    ap.add_argument("--check_files", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    split_json = Path(args.split_json)
    out_dir = Path(args.out_dir)
    ts_re = re.compile(args.ts_regex)

    ts_src = base_dir / args.ts_source_folder
    if not ts_src.exists():
        raise FileNotFoundError(ts_src)

    all_ts = list_timestamps_from_folder(ts_src, ts_re)
    if not all_ts:
        raise RuntimeError(f"Tidak ada timestamp di {ts_src}")

    if args.check_files:
        valid_ts, missing_counts = maybe_check_required_files(base_dir, args.required_folders, all_ts)
        all_ts = [t for t in all_ts if t in valid_ts]
        print("File completeness filter ON")
        for f, c in missing_counts.items():
            print(f"- missing in {f}: {c}")
        print(f"- kept timestamps: {len(all_ts)}")

    by_day = group_by_day(all_ts)

    # all_timestamps.csv
    rows_all = []
    for d, lst in sorted(by_day.items()):
        for idx, ts in enumerate(lst):
            rows_all.append({"ts": ts, "day": d, "slot_in_day": idx})
    write_csv(out_dir / "all_timestamps.csv", rows_all, ["ts", "day", "slot_in_day"])

    splits = read_split_days(split_json)
    split_sets = {k: set(v) for k, v in splits.items()}
    if not (split_sets["train"].isdisjoint(split_sets["val"]) and split_sets["train"].isdisjoint(split_sets["test"]) and split_sets["val"].isdisjoint(split_sets["test"])):
        raise RuntimeError("Split days overlap!")

    lead_steps_map = compute_lead_steps(args.leads_min, args.cadence_min)
    days_in_data = set(by_day.keys())

    for lead_min, lead_steps in lead_steps_map.items():
        for split_name in ("train", "val", "test"):
            keep_days = sorted(list(split_sets[split_name] & days_in_data))
            rows: List[SampleRow] = []
            for day in keep_days:
                rows.extend(build_samples_for_day(by_day[day], split_name, lead_min, lead_steps, args.window_steps, args.stride))
            out_path = out_dir / f"{split_name}_lead{lead_min}.csv"
            write_csv(out_path, (r.__dict__ for r in rows), ["split", "lead_min", "input_start_ts", "input_end_ts", "target_ts", "day", "base_index_in_day"])
            print(f"[OK] {out_path.name}: {len(rows)} samples")

    print(f"Done. Manifests -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
