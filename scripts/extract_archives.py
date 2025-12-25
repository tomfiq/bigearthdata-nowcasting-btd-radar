#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_archives.py
Ekstrak arsip per file ke folder bernama sesuai stem arsip.

Mendukung: .zip, .tar, .tar.gz/.tgz, .rar.
Untuk .rar: akan coba python rarfile (butuh unrar/bsdtar), jika gagal fallback ke 7-Zip bila diberikan.

Contoh (Windows, RAR):
python scripts/extract_archives.py --archives_dir "D:\NPZ_archives" --out_dir "D:\NPZ" --sevenzip "C:\Program Files\7-Zip\7z.exe"

Contoh (Kaggle/Linux):
python scripts/extract_archives.py --archives_dir "/kaggle/input/your-dataset" --out_dir "/kaggle/working/NPZ"
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional

ARCHIVE_SUFFIXES = (".zip", ".rar", ".tar", ".tgz", ".tar.gz")


def strip_archive_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".tar.gz"):
        return name[:-7]
    if lower.endswith(".tgz"):
        return name[:-4]
    for sfx in (".zip", ".rar", ".tar"):
        if lower.endswith(sfx):
            return name[: -len(sfx)]
    return Path(name).stem


def safe_join(base_dir: Path, member_path: str) -> Path:
    """Prevent path traversal (../) in archive members."""
    dest = (base_dir / member_path).resolve()
    base_res = base_dir.resolve()
    if not str(dest).startswith(str(base_res) + os.sep) and dest != base_res:
        raise RuntimeError(f"Unsafe path in archive member: {member_path}")
    return dest


def extract_zip(src: Path, dst_dir: Path) -> None:
    with zipfile.ZipFile(src) as zf:
        for info in zf.infolist():
            member = info.filename.replace("\\", "/")
            if member.endswith("/"):
                (dst_dir / member).mkdir(parents=True, exist_ok=True)
                continue
            out_path = safe_join(dst_dir, member)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as fin, open(out_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)


def extract_tar(src: Path, dst_dir: Path) -> None:
    mode = "r:gz" if src.name.lower().endswith((".tar.gz", ".tgz")) else "r:"
    with tarfile.open(src, mode) as tf:
        for m in tf.getmembers():
            member = m.name.replace("\\", "/")
            out_path = safe_join(dst_dir, member)
            if m.isdir():
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fin = tf.extractfile(m)
            if fin is None:
                continue
            with fin, open(out_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)


def extract_rar_with_7z(src: Path, dst_dir: Path, sevenzip: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(sevenzip), "x", str(src), f"-o{str(dst_dir)}", "-y"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"7z extraction failed for {src.name}.\nOutput:\n{p.stdout}")


def extract_rar(src: Path, dst_dir: Path, sevenzip: Optional[Path] = None) -> None:
    try:
        import rarfile  # type: ignore
        dst_dir.mkdir(parents=True, exist_ok=True)
        with rarfile.RarFile(src) as rf:
            for info in rf.infolist():
                member = info.filename.replace("\\", "/")
                if info.isdir():
                    (dst_dir / member).mkdir(parents=True, exist_ok=True)
                    continue
                out_path = safe_join(dst_dir, member)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with rf.open(info) as fin, open(out_path, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
        return
    except Exception:
        if sevenzip is None:
            raise RuntimeError(
                "Gagal ekstrak RAR via python (rarfile/unrar tidak tersedia). "
                "Sediakan --sevenzip path ke 7z/7z.exe."
            )
        extract_rar_with_7z(src, dst_dir, sevenzip)


def iter_archives(archives_dir: Path) -> Iterable[Path]:
    for p in sorted(archives_dir.iterdir()):
        if p.is_file() and p.name.lower().endswith(ARCHIVE_SUFFIXES):
            yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archives_dir", type=str, required=True, help="Folder berisi arsip (*.zip/*.rar/*.tar.gz)")
    ap.add_argument("--out_dir", type=str, required=True, help="Folder output hasil ekstrak")
    ap.add_argument("--overwrite", action="store_true", help="Jika folder output per-arsip sudah ada, hapus dulu")
    ap.add_argument("--sevenzip", type=str, default="", help="Path ke 7z/7z.exe (fallback ekstrak RAR)")
    args = ap.parse_args()

    archives_dir = Path(args.archives_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sevenzip = Path(args.sevenzip) if args.sevenzip else None
    if sevenzip and not sevenzip.exists():
        raise FileNotFoundError(f"--sevenzip tidak ditemukan: {sevenzip}")

    archives = list(iter_archives(archives_dir))
    if not archives:
        raise RuntimeError(f"Tidak ada arsip ditemukan di: {archives_dir}")

    print(f"Found {len(archives)} archives in {archives_dir}")
    for src in archives:
        stem = strip_archive_suffix(src.name)
        dst = out_dir / stem
        if dst.exists() and args.overwrite:
            shutil.rmtree(dst)
        if dst.exists() and any(dst.iterdir()):
            print(f"[SKIP] {src.name} -> {dst} (sudah ada; gunakan --overwrite untuk re-extract)")
            continue

        print(f"[EXTRACT] {src.name} -> {dst}")
        suffix = src.name.lower()
        if suffix.endswith(".zip"):
            extract_zip(src, dst)
        elif suffix.endswith((".tar", ".tar.gz", ".tgz")):
            extract_tar(src, dst)
        elif suffix.endswith(".rar"):
            extract_rar(src, dst, sevenzip=sevenzip)
        else:
            print(f"[WARN] Unknown archive type: {src.name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
