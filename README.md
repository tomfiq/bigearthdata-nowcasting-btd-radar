# Rainfall Nowcasting (Himawari-8 BTD + Weather Radar) — Reproducible Code Package

This repository provides **reproducible code, scripts, and lightweight notebooks** for the manuscript:

**Radar–Satellite Fusion for Tropical Rainfall Nowcasting: Incorporating Himawari-8 BTD with Weather Radar in a U-Net Framework**

It is structured to support **FAIR/Open** expectations (e.g., Big Earth Data): dataset validation, fixed split protocol, manifest generation, training, and evaluation.

---

## Repository URL

- https://github.com/tomfiq/bigearthdata-nowcasting-btd-radar

---

## Persistent identifiers (DOI)

**Code (this repository, archived from a GitHub release):**
- Code DOI: `10.5281/zenodo.18055290`

**Dataset (Zenodo):**
- Dataset DOI (v1.0.0): `10.5281/zenodo.18055349`
- Dataset concept DOI (all versions): `10.5281/zenodo.18055348`

The dataset record contains compressed NPZ archives (`btd_*`, `t1/t2/t3`, `radar`), `split_days_fixed.json`, `manifests` (per lead time), and `validation_report.json`.

---

## Dataset summary (Zenodo)

- **Region / ROI:** Kendari area, Southeast Sulawesi, Indonesia (280 km × 280 km, **128×128 pixels**), bounded by **2.77°S–5.32°S** and **121.16°E–123.67°E**.
- **Temporal coverage:** **01 Jan–29 Feb 2024** (60 days), **10-minute cadence** (**8640** aligned timestamps).
- **Leads:** 10 / 30 / 60 minutes (manifests provided and/or generated).

---

## Repository layout

```
bigearthdata_nowcasting_repo/
  scripts/
    extract_archives.py
    validate_npz.py
    make_manifests.py
  src/
    __init__.py
    modeling.py
    metrics_tf.py
    eval_utils.py
    train.py
    eval.py
    data/
      __init__.py
      npz_sequence_tf.py
  train.ipynb
  eval.ipynb
  requirements.txt
  split_days_fixed.json
  LICENSE
  CITATION.cff
  README.md
```

---

## Quick start (recommended workflow)

### 0) Install dependencies

```bash
pip install -r requirements.txt
```

---

### 1) Prepare the dataset folder (`NPZ/`)

The scripts assume a `--base_dir` that contains the following subfolders:

`btd_13-08/  btd_15-13/  btd_16-13/  radar/  t1/  t2/  t3/`

**Option A — You already have the extracted folders**  
Point `--base_dir` directly to that directory (e.g., `D:\NPZ` on Windows or `/kaggle/input/<dataset>/NPZ` on Kaggle).

**Option B — You have ZIP archives (as distributed on Zenodo)**  
Extract the archives into a single `NPZ/` directory. Example (Linux/Kaggle):

```bash
mkdir -p NPZ
unzip -q btd_13-08.zip -d NPZ/
unzip -q btd_15-13.zip -d NPZ/
unzip -q btd_16-13.zip -d NPZ/
unzip -q radar.zip    -d NPZ/
unzip -q t1.zip       -d NPZ/
unzip -q t2.zip       -d NPZ/
unzip -q t3.zip       -d NPZ/
```

**Optional helper:** if your archives are `.rar`/mixed formats, you can use `scripts/extract_archives.py` (requires 7-Zip on Windows).

---

### 2) Validate the dataset (`validation_report.json`)

With radar:

```bash
python scripts/validate_npz.py --base_dir "D:\NPZ" --include_radar --expected_count 8640 --expected_hw 128 128 --split_json "split_days_fixed.json" --report_json "D:\NPZ\validation_report.json"
```

Without radar (BTD + targets only):

```bash
python scripts/validate_npz.py --base_dir "D:\NPZ" --expected_count 8640 --expected_hw 128 128 --split_json "split_days_fixed.json" --report_json "D:\NPZ\validation_report.json"
```

---

### 3) Generate train/val/test manifests per lead time (10/30/60 minutes)

Default (BTD + radar + T1/T2/T3):

```bash
python scripts/make_manifests.py --base_dir "D:\NPZ" --split_json "split_days_fixed.json" --out_dir "D:\NPZ\manifests" --check_files
```

If radar is not available, override required folders:

```bash
python scripts/make_manifests.py --base_dir "D:\NPZ" --split_json "split_days_fixed.json" --out_dir "D:\NPZ\manifests" --check_files --required_folders btd_13-08 btd_15-13 btd_16-13 t1 t2 t3
```

Outputs in `manifests/`:
- `train_lead10.csv`, `val_lead10.csv`, `test_lead10.csv`
- `train_lead30.csv`, `val_lead30.csv`, `test_lead30.csv`
- `train_lead60.csv`, `val_lead60.csv`, `test_lead60.csv`

---

### 4) Training (Kaggle / local)

Example (Kaggle):

```bash
python -m src.train \
  --base_dir "/kaggle/input/your-dataset/NPZ" \
  --manifest_dir "/kaggle/input/your-dataset/manifests" \
  --out_dir "/kaggle/working/run_h60_BTDRadar" \
  --mode BTDRadar --horizon 60 --epochs 80 --batch_size 4 --hw 128 128
```

Typical outputs:
- `best.weights.h5`, `last.weights.h5`
- `train_log.csv`, `history.json`
- `norm_stats.json`, `alpha.json`

---

### 5) Test evaluation

```bash
python -m src.eval \
  --base_dir "/kaggle/input/your-dataset/NPZ" \
  --manifest_dir "/kaggle/input/your-dataset/manifests" \
  --weights "/kaggle/working/run_h60_BTDRadar/best.weights.h5" \
  --out_dir "/kaggle/working/run_h60_BTDRadar" \
  --mode BTDRadar --horizon 60 --hw 128 128
```

Outputs:
- `eval_test_lead60.json`
- `eval_test_lead60.csv`

---

## FAIR/Open notes

When releasing and reusing the dataset, include at minimum:
- `split_days_fixed.json`
- `manifests/*.csv` (or `manifests.zip`)
- `validation_report.json`

If radar data are governed by third-party terms (e.g., BMKG), please comply with the applicable permission/redistribution conditions.

---

## License and citation

- License: see `LICENSE` (MIT).
- How to cite: see `CITATION.cff`.

