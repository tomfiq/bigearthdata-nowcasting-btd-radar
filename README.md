# Big Earth Data – Rainfall Nowcasting (Himawari-8 BTD + Weather Radar) – Reproducible Package

This repository provides the **reproducible code, scripts, and lightweight notebooks** for the manuscript:
**“Radar–Satellite Fusion for Tropical Rainfall Nowcasting: Incorporating Himawari-8 BTD with Weather Radar in a U-Net Framework.”**

It is designed to support **FAIR/Open** expectations (Big Earth Data): dataset validation, split protocol, manifest generation, training, and evaluation.

---

## Repository layout (code repo)

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

## Data release (Zenodo / data repo)

The dataset (archives, manifests, `split_days_fixed.json`, `validation_report.json`) should be released on Zenodo.

- Dataset DOI: `10.5281/zenodo.<DATASET_DOI>`
- Code DOI (GitHub Release → Zenodo): `10.5281/zenodo.<CODE_DOI>`

**Note on restricted radar data (if applicable):**  
If the BMKG radar archive cannot be fully open, consider publishing (i) an open dataset (BTD + targets + manifests) and (ii) radar data as restricted/by-request, following your permission terms.

---

## 1) (Optional) Extract dataset archives

If your data are provided as archives (zip/rar/tar.gz), extract them into:

`btd_13-08/  btd_15-13/  btd_16-13/  radar/  t1/  t2/  t3/`

Example (Windows):
```bat
python scripts\extract_archives.py --archives_dir "D:\NPZ_archives" --out_dir "D:\NPZ" --sevenzip "C:\Program Files\7-Zip\7z.exe"
```

Example (Linux/Kaggle):
```bash
python scripts/extract_archives.py --archives_dir "/kaggle/input/your-data/archives" --out_dir "/kaggle/working/NPZ"
```

---

## 2) Validate the dataset

With radar:
```bash
python scripts/validate_npz.py --base_dir "D:\NPZ" --include_radar --expected_count 8640 --expected_hw 128 128 --split_json "split_days_fixed.json" --report_json "D:\NPZ\validation_report.json"
```

Without radar (BTD + targets only):
```bash
python scripts/validate_npz.py --base_dir "D:\NPZ" --expected_count 8640 --expected_hw 128 128 --split_json "split_days_fixed.json" --report_json "D:\NPZ\validation_report.json"
```

---

## 3) Generate train/val/test manifests per lead time (10/30/60 minutes)

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

## 4) Training (Kaggle / local)

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

## 5) Test evaluation

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

When releasing the dataset, include:
- `split_days_fixed.json`
- `manifests/*.csv`
- `validation_report.json`

If radar data are included, attach the relevant permission/terms of use documentation.

---

## License and citation

- License: see `LICENSE` (MIT).
- How to cite: see `CITATION.cff`.
