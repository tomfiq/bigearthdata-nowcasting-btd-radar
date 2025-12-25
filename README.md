# Big Earth Data – Nowcasting (BTD + Radar) – Reproducible Package

Repo kode untuk memenuhi kebutuhan **code + scripts + notebooks ringkas** agar paper Anda dapat direplikasi (Big Earth Data).

## Struktur folder (repo kode)

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
  README.md
```

## Dataset (repo data / Zenodo)

Dataset (archives, manifests, split_days_fixed.json, validation_report.json) disarankan dirilis di Zenodo.
- Dataset DOI: `10.5281/zenodo.<DATASET_DOI>`
- Code DOI (GitHub Release -> Zenodo): `10.5281/zenodo.<CODE_DOI>`

## 1) Ekstrak arsip dataset (opsional)

Jika dataset Anda berupa arsip (zip/rar/tar.gz), ekstrak menjadi folder:
`btd_13-08/ btd_15-13/ btd_16-13/ radar/ t1/ t2/ t3/`

Contoh (Windows):
```bat
python scripts\extract_archives.py --archives_dir "D:\NPZ_archives" --out_dir "D:\NPZ" --sevenzip "C:\Program Files\7-Zip\7z.exe"
```

Contoh (Linux/Kaggle):
```bash
python scripts/extract_archives.py --archives_dir "/kaggle/input/your-data/archives" --out_dir "/kaggle/working/NPZ"
```

## 2) Validasi dataset

Dengan radar:
```bash
python scripts/validate_npz.py --base_dir "D:\NPZ" --include_radar --expected_count 8640 --expected_hw 128 128 --split_json "split_days_fixed.json" --report_json "D:\NPZ\validation_report.json"
```

Tanpa radar (hanya BTD + T1/T2/T3):
```bash
python scripts/validate_npz.py --base_dir "D:\NPZ" --expected_count 8640 --expected_hw 128 128 --split_json "split_days_fixed.json" --report_json "D:\NPZ\validation_report.json"
```

## 3) Buat manifest train/val/test per lead (10/30/60)

Default (BTD + radar + T1/T2/T3):
```bash
python scripts/make_manifests.py --base_dir "D:\NPZ" --split_json "split_days_fixed.json" --out_dir "D:\NPZ\manifests" --check_files
```

Jika radar tidak tersedia, override folder wajib:
```bash
python scripts/make_manifests.py --base_dir "D:\NPZ" --split_json "split_days_fixed.json" --out_dir "D:\NPZ\manifests" --check_files --required_folders btd_13-08 btd_15-13 btd_16-13 t1 t2 t3
```

Output di `manifests/`:
- `train_lead10.csv`, `val_lead10.csv`, `test_lead10.csv`
- `train_lead30.csv`, dst
- `train_lead60.csv`, dst

## 4) Training (Kaggle / local)

Contoh (Kaggle):
```bash
python -m src.train \
  --base_dir "/kaggle/input/your-dataset/NPZ" \
  --manifest_dir "/kaggle/input/your-dataset/manifests" \
  --out_dir "/kaggle/working/run_h60_BTDRadar" \
  --mode BTDRadar --horizon 60 --epochs 80 --batch_size 4 --hw 128 128
```

Output:
- `best.weights.h5`, `last.weights.h5`
- `train_log.csv`, `history.json`
- `norm_stats.json`, `alpha.json`

## 5) Evaluasi test

```bash
python -m src.eval \
  --base_dir "/kaggle/input/your-dataset/NPZ" \
  --manifest_dir "/kaggle/input/your-dataset/manifests" \
  --weights "/kaggle/working/run_h60_BTDRadar/best.weights.h5" \
  --out_dir "/kaggle/working/run_h60_BTDRadar" \
  --mode BTDRadar --horizon 60 --hw 128 128
```

Hasil:
- `eval_test_lead60.json` dan `eval_test_lead60.csv`

## Catatan FAIR/Open

- Sertakan `split_days_fixed.json`, `manifests/*.csv`, dan `validation_report.json` pada paket data yang Anda rilis.
- Jika radar BMKG dibuka, tambahkan dokumen izin/ketentuan pemakaian radar pada repository data.

## Lisensi & sitasi

- Kode: lihat `LICENSE` (MIT).
- Sitasi kode: lihat `CITATION.cff`.
