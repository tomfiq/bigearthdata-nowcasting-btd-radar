# Big Earth Data – Nowcasting (BTD + Radar) – Reproducible Package

Repo template untuk memenuhi kebutuhan **code + data + manifests** agar paper Anda dapat direplikasi (Big Earth Data).

## Struktur folder

```
bigearthdata_nowcasting_repo/
  scripts/
    extract_archives.py
    validate_npz.py
    make_manifests.py
  src/
    modeling.py
    metrics_tf.py
    eval_utils.py
    train.py
    eval.py
    data/
      npz_sequence_tf.py
  notebooks/
    klasifikasi-curah-hujan-v-2-0.ipynb
```

## 1) Ekstrak arsip dataset (opsional)
Jika dataset Anda masih berupa arsip (zip/rar/tar.gz), ekstrak menjadi folder:
`btd_13-08/ btd_15-13/ btd_16-13/ radar/ t1/ t2/ t3/`

Contoh:
```bash
python scripts/extract_archives.py --archives_dir "D:\NPZ_archives" --out_dir "D:\NPZ" --sevenzip "C:\Program Files\7-Zip\7z.exe"
```

## 2) Validasi dataset
```bash
python scripts/validate_npz.py --base_dir "D:\NPZ" --expected_count 8640 --expected_hw 128 128 --split_json "D:\split_days_fixed.json" --report_json "D:\NPZ\validation_report.json"
```

## 3) Buat manifest train/val/test per lead (10/30/60)
> Anda sudah punya `split_days_fixed.json`, tinggal jalankan:

```bash
python scripts/make_manifests.py --base_dir "D:\NPZ" --split_json "D:\split_days_fixed.json" --out_dir "D:\NPZ\manifests" --check_files
```

Output di `manifests/`:
- `train_lead10.csv`, `val_lead10.csv`, `test_lead10.csv`
- `train_lead30.csv`, dst
- `train_lead60.csv`, dst

## 4) Training (Kaggle / local)
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
- Sertakan `split_days_fixed.json`, `manifests/*.csv`, dan `checksums.sha256` pada paket data yang Anda rilis.
- Jika radar BMKG dibuka, tambahkan file lisensi/izin BMKG pada repository data.


## DOI

- Dataset DOI (v1.0.0): `10.5281/zenodo.18055349`
- Dataset concept DOI (all versions): `10.5281/zenodo.18055348`
- Code DOI (GitHub Release → Zenodo): `10.5281/zenodo.18055290`


## Repository URL

- https://github.com/tomfiq/bigearthdata-nowcasting-btd-radar
