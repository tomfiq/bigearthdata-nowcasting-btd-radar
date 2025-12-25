#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/eval_utils.py
Utility evaluasi (PR-AUC numpy, evaluasi per-hari) dari notebook Kaggle Anda.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any

def pr_auc_numpy(y_true, y_prob):
    m = tf.keras.metrics.AUC(curve='PR', summation_method='interpolation')
    m.update_state(y_true.ravel().astype(np.float32), y_prob.ravel().astype(np.float32))
    return float(m.result().numpy())

# -------------------------
# Paths & configuration (match your training namespace)
# -------------------------
file_test = os.path.join(base_dir, f"dataset_test(strTr3-strVa3-strTe3-hor{horizon}-win12)_{mode}.npz")
file_train= os.path.join(base_dir, f"dataset_train(strTr3-strVa3-strTe3-hor{horizon}-win12)_{mode}.npz")

# Which checkpoint to evaluate?
ckpt_paths = [
    os.path.join(model_dir, "best_T1F1.weights.h5"),
]
ckpts_to_run = [p for p in ckpt_paths if os.path.exists(p)]
if not ckpts_to_run:
    raise FileNotFoundError("No checkpoint found in model_dir.")

# -------------------------
# Load TEST (and TRAIN if needed for stats)
# -------------------------
print("Load TEST:", file_test)
te_npz = np.load(file_test, allow_pickle=True)
X_test_raw = te_npz["input_np"]    # (Nte,T,H,W,C)
Y_test_raw = te_npz["target_np"]   # (Nte,1,H,W,3)
input_ts_val   = te_npz["input_timestamps"]
target_ts_val  = te_npz["target_timestamps"]
del te_npz; gc.collect()
print("TEST raw :", X_test_raw.shape, Y_test_raw.shape)

# -------------------------
# Normalization (seperti biasa)
# -------------------------
norm_stats_path = os.path.join(save_dir, "norm_stats.json")
if os.path.exists(norm_stats_path):
    with open(norm_stats_path, "r") as f:
        d = json.load(f)
    mean_c = np.array(d["mean_c"], dtype=np.float32)
    std_c  = np.array(d["std_c"],  dtype=np.float32)
    print("[Eval] Loaded normalization from", norm_stats_path)
else:
    print("[Eval] norm_stats.json not found — recomputing from TRAIN:", file_train)
    tr_npz = np.load(file_train, allow_pickle=True)
    X_train_raw_for_stats = tr_npz["input_np"]
    C_base = X_train_raw_for_stats.shape[-1]
    mean_c = X_train_raw_for_stats.reshape(-1, C_base).mean(axis=0, dtype=np.float32)
    std_c  = X_train_raw_for_stats.reshape(-1, C_base).std (axis=0, dtype=np.float32)
    std_c[std_c == 0] = 1.0
    del tr_npz, X_train_raw_for_stats; gc.collect()

# Reshape & normalize TEST
X_test = reshape_for_model(X_test_raw.astype(np.float32))   # (N,H,W,T*C)
Y_test = reshape_target   (Y_test_raw.astype(np.float32))   # (N,H,W,3)
input_channels = X_test.shape[-1]
WINDOW_STEPS = 12
assert input_channels % WINDOW_STEPS == 0
cpt = input_channels // WINDOW_STEPS

for t in range(WINDOW_STEPS):
    s, e = t * cpt, (t + 1) * cpt
    X_test[..., s:e] = (X_test[..., s:e] - mean_c) / std_c

print("After reshape+norm: X_test", X_test.shape, "Y_test", Y_test.shape)

# -------------------------
# Baca split & cari 2 hari test dengan imputasi terbanyak
# -------------------------
# --- helper: robust extract "YYYYMMDD" from any timestamp-like string/bytes/ints
def day8(x):
    if isinstance(x, (bytes, bytearray)):
        x = x.decode()
    s = str(x)
    m = re.search(r'(\d{8})', s)        # ambil 8 digit berturut
    return m.group(1) if m else None

# 1) daftar hari yang benar-benar ada di TEST set (berdasarkan target_timestamps)
test_days_present = {day8(ts) for ts in target_ts_val}
test_days_present.discard(None)
print("[Info] Unique days in X_test/Y_test:", sorted(test_days_present))

# 2) ambil 2 hari TEST dengan imputasi terbanyak YANG JUGA ADA di dataset test
df_counts = pd.read_csv(IMPUTE_COUNTS_CSV)
df_counts_test = df_counts[df_counts["which_split"] == "test"].copy()
df_counts_test["date"] = df_counts_test["date"].astype(str)

df_counts_test_present = df_counts_test[df_counts_test["date"].isin(test_days_present)]
if len(df_counts_test_present) >= 2:
    top2_days = (df_counts_test_present.sort_values("test", ascending=False)
                                   .head(2)["date"].tolist())
    print("[Info] Top-2 most-imputed TEST days (present in dataset):", top2_days)
else:
    # Fallback: kalau tidak ada data imputasi yang overlap dengan dataset test,
    # jatuhkan ke "2 hari paling sering muncul di dataset test" (agar ada bedanya).
    from collections import Counter
    cnt = Counter(day8(ts) for ts in target_ts_val)
    top2_days = [d for d,_ in cnt.most_common(2)]
    print("[Warn] No overlap between imputed days and this TEST set.",
          "Falling back to most frequent days in TEST:", top2_days)

top2_set = set(top2_days)

# 3) buat indeks KEEP (FILTERED) dengan membuang scene yang tanggalnya termasuk di top2_set
all_idx  = np.arange(len(target_ts_val))
keep_idx = np.array([i for i,ts in enumerate(target_ts_val) if day8(ts) not in top2_set], dtype=int)
drop_idx = np.setdiff1d(all_idx, keep_idx)

print(f"[Info] Test scenes: total={len(all_idx)}, kept(filtered)={len(keep_idx)}, "
      f"dropped={len(drop_idx)}; dropped days={sorted(top2_set)}")

# --- lalu lanjutkan evaluasi:
# per_class_full, macro_full = eval_from_probs(Y_test, y_pred_main)
# per_class_f,    macro_f    = eval_from_probs(Y_test[keep_idx], y_pred_main[keep_idx])

# -------------------------
# Build model & helper evaluator
# -------------------------
enc_drop = (0.10, 0.20, 0.30) if horizon < 60 else (0.12, 0.20, 0.30)
dec_drop = (0.20, 0.10, 0.10) if horizon < 60 else (0.20, 0.12, 0.10)

model_eval = build_model(
    input_size=(128, 128, input_channels),
    base_filters=16,
    l2_reg=1e-4,
    t_steps=12,
    tier_enc=enc_drop,
    tier_dec=dec_drop,
    aspp_rates=(1,2,4,8,16),
    with_deep_supervision=True
)

def eval_from_probs(y_true, y_prob):
    out = []
    for c, cname in enumerate(["T1","T2","T3"]):
        y_t = y_true[..., c]
        y_p = y_prob[..., c]
        prauc = pr_auc_numpy(y_t, y_p)
        y_hat = (y_p >= 0.5).astype(np.float32)
        tp = float((y_hat*y_t).sum())
        fp = float((y_hat*(1-y_t)).sum())
        fn = float(((1-y_hat)*y_t).sum())
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1   = 2*precision*recall/(precision+recall+1e-8)
        ts   = tp/(tp+fp+fn+1e-8)
        bias = (tp+fp)/(tp+fn+1e-8)
        out.append({
            "class": cname,
            "PR_AUC": prauc,
            "F1@0.5": f1,
            "Threat@0.5": ts,
            "Bias@0.5": bias,
            "TP": tp, "FP": fp, "FN": fn
        })
    macro = {
        "class": "Macro(3)",
        "PR_AUC": float(np.mean([r["PR_AUC"] for r in out])),
        "F1@0.5": float(np.mean([r["F1@0.5"] for r in out])),
        "Threat@0.5": float(np.mean([r["Threat@0.5"] for r in out])),
        "Bias@0.5": float(np.mean([r["Bias@0.5"] for r in out])),
    }
    return out, macro

# -------------------------
# Evaluate each checkpoint: FULL vs FILTERED
# -------------------------
for ckpt in ckpts_to_run:
    print(f"\n[Eval] Loading weights: {ckpt}")
    model_eval.load_weights(ckpt)

    # Prediksi sekali untuk seluruh test
    y_pred = model_eval.predict(X_test, batch_size=4, verbose=1)
    y_pred_main = y_pred[0] if isinstance(y_pred, (list, tuple)) else y_pred

    # ---- FULL
    per_class_full, macro_full = eval_from_probs(Y_test, y_pred_main)
    print(f"[FULL] Horizon={horizon} | Mode={mode} | {os.path.basename(ckpt)}")
    for r in per_class_full:
        print(f"  {r['class']}: PR-AUC={r['PR_AUC']:.3f} | F1={r['F1@0.5']:.3f} | TS={r['Threat@0.5']:.3f} | Bias={r['Bias@0.5']:.3f}")
    print(f"  Macro : PR-AUC={macro_full['PR_AUC']:.3f} | F1={macro_full['F1@0.5']:.3f} | TS={macro_full['Threat@0.5']:.3f} | Bias={macro_full['Bias@0.5']:.3f}")

    # ---- FILTERED (drop 2 hari paling terimputasi)
    if len(keep_idx) > 0:
        per_class_f, macro_f = eval_from_probs(Y_test[keep_idx], y_pred_main[keep_idx])
        print(f"[FILTERED] Excluding days {sorted(list(top2_set))} -> kept {len(keep_idx)} scenes")
        for r in per_class_f:
            print(f"  {r['class']}: PR-AUC={r['PR_AUC']:.3f} | F1={r['F1@0.5']:.3f} | TS={r['Threat@0.5']:.3f} | Bias={r['Bias@0.5']:.3f}")
        print(f"  Macro : PR-AUC={macro_f['PR_AUC']:.3f} | F1={macro_f['F1@0.5']:.3f} | TS={macro_f['Threat@0.5']:.3f} | Bias={macro_f['Bias@0.5']:.3f}")
    else:
        per_class_f, macro_f = [], {"class":"Macro(3)","PR_AUC":np.nan,"F1@0.5":np.nan,"Threat@0.5":np.nan,"Bias@0.5":np.nan}
        print("[FILTERED] Tidak ada sampel yang tersisa setelah eksklusi.")

    # ---- Simpan keluaran
    tag_full = "full"
    tag_filt = f"filtered_no_{'_'.join(sorted(list(top2_set)))}"

    out_csv_full = os.path.join(save_dir, f"test_metrics_{os.path.basename(ckpt)}_{tag_full}.csv")
    out_json_full= os.path.join(save_dir, f"test_metrics_{os.path.basename(ckpt)}_{tag_full}.json")
    pd.DataFrame(per_class_full + [macro_full]).to_csv(out_csv_full, index=False)
    with open(out_json_full, "w") as f:
        json.dump({"horizon": horizon, "mode": mode, "checkpoint": ckpt,
                   "set": "FULL", "excluded_days": [],
                   "per_class": per_class_full, "macro": macro_full}, f, indent=2)

    out_csv_filt = os.path.join(save_dir, f"test_metrics_{os.path.basename(ckpt)}_{tag_filt}.csv")
    out_json_filt= os.path.join(save_dir, f"test_metrics_{os.path.basename(ckpt)}_{tag_filt}.json")
    pd.DataFrame(per_class_f + [macro_f]).to_csv(out_csv_filt, index=False)
    with open(out_json_filt, "w") as f:
        json.dump({"horizon": horizon, "mode": mode, "checkpoint": ckpt,
                   "set": "FILTERED", "excluded_days": sorted(list(top2_set)),
                   "kept_scenes": int(len(keep_idx)),
                   "per_class": per_class_f, "macro": macro_f}, f, indent=2)
    print("[Saved]", out_csv_full)
    print("[Saved]", out_json_full)
    print("[Saved]", out_csv_filt)
    print("[Saved]", out_json_filt)

# Optional cleanup
# del X_test_raw, Y_test_raw, X_test, Y_test, model_eval
# gc.collect()
# tf.keras.backend.clear_session()
