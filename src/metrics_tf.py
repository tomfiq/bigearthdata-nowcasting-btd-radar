#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/metrics_tf.py
Metrik & callback custom dari notebook Kaggle Anda.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class F1ScorePerChannel(tf.keras.metrics.Metric):
    def __init__(self, channel_index, name=f'f1_score_channel', **kwargs):
        super(F1ScorePerChannel, self).__init__(name=name, **kwargs)
        self.channel_index = channel_index
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_c = y_true[..., self.channel_index]
        y_pred_c = tf.round(y_pred[..., self.channel_index])

        tp = tf.reduce_sum(y_true_c * y_pred_c)
        fp = tf.reduce_sum((1 - y_true_c) * y_pred_c)
        fn = tf.reduce_sum(y_true_c * (1 - y_pred_c))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1 = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
        return f1

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

class ThreatScorePerChannel(tf.keras.metrics.Metric):
    def __init__(self, channel_index, name='threat_score_channel', **kwargs):
        super(ThreatScorePerChannel, self).__init__(name=name, **kwargs)
        self.channel_index = channel_index
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_c = y_true[..., self.channel_index]
        y_pred_c = tf.round(y_pred[..., self.channel_index])

        tp = tf.reduce_sum(y_true_c * y_pred_c)
        fp = tf.reduce_sum((1 - y_true_c) * y_pred_c)
        fn = tf.reduce_sum(y_true_c * (1 - y_pred_c))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        ts = self.tp / (self.tp + self.fn + self.fp + tf.keras.backend.epsilon())
        return ts

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

class BiasMetricPerChannel(tf.keras.metrics.Metric):
    def __init__(self, channel_index, name='bias_metric_channel', **kwargs):
        super(BiasMetricPerChannel, self).__init__(name=name, **kwargs)
        self.channel_index = channel_index
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_c = y_true[..., self.channel_index]
        y_pred_c = tf.round(y_pred[..., self.channel_index])

        tp = tf.reduce_sum(y_true_c * y_pred_c)
        fp = tf.reduce_sum((1 - y_true_c) * y_pred_c)
        fn = tf.reduce_sum(y_true_c * (1 - y_pred_c))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        bias = (self.tp + self.fp) / (self.tp + self.fn + tf.keras.backend.epsilon())
        return bias

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)


# -------------------------
# Load TRAIN & VAL dari NPZ
# -------------------------
file_train = os.path.join(base_dir, f"dataset_train(strTr3-strVa3-strTe3-hor{horizon}-win12)_{mode}.npz")
file_val   = os.path.join(base_dir, f"dataset_val(strTr3-strVa3-strTe3-hor{horizon}-win12)_{mode}.npz")

print("Load TRAIN:", file_train)
train_npz = np.load(file_train, allow_pickle=True)
X_all     = train_npz["input_np"]        # (Ntr,T,H,W,C)
Y_all     = train_npz["target_np"]       # (Ntr,1,H,W,3)
Tin_all   = train_npz["input_timestamps"]
Ttar_all  = train_npz["target_timestamps"]
del train_npz; gc.collect()

print("Load  VAL :", file_val)
val_npz   = np.load(file_val, allow_pickle=True)
X_val_raw = val_npz["input_np"]          # (Nva,T,H,W,C)
Y_val_raw = val_npz["target_np"]         # (Nva,1,H,W,3)
Tin_val   = val_npz["input_timestamps"]
Ttar_val  = val_npz["target_timestamps"]
del val_npz; gc.collect()

print(f"TRAIN raw : X={X_all.shape}, Y={Y_all.shape}")
print(f"VAL   raw : X={X_val_raw.shape}, Y={Y_val_raw.shape}")

# -------------------------
# Oversampling pada TRAIN (berbasis T3)
# -------------------------
tr_idx = np.arange(X_all.shape[0], dtype=int)
tr_idx_os, Y_train_pre = oversample_indices_by_last_class_grow(
    tr_idx, Y_all, reshape_target,
    target_h=0.90,         # atur sesuai kebutuhan (0.7–0.9 umum)
    last_class=2,          # T3
    seed=123,
    max_rep_per_sample=10  # batasi duplikasi per sampel positif
)

# Logging rasio sebelum/sesudah OS (optional)
N_old      = len(tr_idx)
Y_before   = reshape_target(np.take(Y_all, tr_idx, axis=0).astype(np.float32))
pos_before = (Y_before[..., 2].sum(axis=(1,2)) > 0)
P_old      = int(pos_before.sum())
r_old      = P_old / max(1, N_old)

N_new      = len(tr_idx_os)
Y_after    = reshape_target(np.take(Y_all, tr_idx_os, axis=0).astype(np.float32))
pos_after  = (Y_after[..., 2].sum(axis=(1,2)) > 0)
r_new      = float(pos_after.mean())
print(f"[OS] N_old={N_old} P_old={P_old} r_old={r_old:.3f}  ->  N_new={N_new} r_new={r_new:.3f}")

# -------------------------
# Reshape untuk model
# -------------------------
X_train = reshape_for_model(np.take(X_all, tr_idx_os, axis=0).astype(np.float32))   # (Ntr_os,H,W,T*C)
Y_train = reshape_target   (np.take(Y_all, tr_idx_os, axis=0).astype(np.float32))   # (Ntr_os,H,W,3)

X_val   = reshape_for_model(X_val_raw.astype(np.float32))                           # (Nva,H,W,T*C)
Y_val   = reshape_target   (Y_val_raw.astype(np.float32))                           # (Nva,H,W,3)

input_channels = X_train.shape[-1]
assert input_channels % WINDOW_STEPS == 0, "Dim channel akhir harus habis dibagi 12 (WINDOW_STEPS)."
cpt = input_channels // WINDOW_STEPS   # 1 (onlyRadar) atau 4 (BTD+Radar)

# -------------------------
# Normalisasi pakai statistik TRAIN (pra-OS) pada kanal dasar C
# -------------------------
C_base = X_all.shape[-1]   # 1 atau 4
mean_c = X_all.reshape(-1, C_base).mean(axis=0, dtype=np.float32)   # (C_base,)
std_c  = X_all.reshape(-1, C_base).std (axis=0, dtype=np.float32)   # (C_base,)
std_c[std_c == 0] = 1.0

for t in range(WINDOW_STEPS):
    s, e = t * cpt, (t + 1) * cpt
    X_train[..., s:e] = (X_train[..., s:e] - mean_c) / std_c
    X_val  [..., s:e] = (X_val  [..., s:e] - mean_c) / std_c

print(f"After reshape+norm: X_train {X_train.shape}, Y_train {Y_train.shape}, X_val {X_val.shape}, Y_val {Y_val.shape}")

# -------------------------
# Build & compile model
# -------------------------
# Dropout tier disesuaikan horizon (opsional)
enc_drop = (0.10, 0.20, 0.30) if horizon < 60 else (0.12,0.20,0.30)
dec_drop = (0.20, 0.10, 0.10) if horizon < 60 else (0.20,0.12,0.10)

model = build_model(
    input_size=(128, 128, input_channels),
    base_filters=16,
    l2_reg=1e-4,
    t_steps=12,
    tier_enc=enc_drop,
    tier_dec=dec_drop,
    aspp_rates=(1,2,4,8,16),
    with_deep_supervision=True
)

steps_per_epoch = math.ceil(X_train.shape[0] / BATCH_SIZE)

# Metrik custom pada head 'main'
extra_metrics = [
    F1ScorePerChannel(channel_index=0, name='f1_score_channel_0'),
    F1ScorePerChannel(channel_index=1, name='f1_score_channel_1'),
    F1ScorePerChannel(channel_index=2, name='f1_score_channel_2'),
    ThreatScorePerChannel(channel_index=0, name='threat_score_channel_0'),
    ThreatScorePerChannel(channel_index=1, name='threat_score_channel_1'),
    ThreatScorePerChannel(channel_index=2, name='threat_score_channel_2'),
    BiasMetricPerChannel(channel_index=0, name='bias_metric_channel_0'),
    BiasMetricPerChannel(channel_index=1, name='bias_metric_channel_1'),
    BiasMetricPerChannel(channel_index=2, name='bias_metric_channel_2'),
]

# Pakai alpha dari distribusi ASLI (pra-OS)
alpha_pre = alpha_from_prevalence(Y_train_pre)

model, used_alpha = compile_model(
    model, Y_train,
    steps_per_epoch=steps_per_epoch, total_epochs=TOTAL_EPOCHS,
    base_lr=BASE_LR, warmup_epochs=1,
    weight_decay=1e-4, clip_norm=1.0,
    focal_gamma=2.0, lambda_h=0.05,
    alpha_override=alpha_pre,      # penting: bobot kelas dari pra-OS
    extra_metrics=extra_metrics
)
print(f"[FULL] Alpha bobot kelas dipakai (pra-OS): {used_alpha}")

# -------------------------
# Callbacks
# -------------------------

# Interval saver — simpan WEIGHTS saja agar file kecil & bebas masalah Lambda

class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, save_freq=10, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.save_freq = int(save_freq)
        self.epochs_since_last_save = 0
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.save_freq:
            self.epochs_since_last_save = 0
            fp = self.filepath.format(epoch=epoch + 1)
            self.model.save_weights(fp)
            print(f"\n[Interval] Epoch {epoch + 1}: save weights -> {fp}")

# Log LR ke history

class LRSnapshot(Callback):
    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        lr = opt.learning_rate(opt.iterations) if callable(opt.learning_rate) else opt.learning_rate
        logs = logs or {}
        logs["lr"] = float(tf.keras.backend.get_value(lr))

# Simpan terbaik berdasarkan PR-AUC T3 (head 'main')
best_t3_ckpt = ModelCheckpoint(
    os.path.join(save_dir, 'best_T3.weights.h5'),
    monitor='val_main_pr_auc_T3', mode='max',
    save_best_only=True, verbose=1,
    save_weights_only=True
)

best_t1f1_ckpt = ModelCheckpoint(
    os.path.join(save_dir, 'best_T1F1.weights.h5'),
    monitor='val_main_f1_score_channel_0', mode='max',
    save_best_only=True, verbose=1,
    save_weights_only=True
)

# Saver berkala (per 10 epoch)
interval_ckpt = CustomModelCheckpoint(
    filepath=os.path.join(save_dir, 'model_epoch{epoch:04d}.weights.h5'),
    save_freq=10
)

csv_logger = CSVLogger(os.path.join(save_dir, 'train_full.csv'))
lr_snap    = LRSnapshot()
early_stop = EarlyStopping(
    monitor='val_main_pr_auc_T3', mode='max',
    patience=20, restore_best_weights=True, verbose=1
)

# -------------------------
# Training
# -------------------------
history = model.fit(
    X_train, {"main": Y_train, "aux1": Y_train},
    validation_data=(X_val, {"main": Y_val, "aux1": Y_val}),
    epochs=TOTAL_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[lr_snap, csv_logger, interval_ckpt, best_t3_ckpt, best_t1f1_ckpt], #tanpa early_stop
    verbose=1
)

# -------------------------
# Simpan history
# -------------------------
np.save(os.path.join(save_dir, "history_full.npy"), history.history)

# (opsional) bersih-bersih
del X_train, Y_train, X_val, Y_val, X_all, Y_all, X_val_raw, Y_val_raw
gc.collect()
tf.keras.backend.clear_session()
