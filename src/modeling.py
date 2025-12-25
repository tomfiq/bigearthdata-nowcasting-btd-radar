#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/modeling.py
Model + losses + optimizer (U-Net separable + temporal 3D stem + ASPP + deep supervision)
Diambil dari notebook Kaggle Anda dan dirapikan menjadi modul.

Catatan:
- Model menerima input (B, H, W, T*C_base) dan melakukan reshape internal ke (B,H,W,T,C_base).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers

model_name = "unet_separable_tstem_aspp_ds"
# =========================================================
# 1) Losses (class-balanced focal + hierarchical penalty)
# =========================================================
def class_balanced_focal_loss(alpha, gamma=2.0, eps=1e-6):
    alpha = tf.constant(alpha, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -(y_true*tf.math.log(y_pred) + (1.0 - y_true)*tf.math.log(1.0 - y_pred))
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        focal = tf.pow(1.0 - pt, gamma) * ce
        w = tf.reshape(alpha, [1,1,1,3])
        focal = focal * w
        return tf.reduce_mean(focal)
    return loss

def hierarchical_penalty(lambda_h=0.05):
    # Menjaga T3 ≤ T2 ≤ T1
    def penalty(y_true, y_pred):
        p1, p2, p3 = y_pred[...,0], y_pred[...,1], y_pred[...,2]
        viol = tf.nn.relu(p2 - p1) + tf.nn.relu(p3 - p2)
        return lambda_h * tf.reduce_mean(viol)
    return penalty

def make_total_loss(alpha=(1.0,2.0,4.0), gamma=2.0, lambda_h=0.05):
    cb_focal = class_balanced_focal_loss(alpha=alpha, gamma=gamma)
    hpen     = hierarchical_penalty(lambda_h=lambda_h)
    def total(y_true, y_pred):
        return cb_focal(y_true, y_pred) + hpen(y_true, y_pred)
    return total

# =========================================================
# 2) Metrics (PR-AUC per channel untuk output 'main')
# =========================================================
class PerChannelPR(tf.keras.metrics.AUC):
    def __init__(self, idx, name):
        super().__init__(name=name, curve="PR")
        self.idx = idx
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_t = tf.cast(y_true[..., self.idx], tf.float32)
        y_p = tf.cast(y_pred[..., self.idx], tf.float32)
        return super().update_state(y_t, y_p, sample_weight=sample_weight)

# =========================================================
# 3) Optimizer schedule (warmup + cosine) + LR logging
# =========================================================
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, name=None):
        super().__init__()
        # simpan sebagai tipe Python agar gampang diserialisasi
        self.base_lr = float(base_lr)
        self.warmup_steps = int(max(1, warmup_steps))
        self.total_steps  = int(max(1, total_steps))
        self.name = name

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        # linear warmup
        lr_warm = self.base_lr * (step / warm_steps)

        # cosine after warmup
        progress = tf.clip_by_value((step - warm_steps) / tf.maximum(1.0, total_steps - warm_steps), 0.0, 1.0)
        lr_cos = 0.5 * self.base_lr * (1.0 + tf.cos(math.pi * progress))

        return tf.where(step < warm_steps, lr_warm, lr_cos)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class LRSnapshot(Callback):
    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        lr = opt.learning_rate(opt.iterations) if callable(opt.learning_rate) else opt.learning_rate
        logs = logs or {}
        logs["lr"] = float(tf.keras.backend.get_value(lr))

def make_optimizer(base_lr, steps_per_epoch, total_epochs, warmup_epochs=1, weight_decay=1e-4, clip_norm=1.0):
    lr_fn = WarmupCosine(
        base_lr=base_lr,
        warmup_steps=warmup_epochs * steps_per_epoch,
        total_steps=total_epochs * steps_per_epoch
    )
    opt = tf.keras.optimizers.AdamW(learning_rate=lr_fn, weight_decay=weight_decay, clipnorm=clip_norm)
    return opt

# =========================================================
# 4) Blocks & Model (temporal 3D stem + ASPP lebar + deep supervision)
# =========================================================
def sep_conv_block(x, filters, k=3, activation="relu", use_bn=True, sd_rate=0.0, l2_reg=1e-4):
    init = x
    x = layers.SeparableConv2D(filters, k, padding="same", use_bias=not use_bn,
                               depthwise_initializer="he_normal", pointwise_initializer="he_normal",
                               depthwise_regularizer=regularizers.l2(l2_reg),
                               pointwise_regularizer=regularizers.l2(l2_reg))(x)
    if use_bn: x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.SeparableConv2D(filters, k, padding="same", use_bias=not use_bn,
                               depthwise_initializer="he_normal", pointwise_initializer="he_normal",
                               depthwise_regularizer=regularizers.l2(l2_reg),
                               pointwise_regularizer=regularizers.l2(l2_reg))(x)
    if use_bn: x = layers.BatchNormalization()(x)
    if init.shape[-1] != filters:
        init = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                             kernel_initializer="he_normal",
                             kernel_regularizer=regularizers.l2(l2_reg))(init)
        if use_bn: init = layers.BatchNormalization()(init)
    x = layers.Add()([x, init])
    x = layers.Activation(activation)(x)
    if sd_rate > 0: x = layers.SpatialDropout2D(sd_rate)(x)
    return x

def dilated_pyramid(x, filters, activation="relu", use_bn=True, l2_reg=1e-4, rates=(1,2,4,8,16)):
    outs = []
    for r in rates:
        y = layers.SeparableConv2D(filters, 3, padding="same", dilation_rate=r,
                                   use_bias=not use_bn,
                                   depthwise_regularizer=regularizers.l2(l2_reg),
                                   pointwise_regularizer=regularizers.l2(l2_reg))(x)
        if use_bn: y = layers.BatchNormalization()(y)
        y = layers.Activation(activation)(y)
        outs.append(y)
    x = layers.Concatenate()(outs)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=not use_bn,
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    if use_bn: x = layers.BatchNormalization()(x)
    return layers.Activation(activation)(x)
def temporal_stem_3d(inputs, t_steps=12):
    ctot = inputs.shape[-1]
    assert ctot % t_steps == 0, "Jumlah channel input harus kelipatan t_steps"
    c_per = ctot // t_steps
    def _reshape_to_3d(x):
        b,h,w,c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        return tf.reshape(x, (b,h,w,t_steps,c_per))  # (B,H,W,T,C_per)
    x = layers.Lambda(_reshape_to_3d)(inputs)
    x = layers.Conv3D(16, (3,3,3), padding="same", activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Conv3D(16, (3,3,3), padding="same", activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Conv3D(32, (1,1,t_steps), padding="valid", activation="relu")(x); x = layers.BatchNormalization()(x)
    x = layers.Lambda(lambda z: tf.squeeze(z, axis=3))(x)  # -> (B,H,W,32)
    return x

def build_model(input_size=(128,128,48), base_filters=16, use_bn=True, activation="relu",
                l2_reg=1e-4, t_steps=12,
                tier_enc=(0.10,0.20,0.30), tier_dec=(0.20,0.10,0.10),
                aspp_rates=(1,2,4,8,16), with_deep_supervision=True):
    inputs = tf.keras.Input(input_size)
    x0 = temporal_stem_3d(inputs, t_steps=t_steps)

    c1 = sep_conv_block(x0, base_filters*1, use_bn=use_bn, activation=activation, sd_rate=tier_enc[0], l2_reg=l2_reg)
    p1 = layers.MaxPooling2D(2)(c1)
    c2 = sep_conv_block(p1, base_filters*2, use_bn=use_bn, activation=activation, sd_rate=tier_enc[1], l2_reg=l2_reg)
    p2 = layers.MaxPooling2D(2)(c2)
    c3 = sep_conv_block(p2, base_filters*4, use_bn=use_bn, activation=activation, sd_rate=tier_enc[2], l2_reg=l2_reg)
    p3 = layers.MaxPooling2D(2)(c3)

    bn = dilated_pyramid(p3, base_filters*8, use_bn=use_bn, activation=activation, l2_reg=l2_reg, rates=aspp_rates)
    bn = layers.SpatialDropout2D(0.30)(bn)

    u1 = layers.Conv2DTranspose(base_filters*4, 3, strides=2, padding="same")(bn)
    u1 = layers.Concatenate()([u1, c3])
    c4 = sep_conv_block(u1, base_filters*4, use_bn=use_bn, activation=activation, sd_rate=tier_dec[0], l2_reg=l2_reg)

    u2 = layers.Conv2DTranspose(base_filters*2, 3, strides=2, padding="same")(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = sep_conv_block(u2, base_filters*2, use_bn=use_bn, activation=activation, sd_rate=tier_dec[1], l2_reg=l2_reg)

    u3 = layers.Conv2DTranspose(base_filters*1, 3, strides=2, padding="same")(c5)
    u3 = layers.Concatenate()([u3, c1])
    c6 = sep_conv_block(u3, base_filters*1, use_bn=use_bn, activation=activation, sd_rate=tier_dec[2], l2_reg=l2_reg)

    main_out = layers.Conv2D(3, 1, activation="sigmoid", name="main", dtype="float32")(c6)
    if with_deep_supervision:
        aux1 = layers.Conv2DTranspose(3, 3, strides=2, padding="same", activation="sigmoid", name="aux1", dtype="float32")(c5)
        model = tf.keras.Model(inputs, [main_out, aux1], name=model_name)
    else:
        model = tf.keras.Model(inputs, main_out, name=model_name)
    return model

# =========================================================
# 5) Utils kompilasi (alpha dari prevalensi) & compile
# =========================================================
def alpha_from_prevalence(y, eps=1e-6, normalize=True):
    y = tf.cast(y, tf.float32)
    pos_rate = tf.reduce_mean(y, axis=[0,1,2])  # (3,)
    alpha = 1.0 / (pos_rate + eps)
    if normalize:
        alpha = alpha / tf.reduce_min(alpha)
    return [float(a) for a in alpha.numpy()]

def compile_model(model, y_train, steps_per_epoch, total_epochs,
                  base_lr=3e-4, warmup_epochs=1, weight_decay=1e-4, clip_norm=1.0,
                  focal_gamma=2.0, lambda_h=0.05, alpha_override=None,
                  extra_metrics=None):
    alpha = alpha_override if alpha_override is not None else alpha_from_prevalence(y_train)
    opt = make_optimizer(base_lr, steps_per_epoch, total_epochs,
                         warmup_epochs=1, weight_decay=weight_decay, clip_norm=clip_norm)
    total_loss = make_total_loss(alpha=alpha, gamma=focal_gamma, lambda_h=lambda_h)

    metrics_main = [
        PerChannelPR(0, "pr_auc_T1"),
        PerChannelPR(1, "pr_auc_T2"),
        PerChannelPR(2, "pr_auc_T3"),
    ]
    if extra_metrics:
        metrics_main += list(extra_metrics)  # pakai metrik custom kamu (F1/TS/Bias)

    if isinstance(model.output, (list, tuple)):
        model.compile(
            optimizer=opt,
            loss={"main": total_loss, "aux1": class_balanced_focal_loss(alpha=alpha, gamma=focal_gamma)},
            loss_weights={"main": 1.0, "aux1": 0.3},
            metrics={"main": metrics_main}
        )
    else:
        model.compile(optimizer=opt, loss=total_loss, metrics=metrics_main)
    return model, alpha


def oversample_indices_by_last_class_grow(tr_idx, Y_all, reshape_target,
                                          target_h=0.90, last_class=2,
                                          seed=0, max_rep_per_sample=None):
    """
    Upsample positif (sekuen yang mengandung T3) dengan menambah indeks duplikasi
    hingga proporsi ~ target_h. Total N akan BERTAMBAH.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    # Label sekuen pada subset train saat ini
    Y_subset = reshape_target(np.take(Y_all, tr_idx, axis=0).astype(np.float32)) # (N, H, W, 3)
    y_last   = Y_subset[..., last_class]                                         # (N, H, W)
    # Positif bila ada piksel T3 >0 di sekuen tsb
    is_pos   = (y_last.sum(axis=(1,2)) > 0)

    tr_idx   = np.asarray(tr_idx)
    pos_idx  = tr_idx[is_pos]
    neg_idx  = tr_idx[~is_pos]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    # Hitung target banyaknya positif setelah upsample
    if target_h >= 1.0:
        target_h = 0.9999
    n_pos_target = int(np.ceil((target_h/(1.0 - target_h)) * n_neg))
    n_add = max(0, n_pos_target - n_pos)

    if n_add > 0:
        # Batasi duplikasi per-sampel bila perlu
        if max_rep_per_sample is not None and n_pos > 0:
            # alokasi duplikasi merata, tidak melebihi max_rep_per_sample
            reps_per = np.zeros(n_pos, dtype=int)
            k = n_add
            i = 0
            while k > 0 and i < n_pos * max_rep_per_sample:
                reps_per[i % n_pos] += 1
                k -= 1
                i += 1
            dup_idx = np.repeat(pos_idx, reps_per)
        else:
            dup_idx = pos_idx[rng.integers(0, n_pos, size=n_add)]
        tr_idx_os = np.concatenate([neg_idx, pos_idx, dup_idx])
    else:
        # Sudah memenuhi (atau melebihi) target_h → cukup gabungkan saja
        tr_idx_os = np.concatenate([neg_idx, pos_idx])

    rng.shuffle(tr_idx_os)
    return tr_idx_os, Y_subset  # Y_train_pre (distribusi sebelum OS)

