#!/usr/bin/env python3
"""
run_interpretation.py — SHAP + LIME for finetuned_lora_mid4_mean.

The ridge model was saved without weights.  We refit them here for the
top-N voxels using the same SVD solver used during training — fast (~minutes).

SHAP (closed-form linear):
    phi_j(x_i) = x_ij * w_jv
    X is zero-mean after z-scoring, so E[X_j] ≈ 0 and the full formula
    reduces to this.  Aggregated across delay blocks to yield per-BERT-dim
    importance; also mapped back to per-word influence per story.

LIME (word masking):
    Perturb input by zeroing delay-block contributions of individual words
    in the pre-built design matrix.  Fits a local linear model to measure
    per-word influence on the peak predicted-response window.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from lime.lime_tabular import LimeTabularExplainer
    _LIME_OK = True
except ImportError:
    LimeTabularExplainer = None
    _LIME_OK = False

# ── preprocessing constants (match preprocessing_utils.py) ───────────────────
TRIM_START = 5
TRIM_END   = 10
DELAYS     = [1, 2, 3, 4]
BERT_DIM   = 768
N_DELAYS   = len(DELAYS)   # 4  →  total features = 4 × 768 = 3 072


# ─────────────────────────────── data helpers ─────────────────────────────────

def _find_embedding(embedding_dir: str, sid: str, split: str, method: str) -> str:
    for ext in ("npy", "npz"):
        p = os.path.join(embedding_dir, f"{sid}_{split}_{method}_embeddings.{ext}")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"No embedding for {sid}/{split}/{method} in {embedding_dir}"
    )


def _load_embedding(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    data = np.load(path, allow_pickle=False)
    key  = "X" if "X" in data.files else data.files[0]
    return np.asarray(data[key], dtype=np.float32)


def _zscore_X(
    X_tr: np.ndarray, X_te: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu  = X_tr.mean(axis=0, keepdims=True)
    sig = X_tr.std(axis=0, keepdims=True)
    sig[sig < 1e-8] = 1.0
    return (
        ((X_tr - mu) / sig).astype(np.float32),
        ((X_te - mu) / sig).astype(np.float32),
        mu.astype(np.float32),
        sig.astype(np.float32),
    )


def _zscore_cols(arr: np.ndarray) -> np.ndarray:
    mu  = arr.mean(axis=0, keepdims=True)
    sig = arr.std(axis=0, keepdims=True)
    sig[sig < 1e-6] = 1.0
    return (arr - mu) / sig


def _col_corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (_zscore_cols(y_true) * _zscore_cols(y_pred)).mean(axis=0).astype(np.float32)


def _load_fmri_chunk(
    story_maps: Dict[str, np.ndarray],
    stories: List[str],
    lengths: List[int],
    total: int,
    voxel_idx: np.ndarray,
) -> np.ndarray:
    out = np.empty((total, len(voxel_idx)), dtype=np.float32)
    offset = 0
    for story, n in zip(stories, lengths):
        block = np.asarray(story_maps[story][:, voxel_idx], dtype=np.float32)
        np.nan_to_num(block, nan=0.0, copy=False)
        mu  = block.mean(axis=0, keepdims=True)
        sig = block.std(axis=0, keepdims=True)
        sig[sig < 1e-6] = 1.0
        out[offset : offset + n] = (block - mu) / sig
        offset += n
    return out


# ──────────────────────────── SVD ridge refit ─────────────────────────────────

def _svd_ridge_fit(
    X_train: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    alphas: np.ndarray,
    cv_frac: float = 0.2,
    singcutoff: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit ridge with per-voxel alpha selection. Returns (test_corrs, weights)."""
    n     = X_train.shape[0]
    n_val = max(1, int(n * cv_frac))
    n_cv  = n - n_val
    X_cv, X_val = X_train[:n_cv], X_train[n_cv:]
    Y_cv, Y_val = Y_train[:n_cv], Y_train[n_cv:]

    print("    SVD (cv split) …", flush=True)
    Ucv, Scv, Vhcv = np.linalg.svd(X_cv.astype(np.float64), full_matrices=False)
    Ucv, Scv, Vhcv = Ucv.astype(np.float32), Scv.astype(np.float32), Vhcv.astype(np.float32)
    ok = Scv > singcutoff
    Ucv, Scv, Vhcv = Ucv[:, ok], Scv[ok], Vhcv[ok]

    UR_cv   = (Ucv.T @ Y_cv).astype(np.float32)
    PVh_val = (X_val @ Vhcv.T).astype(np.float32)
    zY_val  = _zscore_cols(Y_val)

    best_a = np.full(Y_cv.shape[1], alphas[0], dtype=np.float32)
    best_r = np.full(Y_cv.shape[1], -np.inf,   dtype=np.float32)
    for a in alphas:
        D    = Scv / (Scv ** 2 + np.float32(a))
        pred = PVh_val @ (D[:, None] * UR_cv)
        r    = (_zscore_cols(pred) * zY_val).mean(axis=0)
        m    = r > best_r
        best_r[m] = r[m]
        best_a[m] = np.float32(a)

    print("    SVD (full train) …", flush=True)
    Uf, Sf, Vhf = np.linalg.svd(X_train.astype(np.float64), full_matrices=False)
    Uf, Sf, Vhf = Uf.astype(np.float32), Sf.astype(np.float32), Vhf.astype(np.float32)
    ok = Sf > singcutoff
    Uf, Sf, Vhf = Uf[:, ok], Sf[ok], Vhf[ok]
    UR_f = (Uf.T @ Y_train).astype(np.float32)

    W = np.empty((Vhf.shape[1], Y_train.shape[1]), dtype=np.float32)
    for a in np.unique(best_a):
        mask = best_a == a
        D    = Sf / (Sf ** 2 + np.float32(a))
        W[:, mask] = Vhf.T @ (D[:, None] * UR_f[:, mask])

    corrs = _col_corr(Y_test, X_test @ W)
    return corrs, W


def refit_top_voxels(
    model_pkl: Dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    fmri_dir: str,
    train_stories: List[str],
    test_stories: List[str],
    n_candidate: int = 256,
) -> Dict:
    """Refit weights for the top-n_candidate voxels by overall test CC."""
    test_corrs_all = np.asarray(model_pkl["test_corrs"], dtype=np.float32)
    voxel_indices  = np.asarray(model_pkl["voxel_indices"], dtype=np.int32)
    alphas         = np.asarray(model_pkl["alphas_grid"], dtype=np.float32)

    ranked   = np.argsort(test_corrs_all)[::-1]
    chosen   = ranked[: min(n_candidate, len(ranked))]
    cand_ids = np.sort(voxel_indices[chosen]).astype(np.int32)

    train_maps = {
        s: np.load(os.path.join(fmri_dir, f"{s}.npy"), mmap_mode="r")
        for s in train_stories
    }
    test_maps = {
        s: np.load(os.path.join(fmri_dir, f"{s}.npy"), mmap_mode="r")
        for s in test_stories
    }
    tr_lens = [train_maps[s].shape[0] for s in train_stories]
    te_lens = [test_maps[s].shape[0]  for s in test_stories]

    print(f"  Loading fMRI ({len(cand_ids)} voxels) …", flush=True)
    Y_train = _load_fmri_chunk(train_maps, train_stories, tr_lens, sum(tr_lens), cand_ids)
    Y_test  = _load_fmri_chunk(test_maps,  test_stories,  te_lens, sum(te_lens), cand_ids)

    print("  Refitting ridge …", flush=True)
    corrs, W = _svd_ridge_fit(X_train, X_test, Y_train, Y_test, alphas)

    slices: Dict[str, slice] = {}
    off = 0
    for story, n in zip(test_stories, te_lens):
        slices[story] = slice(off, off + n)
        off += n

    preds  = X_test @ W
    story_preds  = {s: preds[sl].astype(np.float32)  for s, sl in slices.items()}
    story_truths = {s: Y_test[sl].astype(np.float32) for s, sl in slices.items()}
    story_corrs  = {
        s: _col_corr(story_truths[s], story_preds[s]) for s in test_stories
    }

    return dict(
        voxel_indices=cand_ids,
        weights=W,
        test_corrs=corrs,
        story_corrs=story_corrs,
        story_preds=story_preds,
        story_truths=story_truths,
        story_slices=slices,
    )


# ──────────────────────────── SHAP (linear) ───────────────────────────────────

def run_shap(
    state: Dict,
    X_test: np.ndarray,
    stories: List[str],
    raw_text: Dict,
    out_dir: str,
    sid: str,
    method: str,
    n_top_dims: int = 20,
) -> None:
    W     = state["weights"]      # (p=3072, M)
    corrs = state["test_corrs"]   # (M,)

    # Importance per feature dim j: weighted sum over voxels of |w_jv| * mean_i|X_ij|
    # Weights proportional to voxel CC (only positive CC contributes).
    cc_w = np.maximum(corrs, 0.0).astype(np.float32)
    if cc_w.sum() < 1e-8:
        cc_w[:] = 1.0
    cc_w /= cc_w.sum()

    abs_X_mean = np.abs(X_test).mean(axis=0)          # (p,)
    # importance_j = sum_v  cc_v * |w_jv| * mean_i|X_ij|
    importance = (np.abs(W) * cc_w[None, :]).sum(axis=1) * abs_X_mean  # (p,)

    # Aggregate across delay blocks → (BERT_DIM,) for readable interpretation
    imp_blocks = importance.reshape(N_DELAYS, BERT_DIM)   # (4, 768)
    imp_agg    = imp_blocks.sum(axis=0)                    # (768,)
    top_dims   = np.argsort(-imp_agg)[:n_top_dims]

    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"{sid}_{method}")

    # Save full 3072-dim feature importance
    with open(f"{prefix}_shap_feature_importance.csv", "w", newline="") as f:
        wr = csv.DictWriter(
            f, fieldnames=["rank", "feature_dim", "delay_block", "bert_dim", "importance"]
        )
        wr.writeheader()
        for rank, j in enumerate(np.argsort(-importance)[:200], 1):
            wr.writerow({
                "rank": rank,
                "feature_dim": int(j),
                "delay_block": int(j // BERT_DIM) + 1,
                "bert_dim":    int(j %  BERT_DIM),
                "importance":  float(importance[j]),
            })

    # Save delay-aggregated BERT-dim importance
    with open(f"{prefix}_shap_bert_dim_importance.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["rank", "bert_dim", "importance"])
        wr.writeheader()
        for rank, d in enumerate(np.argsort(-imp_agg)[:200], 1):
            wr.writerow({"rank": rank, "bert_dim": int(d), "importance": float(imp_agg[d])})

    # Plot top-20 BERT dims
    _bar_plot(
        f"{prefix}_shap_bert_dim_top20.png",
        labels=[f"dim {d}" for d in top_dims],
        scores=imp_agg[top_dims].tolist(),
        title=f"{sid} | {method} | SHAP top-{n_top_dims} BERT dims (delay-aggregated)",
        xlabel="Aggregated |SHAP| × |weight|",
    )

    # Per-story word-level SHAP
    weighted_w = (np.abs(W) * cc_w[None, :]).sum(axis=1)  # (p,) weighted mean |w|
    for story in stories:
        if story not in state["story_slices"]:
            continue
        _story_word_shap(
            story=story,
            raw_text=raw_text,
            X_story=X_test[state["story_slices"][story]],
            weighted_w=weighted_w,
            out_dir=out_dir,
            prefix=f"{prefix}_{story}",
        )

    print(f"  SHAP → {out_dir}", flush=True)


def _story_word_shap(
    story: str,
    raw_text: Dict,
    X_story: np.ndarray,
    weighted_w: np.ndarray,
    out_dir: str,
    prefix: str,
    n_top: int = 20,
) -> None:
    story_data = raw_text.get(story)
    if story_data is None:
        return

    words      = list(story_data.data)
    word_times = np.asarray(story_data.data_times, dtype=np.float32)
    tr_times   = np.asarray(story_data.tr_times,   dtype=np.float32)
    trim_len   = len(tr_times) - TRIM_START - TRIM_END
    tr_trimmed = tr_times[TRIM_START : TRIM_START + trim_len]
    n_trs      = X_story.shape[0]

    word_scores = np.zeros(len(words), dtype=np.float64)
    for w_idx, w_time in enumerate(word_times):
        if tr_trimmed.size == 0:
            continue
        tr0 = int(np.argmin(np.abs(tr_trimmed - float(w_time))))
        for d_idx, delay in enumerate(DELAYS):
            tr_delayed = tr0 + delay
            if not (0 <= tr_delayed < n_trs):
                continue
            c0     = d_idx * BERT_DIM
            c1     = c0 + BERT_DIM
            # |phi| = |X[tr, dim]| * weighted_w[dim]  summed over bert dims
            word_scores[w_idx] += float(
                (np.abs(X_story[tr_delayed, c0:c1]) * weighted_w[c0:c1]).sum()
            )

    top_idx    = np.argsort(-word_scores)[:n_top]
    top_words  = [words[i] for i in top_idx]
    top_scores = [float(word_scores[i]) for i in top_idx]

    with open(f"{prefix}_shap_words.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["rank", "word", "shap_score"])
        wr.writeheader()
        for rank, (wd, sc) in enumerate(zip(top_words, top_scores), 1):
            wr.writerow({"rank": rank, "word": wd, "shap_score": sc})

    mx = max(top_scores) if top_scores else 1.0
    norm_scores = [s / (mx + 1e-12) for s in top_scores]

    _bar_plot(
        f"{prefix}_shap_words_top{n_top}.png",
        labels=top_words,
        scores=norm_scores,
        title=f"{story[:50]} | SHAP word influence (top {n_top})",
        xlabel="Normalised SHAP word score",
    )
    print(f"    {story} SHAP top-5: {', '.join(top_words[:5])}", flush=True)


# ─────────────────────────────── LIME ─────────────────────────────────────────

def _nearest_tr(word_time: float, tr_trimmed: np.ndarray) -> Optional[int]:
    if tr_trimmed.size == 0:
        return None
    return int(np.argmin(np.abs(tr_trimmed - word_time)))


def _zero_word_delays(X_copy: np.ndarray, tr_idx: int) -> None:
    """Zero out all delay-block columns corresponding to one word's TR."""
    n_trs = X_copy.shape[0]
    for d_idx, delay in enumerate(DELAYS):
        row = tr_idx + delay
        if 0 <= row < n_trs:
            c0 = d_idx * BERT_DIM
            X_copy[row, c0 : c0 + BERT_DIM] = 0.0


def run_lime(
    state: Dict,
    X_test_raw: np.ndarray,
    x_mu: np.ndarray,
    x_sigma: np.ndarray,
    test_stories: List[str],
    selected_stories: List[str],
    raw_text: Dict,
    out_dir: str,
    sid: str,
    method: str,
    top_k_voxels: int = 3,
    word_window: int = 48,
    peak_trs: int = 5,
    n_lime_samples: int = 300,
    n_features: int = 12,
    random_state: int = 42,
) -> None:
    if not _LIME_OK:
        print("  WARNING: 'lime' package not installed — skipping LIME.")
        return

    os.makedirs(out_dir, exist_ok=True)

    for story in selected_stories:
        if story not in test_stories:
            print(f"  SKIP: {story} not in test split", flush=True)
            continue

        story_data = raw_text.get(story)
        if story_data is None:
            print(f"  SKIP: {story} not in raw_text", flush=True)
            continue

        words      = list(story_data.data)
        word_times = np.asarray(story_data.data_times, dtype=np.float32)
        tr_times   = np.asarray(story_data.tr_times,   dtype=np.float32)
        trim_len   = len(tr_times) - TRIM_START - TRIM_END
        tr_trimmed = tr_times[TRIM_START : TRIM_START + trim_len]

        story_sl   = state["story_slices"][story]
        X_story_raw = X_test_raw[story_sl].copy()   # (T_story, 3072)

        story_corr = state["story_corrs"][story]
        ranked     = np.argsort(story_corr)[::-1]
        top_k      = ranked[: min(top_k_voxels, len(ranked))]

        story_dir = os.path.join(out_dir, story)
        os.makedirs(story_dir, exist_ok=True)

        for rank_idx, local_pos in enumerate(top_k, start=1):
            voxel_id  = int(state["voxel_indices"][local_pos])
            voxel_cc  = float(story_corr[local_pos])
            pred_sig  = state["story_preds"][story][:, local_pos]
            truth_sig = state["story_truths"][story][:, local_pos]
            w_vec     = state["weights"][:, local_pos].astype(np.float32)

            # Peak TR window
            top_tr_idx = np.sort(np.argsort(pred_sig)[::-1][:peak_trs]).astype(int)
            center_tr  = int(top_tr_idx[np.argmax(pred_sig[top_tr_idx])])
            if center_tr < len(tr_trimmed):
                center_time = float(tr_trimmed[center_tr])
            else:
                center_time = float(tr_trimmed[-1])
            center_word = int(np.argmin(np.abs(word_times - center_time)))

            half_w  = word_window // 2
            w_start = max(0, center_word - half_w)
            w_end   = min(len(words), w_start + word_window)
            w_start = max(0, w_end - word_window)

            local_words = words[w_start:w_end]
            feat_names  = [f"{w_start + i}:{wd}" for i, wd in enumerate(local_words)]
            word_to_tr  = [
                _nearest_tr(float(word_times[w_start + i]), tr_trimmed)
                for i in range(len(local_words))
            ]

            # Capture loop vars in default args to avoid closure pitfall
            def predict_fn(
                mask_matrix: np.ndarray,
                _X   = X_story_raw,
                _wv  = w_vec,
                _mu  = x_mu,
                _sig = x_sigma,
                _trs = top_tr_idx,
                _w2tr = word_to_tr,
                _wlen = len(local_words),
            ) -> np.ndarray:
                out = np.zeros(mask_matrix.shape[0], dtype=np.float32)
                for i, mask in enumerate(mask_matrix):
                    X_masked = _X.copy()
                    for j in range(_wlen):
                        if mask[j] < 0.5 and _w2tr[j] is not None:
                            _zero_word_delays(X_masked, _w2tr[j])
                    X_z    = ((X_masked - _mu) / _sig).astype(np.float32)
                    pred   = X_z @ _wv
                    out[i] = float(pred[_trs].mean())
                return out

            explainer = LimeTabularExplainer(
                training_data=np.vstack([
                    np.ones(len(local_words),  dtype=np.float32),
                    np.zeros(len(local_words), dtype=np.float32),
                ]),
                feature_names=feat_names,
                mode="regression",
                discretize_continuous=False,
                random_state=random_state,
            )
            exp     = explainer.explain_instance(
                np.ones(len(local_words), dtype=np.float32),
                predict_fn,
                num_features=min(n_features, len(local_words)),
                num_samples=n_lime_samples,
            )
            lime_df = _aggregate_lime(exp.as_list())

            prefix = os.path.join(story_dir, f"voxel{voxel_id:05d}_rank{rank_idx}")
            lime_df.to_csv(f"{prefix}_lime.csv", index=False)

            _save_response_plot(
                f"{prefix}_response.png",
                pred_sig, truth_sig,
                peak_min=int(top_tr_idx[0]),
                peak_max=int(top_tr_idx[-1]),
                title=f"{story[:45]} | voxel {voxel_id} | CC={voxel_cc:.3f}",
            )
            _save_lime_plot(
                f"{prefix}_lime.png",
                lime_df,
                title=f"{story[:45]} | voxel {voxel_id} | CC={voxel_cc:.3f}",
                max_rows=n_features,
            )
            top5 = ", ".join(lime_df["token"].head(5).tolist())
            print(
                f"    {story} rank{rank_idx} voxel{voxel_id} CC={voxel_cc:.3f}"
                f"  top: {top5}",
                flush=True,
            )

    print(f"  LIME → {out_dir}", flush=True)


def _aggregate_lime(exp_list: List[Tuple[str, float]]) -> pd.DataFrame:
    rows = []
    for feat, score in exp_list:
        token = feat.split(":", 1)[1] if ":" in feat else feat
        rows.append({"feature": feat, "token": token, "score": float(score)})
    if not rows:
        return pd.DataFrame(columns=["feature", "token", "score", "abs_score"])
    df = pd.DataFrame(rows)
    return (
        df.groupby("token", as_index=False)
          .agg(
              score     = ("score", "sum"),
              abs_score = ("score", lambda s: float(np.abs(s).sum())),
          )
          .sort_values("abs_score", ascending=False)
          .reset_index(drop=True)
    )


# ─────────────────────────────── plot helpers ─────────────────────────────────

def _bar_plot(
    path: str,
    labels: List[str],
    scores: List[float],
    title: str,
    xlabel: str,
) -> None:
    if not labels:
        return
    n    = min(20, len(labels))
    lbls = labels[:n][::-1]
    vals = scores[:n][::-1]
    colors = ["#54A24B" if v >= 0 else "#E45756" for v in vals]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * n)))
    ax.barh(lbls, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_response_plot(
    path: str,
    pred: np.ndarray,
    truth: np.ndarray,
    peak_min: int,
    peak_max: int,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    trs = np.arange(pred.shape[0])
    ax.plot(trs, truth, label="Observed",  color="#4C78A8", linewidth=1.4, alpha=0.8)
    ax.plot(trs, pred,  label="Predicted", color="#E45756", linewidth=1.6)
    ax.axvspan(peak_min, peak_max, color="#F3A712", alpha=0.25)
    ax.set_xlabel("TR within story")
    ax.set_ylabel("Voxel response (z-scored)")
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_lime_plot(
    path: str, df: pd.DataFrame, title: str, max_rows: int = 12
) -> None:
    if df.empty:
        return
    _bar_plot(
        path,
        labels=df["token"].tolist()[:max_rows],
        scores=df["score"].tolist()[:max_rows],
        title=title,
        xlabel="LIME weight",
    )


# ──────────────────────────────── CLI ─────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP + LIME for finetuned_lora_mid4_mean")
    p.add_argument("--subject",       default="subject3")
    p.add_argument("--subject-id",    default="s3")
    p.add_argument("--method",        default="finetuned_lora_mid4_mean")
    p.add_argument("--data-path",     default="/scratch/users/s214/lab3")
    p.add_argument("--embedding-dir", default="../../data/embeddings_mid4_mean")
    p.add_argument("--split-path",    default="../../data/train_test_split.json")
    p.add_argument("--result-dir",    default="../../results")
    p.add_argument("--output-dir",    default="../../results/interpretation")
    p.add_argument(
        "--stories",
        nargs="+",
        default=["onapproachtopluto", "marryamanwholoveshismother", "birthofanation"],
    )
    p.add_argument("--n-candidate",   type=int, default=256)
    p.add_argument("--top-k-voxels",  type=int, default=3)
    p.add_argument("--word-window",   type=int, default=48)
    p.add_argument("--peak-trs",      type=int, default=5)
    p.add_argument("--lime-samples",  type=int, default=300)
    p.add_argument("--n-features",    type=int, default=12)
    p.add_argument("--skip-shap",     action="store_true")
    p.add_argument("--skip-lime",     action="store_true")
    p.add_argument(
        "--state-pkl",
        default=None,
        help="Path to save/load the refitted top-voxel state pkl. "
             "If the file already exists the refit is skipped and the cached "
             "state is loaded instead.",
    )
    return p.parse_args()


def _resolve(script_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.normpath(
        os.path.join(script_dir, path)
    )


def main() -> None:
    args       = _parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    embedding_dir = _resolve(script_dir, args.embedding_dir)
    split_path    = _resolve(script_dir, args.split_path)
    result_dir    = _resolve(script_dir, args.result_dir)
    out_dir       = _resolve(script_dir, args.output_dir)
    fmri_dir      = os.path.join(args.data_path, args.subject)

    print(f"Subject : {args.subject} ({args.subject_id})")
    print(f"Method  : {args.method}")
    print(f"Stories : {args.stories}", flush=True)

    with open(split_path) as f:
        split = json.load(f)
    train_stories = split["train"]
    test_stories  = split["test"]

    for s in args.stories:
        if s not in test_stories:
            raise ValueError(f"Story '{s}' is not in the test split.")

    # ── embeddings ────────────────────────────────────────────────────────────
    print("Loading embeddings …", flush=True)
    X_train_raw = _load_embedding(
        _find_embedding(embedding_dir, args.subject_id, "train", args.method)
    )
    X_test_raw = _load_embedding(
        _find_embedding(embedding_dir, args.subject_id, "test", args.method)
    )
    X_train, X_test, x_mu, x_sigma = _zscore_X(X_train_raw, X_test_raw)
    print(f"  X_train {X_train.shape}  X_test {X_test.shape}", flush=True)

    # ── model pkl (voxel_indices + alphas_grid — no weights stored) ───────────
    model_path = os.path.join(
        result_dir, "models",
        f"{args.subject_id}_{args.method}",
        f"{args.subject_id}_{args.method}_model.pkl",
    )
    print(f"Loading model: {model_path}", flush=True)
    with open(model_path, "rb") as f:
        model_pkl = pickle.load(f)

    # ── raw text ──────────────────────────────────────────────────────────────
    # raw_text.pkl was serialised with ridge_utils classes; add code/ to path
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    raw_text_path = os.path.join(args.data_path, "raw_text.pkl")
    print(f"Loading raw_text: {raw_text_path}", flush=True)
    with open(raw_text_path, "rb") as f:
        raw_text = pickle.load(f)

    # ── refit top voxels (or load cached state) ───────────────────────────────
    state_pkl_path = args.state_pkl
    if state_pkl_path is None:
        state_pkl_path = os.path.join(
            result_dir, "models",
            f"{args.subject_id}_{args.method}",
            f"{args.subject_id}_{args.method}_top{args.n_candidate}_state.pkl",
        )

    if os.path.exists(state_pkl_path):
        print(f"Loading cached state: {state_pkl_path}", flush=True)
        with open(state_pkl_path, "rb") as f:
            state = pickle.load(f)
    else:
        print(f"Refitting top {args.n_candidate} voxels …", flush=True)
        state = refit_top_voxels(
            model_pkl, X_train, X_test,
            fmri_dir, train_stories, test_stories,
            n_candidate=args.n_candidate,
        )
        os.makedirs(os.path.dirname(state_pkl_path), exist_ok=True)
        with open(state_pkl_path, "wb") as f:
            pickle.dump(state, f)
        print(f"  Saved state → {state_pkl_path}", flush=True)

    cc = state["test_corrs"]
    print(
        f"  Refitted: mean={cc.mean():.4f}  "
        f"median={np.median(cc):.4f}  "
        f"top1%={np.percentile(cc, 99):.4f}",
        flush=True,
    )

    shap_dir = os.path.join(out_dir, args.subject_id, args.method, "shap")
    lime_dir = os.path.join(out_dir, args.subject_id, args.method, "lime")

    if not args.skip_shap:
        print("Running SHAP …", flush=True)
        run_shap(
            state, X_test, args.stories, raw_text,
            shap_dir, args.subject_id, args.method,
        )

    if not args.skip_lime:
        print("Running LIME …", flush=True)
        run_lime(
            state, X_test_raw, x_mu, x_sigma,
            test_stories, args.stories, raw_text,
            lime_dir, args.subject_id, args.method,
            top_k_voxels=args.top_k_voxels,
            word_window=args.word_window,
            peak_trs=args.peak_trs,
            n_lime_samples=args.lime_samples,
            n_features=args.n_features,
        )


if __name__ == "__main__":
    main()
