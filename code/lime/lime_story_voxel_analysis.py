#!/usr/bin/env python3
"""
LIME analysis for story-specific voxel interpretation with fine-tuned BERT.

Workflow:
1. Load merged ridge results for `finetuned_lora`.
2. Refit only a candidate subset of high-performing voxels so we can recover
   voxel weights even if the saved ridge model did not include `--save-weights`.
3. For each requested test story, compute story-specific voxel correlations and
   choose the top-performing voxels for interpretation.
4. For each selected voxel, identify the peak predicted response window and run
   LIME on a local word window around that peak.

Important approximation for tractability:
- We do not require the missing LoRA checkpoint.
- Instead, we operate directly on the saved TR-level delayed embeddings from
  Hugging Face and use raw_text timing to map each word to its nearest trimmed
  TR index.
- Masking a word zeros that word's contribution across the corresponding delay
  blocks in the saved design matrix. This preserves the final ridge feature
  space while giving an interpretable local approximation.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception as exc:  # pragma: no cover - handled at runtime
    LimeTabularExplainer = None
    LIME_IMPORT_ERROR = exc
else:
    LIME_IMPORT_ERROR = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_ROOT = os.path.join(REPO_ROOT, "code")
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from preprocessing_utils import DELAYS, TRIM_END, TRIM_START
from ridge_regression.run_ridge import SVDRidgeSolver, load_embedding, load_fmri_chunk, zscore_X


@dataclass
class RefitResult:
    voxel_indices: np.ndarray
    weights: np.ndarray
    story_corrs: Dict[str, np.ndarray]
    story_preds: Dict[str, np.ndarray]
    story_truth: Dict[str, np.ndarray]
    story_slices: Dict[str, slice]
    alphas: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIME for selected test stories and top voxels.")
    parser.add_argument("--subject", required=True, help="Subject folder, e.g. subject2 or subject3.")
    parser.add_argument("--subject-id", required=True, help="Subject id, e.g. s2 or s3.")
    parser.add_argument("--stories", nargs="+", required=True,
                        help="Test stories to interpret, e.g. onapproachtopluto marryamanwholoveshismother")
    parser.add_argument("--method", default="finetuned_lora")
    parser.add_argument("--data-path", default="data",
                        help="Path containing raw_text.pkl and subject folders.")
    parser.add_argument("--embedding-dir", default="data/embeddings")
    parser.add_argument("--split-path", default="data/train_test_split.json")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--figure-dir", default="results/figures/lime",
                        help="Directory for selected report-ready LIME figures.")
    parser.add_argument("--no-report-figures", action="store_true",
                        help="Do not copy top-ranked story figures into --figure-dir.")
    parser.add_argument("--candidate-voxels", type=int, default=256,
                        help="Top overall voxels to refit before story-specific ranking.")
    parser.add_argument("--top-k-voxels", type=int, default=3,
                        help="How many voxels to interpret per story.")
    parser.add_argument("--word-window", type=int, default=48,
                        help="Number of local words given to LIME around the peak response.")
    parser.add_argument("--peak-trs", type=int, default=5,
                        help="Number of peak TRs whose average predicted response is explained.")
    parser.add_argument("--lime-samples", type=int, default=300)
    parser.add_argument("--num-features", type=int, default=12,
                        help="Top features displayed in each LIME explanation.")
    parser.add_argument("--delays", nargs="+", type=int, default=DELAYS,
                        help="Delay values used when creating the saved embeddings.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def resolve_repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def require_lime() -> None:
    if LimeTabularExplainer is None:
        raise ImportError(
            "lime is not available in this environment. Install it in your lab "
            "environment before running this script."
        ) from LIME_IMPORT_ERROR


def load_raw_text(data_path: str):
    with open(os.path.join(data_path, "raw_text.pkl"), "rb") as f:
        return pickle.load(f)


def load_split(split_path: str) -> Dict[str, List[str]]:
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_merged_model(result_dir: str, sid: str, method: str) -> Dict:
    model_path = os.path.join(result_dir, "models", f"{sid}_{method}", f"{sid}_{method}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Merged ridge model not found: {model_path}. "
            f"Run merge_ridge_results.py for {sid} {method} first."
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def compute_story_slices(test_stories: Sequence[str], test_maps: Dict[str, np.ndarray]) -> Dict[str, slice]:
    story_slices: Dict[str, slice] = {}
    start = 0
    for story in test_stories:
        end = start + int(test_maps[story].shape[0])
        story_slices[story] = slice(start, end)
        start = end
    return story_slices


def zscore_with_train_stats(X_train: np.ndarray, X_other: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0, keepdims=True)
    sigma = X_train.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32), ((X_other - mu) / sigma).astype(np.float32)


def load_train_test_embeddings(embedding_dir: str, sid: str, method: str) -> Tuple[np.ndarray, np.ndarray]:
    train_path = os.path.join(embedding_dir, f"{sid}_train_{method}_embeddings.npz")
    test_path = os.path.join(embedding_dir, f"{sid}_test_{method}_embeddings.npz")
    return load_embedding(train_path), load_embedding(test_path)


def refit_candidate_voxels(
    model_obj: Dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    subject: str,
    data_path: str,
    train_stories: Sequence[str],
    test_stories: Sequence[str],
    candidate_voxels: int,
) -> RefitResult:
    voxel_indices_all = np.asarray(model_obj["voxel_indices"], dtype=np.int32)
    test_corrs_all = np.asarray(model_obj["test_corrs"], dtype=np.float32)
    alphas_grid = np.asarray(model_obj["alphas_grid"], dtype=np.float32)

    ranked = np.argsort(test_corrs_all)[::-1]
    chosen = ranked[: min(candidate_voxels, ranked.shape[0])]
    candidate_ids = np.sort(voxel_indices_all[chosen]).astype(np.int32)

    fmri_dir = os.path.join(data_path, subject)
    train_maps = {s: np.load(os.path.join(fmri_dir, f"{s}.npy"), mmap_mode="r") for s in train_stories}
    test_maps = {s: np.load(os.path.join(fmri_dir, f"{s}.npy"), mmap_mode="r") for s in test_stories}

    train_lengths = [train_maps[s].shape[0] for s in train_stories]
    test_lengths = [test_maps[s].shape[0] for s in test_stories]
    train_total = int(sum(train_lengths))
    test_total = int(sum(test_lengths))

    Y_train = load_fmri_chunk(train_maps, train_stories, train_lengths, train_total, candidate_ids)
    Y_test = load_fmri_chunk(test_maps, test_stories, test_lengths, test_total, candidate_ids)

    solver = SVDRidgeSolver(X_train, X_test)
    selected_alphas = solver.select_alpha(Y_train, alphas_grid)
    _, weights = solver.fit_predict(Y_train, Y_test, selected_alphas)
    preds = np.asarray(X_test, dtype=np.float32) @ weights

    story_slices = compute_story_slices(test_stories, test_maps)
    story_corrs: Dict[str, np.ndarray] = {}
    story_preds: Dict[str, np.ndarray] = {}
    story_truth: Dict[str, np.ndarray] = {}

    for story in test_stories:
        sl = story_slices[story]
        y_story = Y_test[sl]
        pred_story = preds[sl]
        y_story_z = (y_story - y_story.mean(axis=0, keepdims=True)) / np.maximum(y_story.std(axis=0, keepdims=True), 1e-8)
        pred_story_z = (pred_story - pred_story.mean(axis=0, keepdims=True)) / np.maximum(pred_story.std(axis=0, keepdims=True), 1e-8)
        story_corrs[story] = (y_story_z * pred_story_z).mean(axis=0).astype(np.float32)
        story_preds[story] = pred_story.astype(np.float32)
        story_truth[story] = y_story.astype(np.float32)

    return RefitResult(
        voxel_indices=candidate_ids,
        weights=weights.astype(np.float32),
        story_corrs=story_corrs,
        story_preds=story_preds,
        story_truth=story_truth,
        story_slices=story_slices,
        alphas=selected_alphas.astype(np.float32),
    )


def split_stacked_embeddings_by_story(
    X_test_raw: np.ndarray,
    story_slices: Dict[str, slice],
) -> Dict[str, np.ndarray]:
    story_embeddings: Dict[str, np.ndarray] = {}
    for story, sl in story_slices.items():
        story_embeddings[story] = np.asarray(X_test_raw[sl], dtype=np.float32).copy()
    return story_embeddings


def word_index_to_trimmed_tr(story: str, raw_text, word_idx: int) -> int | None:
    tr_times = np.asarray(raw_text[story].tr_times, dtype=np.float32)
    trim_len = len(tr_times) - TRIM_START - TRIM_END
    trimmed_tr_times = tr_times[TRIM_START:TRIM_START + trim_len]
    if trimmed_tr_times.size == 0:
        return None
    word_time = float(raw_text[story].data_times[word_idx])
    return int(np.argmin(np.abs(trimmed_tr_times - word_time)))


def zero_word_contribution_in_delayed_matrix(
    X_story: np.ndarray,
    trimmed_tr_idx: int,
    delays: Sequence[int],
) -> None:
    n_delays = len(delays)
    if n_delays == 0:
        return
    block_dim = X_story.shape[1] // n_delays
    if block_dim * n_delays != X_story.shape[1]:
        raise ValueError("Embedding dimension is not divisible by the number of delays.")

    for block_idx, delay in enumerate(delays):
        row_idx = trimmed_tr_idx + int(delay)
        if 0 <= row_idx < X_story.shape[0]:
            c0 = block_idx * block_dim
            c1 = c0 + block_dim
            X_story[row_idx, c0:c1] = 0.0


def get_peak_summary(
    story: str,
    raw_text,
    pred_story: np.ndarray,
    peak_trs: int,
    word_window: int,
) -> Dict[str, int]:
    pred_signal = pred_story.copy()
    ranked_trs = np.argsort(pred_signal)[::-1]
    top_trs = np.sort(ranked_trs[: min(peak_trs, ranked_trs.shape[0])]).astype(int)

    center_tr = int(top_trs[np.argmax(pred_signal[top_trs])])
    tr_time = float(raw_text[story].tr_times[TRIM_START + center_tr])
    word_times = np.asarray(raw_text[story].data_times, dtype=np.float32)
    center_word = int(np.argmin(np.abs(word_times - tr_time)))

    half_window = word_window // 2
    word_start = max(0, center_word - half_window)
    word_end = min(len(raw_text[story].data), word_start + word_window)
    word_start = max(0, word_end - word_window)

    return {
        "center_tr": center_tr,
        "word_center": center_word,
        "word_start": word_start,
        "word_end": word_end,
        "peak_tr_min": int(top_trs[0]),
        "peak_tr_max": int(top_trs[-1]),
    }


def build_lime_prediction_fn(
    story: str,
    raw_text,
    base_story_embeddings: np.ndarray,
    local_slice: slice,
    weight_vector: np.ndarray,
    x_mu: np.ndarray,
    x_sigma: np.ndarray,
    target_trs: np.ndarray,
    delays: Sequence[int],
):
    local_len = local_slice.stop - local_slice.start
    word_indices = np.arange(local_slice.start, local_slice.stop, dtype=int)
    word_to_tr = [word_index_to_trimmed_tr(story, raw_text, idx) for idx in word_indices]

    def predict_fn(mask_matrix: np.ndarray) -> np.ndarray:
        outputs = np.zeros(mask_matrix.shape[0], dtype=np.float32)
        for row_idx, mask in enumerate(mask_matrix):
            masked_story = base_story_embeddings.copy()
            keep = mask[:local_len] >= 0.5
            for keep_flag, trimmed_tr_idx in zip(keep, word_to_tr):
                if keep_flag or trimmed_tr_idx is None:
                    continue
                zero_word_contribution_in_delayed_matrix(masked_story, trimmed_tr_idx, delays)

            x_story_z = ((masked_story - x_mu) / x_sigma).astype(np.float32)
            pred_signal = x_story_z @ weight_vector
            outputs[row_idx] = float(pred_signal[target_trs].mean())
        return outputs

    return predict_fn


def aggregate_lime_terms(explanation_list: List[Tuple[str, float]]) -> pd.DataFrame:
    rows = []
    for feature_name, score in explanation_list:
        token = feature_name.split(":", 1)[1] if ":" in feature_name else feature_name
        rows.append({"feature": feature_name, "token": token, "score": float(score)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    grouped = (
        df.groupby("token", as_index=False)
        .agg(score=("score", "sum"), abs_score=("score", lambda s: float(np.abs(s).sum())))
        .sort_values("abs_score", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def save_response_plot(out_path: str, pred_signal: np.ndarray, truth_signal: np.ndarray, peak_info: Dict[str, int]) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    trs = np.arange(pred_signal.shape[0])
    ax.plot(trs, truth_signal, label="Observed", color="#4C78A8", linewidth=1.4, alpha=0.8)
    ax.plot(trs, pred_signal, label="Predicted", color="#E45756", linewidth=1.6)
    ax.axvspan(peak_info["peak_tr_min"], peak_info["peak_tr_max"], color="#F3A712", alpha=0.25)
    ax.set_xlabel("TR within story")
    ax.set_ylabel("Voxel response (z-scored)")
    ax.set_title("Story-level voxel response and interpreted peak window")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_lime_plot(out_path: str, lime_df: pd.DataFrame, title: str, max_rows: int = 12) -> None:
    plot_df = lime_df.head(max_rows).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.iloc[::-1]
    colors = ["#54A24B" if val > 0 else "#E45756" for val in plot_df["score"]]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(plot_df["token"], plot_df["score"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME weight")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def copy_report_figures(
    response_plot: str,
    lime_plot: str,
    figure_dir: str,
    sid: str,
    story: str,
    voxel_id: int,
) -> Tuple[str, str]:
    os.makedirs(figure_dir, exist_ok=True)
    base = f"{sid}_{story}_voxel{voxel_id:05d}"
    report_response = os.path.join(figure_dir, f"{base}_response.png")
    report_lime = os.path.join(figure_dir, f"{base}_lime.png")
    shutil.copy2(response_plot, report_response)
    shutil.copy2(lime_plot, report_lime)
    return os.path.abspath(report_response), os.path.abspath(report_lime)


def run_story_lime(
    story: str,
    raw_text,
    base_story_embeddings: np.ndarray,
    refit: RefitResult,
    x_mu: np.ndarray,
    x_sigma: np.ndarray,
    out_dir: str,
    figure_dir: str,
    args: argparse.Namespace,
) -> None:
    story_corr = refit.story_corrs[story]
    ranked = np.argsort(story_corr)[::-1]
    top_k = ranked[: min(args.top_k_voxels, ranked.shape[0])]

    story_summary_rows = []

    for rank_idx, local_voxel_pos in enumerate(top_k, start=1):
        voxel_id = int(refit.voxel_indices[local_voxel_pos])
        voxel_cc = float(story_corr[local_voxel_pos])
        pred_signal = refit.story_preds[story][:, local_voxel_pos]
        truth_signal = refit.story_truth[story][:, local_voxel_pos]
        peak_info = get_peak_summary(story, raw_text, pred_signal, args.peak_trs, args.word_window)
        local_slice = slice(peak_info["word_start"], peak_info["word_end"])
        local_words = list(raw_text[story].data[local_slice])
        feature_names = [f"{peak_info['word_start'] + idx}:{word}" for idx, word in enumerate(local_words)]
        target_trs = np.arange(peak_info["peak_tr_min"], peak_info["peak_tr_max"] + 1)

        explainer = LimeTabularExplainer(
            training_data=np.vstack([
                np.ones(len(local_words), dtype=np.float32),
                np.zeros(len(local_words), dtype=np.float32),
            ]),
            feature_names=feature_names,
            mode="regression",
            discretize_continuous=False,
            random_state=args.random_state,
        )

        predict_fn = build_lime_prediction_fn(
            story=story,
            raw_text=raw_text,
            base_story_embeddings=base_story_embeddings,
            local_slice=local_slice,
            weight_vector=refit.weights[:, local_voxel_pos],
            x_mu=x_mu,
            x_sigma=x_sigma,
            target_trs=target_trs,
            delays=args.delays,
        )

        explanation = explainer.explain_instance(
            data_row=np.ones(len(local_words), dtype=np.float32),
            predict_fn=predict_fn,
            num_features=min(args.num_features, len(local_words)),
            num_samples=args.lime_samples,
        )
        explanation_list = explanation.as_list()
        lime_df = aggregate_lime_terms(explanation_list)

        prefix = os.path.join(out_dir, f"{story}_voxel{voxel_id:05d}_rank{rank_idx}")
        response_plot = f"{prefix}_response.png"
        lime_plot = f"{prefix}_lime.png"
        lime_csv = f"{prefix}_lime.csv"

        save_response_plot(response_plot, pred_signal, truth_signal, peak_info)
        save_lime_plot(
            lime_plot,
            lime_df,
            title=f"{story} | voxel {voxel_id} | CC={voxel_cc:.3f}",
            max_rows=args.num_features,
        )
        lime_df.to_csv(lime_csv, index=False)

        report_response_plot = ""
        report_lime_plot = ""
        if rank_idx == 1 and not args.no_report_figures:
            report_response_plot, report_lime_plot = copy_report_figures(
                response_plot=response_plot,
                lime_plot=lime_plot,
                figure_dir=figure_dir,
                sid=args.subject_id,
                story=story,
                voxel_id=voxel_id,
            )

        top_tokens = ", ".join(lime_df["token"].head(5).tolist())
        story_summary_rows.append({
            "story": story,
            "voxel_id": voxel_id,
            "story_cc": voxel_cc,
            "rank_within_story": rank_idx,
            "peak_tr_min": peak_info["peak_tr_min"],
            "peak_tr_max": peak_info["peak_tr_max"],
            "word_start": peak_info["word_start"],
            "word_end": peak_info["word_end"],
            "top_tokens": top_tokens,
            "response_plot": os.path.abspath(response_plot),
            "lime_plot": os.path.abspath(lime_plot),
            "lime_csv": os.path.abspath(lime_csv),
            "report_response_plot": report_response_plot,
            "report_lime_plot": report_lime_plot,
        })

    summary_df = pd.DataFrame(story_summary_rows).sort_values("story_cc", ascending=False)
    summary_df.to_csv(os.path.join(out_dir, f"{story}_lime_summary.csv"), index=False)


def main() -> None:
    require_lime()
    args = parse_args()

    data_path = resolve_repo_path(args.data_path)
    embedding_dir = resolve_repo_path(args.embedding_dir)
    split_path = resolve_repo_path(args.split_path)
    result_dir = resolve_repo_path(args.result_dir)
    figure_dir = resolve_repo_path(args.figure_dir)
    raw_text = load_raw_text(data_path)
    split = load_split(split_path)
    train_stories = split["train"]
    test_stories = split["test"]

    for story in args.stories:
        if story not in test_stories:
            raise ValueError(f"Story '{story}' is not in the test split.")

    model_obj = load_merged_model(result_dir, args.subject_id, args.method)
    X_train_raw, X_test_raw = load_train_test_embeddings(embedding_dir, args.subject_id, args.method)
    X_train, X_test = zscore_X(X_train_raw, X_test_raw)
    x_mu, x_sigma, _ = zscore_with_train_stats(np.asarray(X_train_raw, dtype=np.float32), np.asarray(X_test_raw, dtype=np.float32))

    refit = refit_candidate_voxels(
        model_obj=model_obj,
        X_train=np.asarray(X_train, dtype=np.float32),
        X_test=np.asarray(X_test, dtype=np.float32),
        subject=args.subject,
        data_path=data_path,
        train_stories=train_stories,
        test_stories=test_stories,
        candidate_voxels=args.candidate_voxels,
    )

    story_embeddings = split_stacked_embeddings_by_story(
        np.asarray(X_test_raw, dtype=np.float32),
        refit.story_slices,
    )

    out_root = os.path.join(result_dir, "interpretation", args.subject_id, args.method, "lime")
    os.makedirs(out_root, exist_ok=True)

    for story in args.stories:
        story_out_dir = os.path.join(out_root, story)
        os.makedirs(story_out_dir, exist_ok=True)
        run_story_lime(
            story=story,
            raw_text=raw_text,
            base_story_embeddings=story_embeddings[story],
            refit=refit,
            x_mu=x_mu,
            x_sigma=x_sigma,
            out_dir=story_out_dir,
            figure_dir=figure_dir,
            args=args,
        )

    print(f"Saved LIME outputs under: {out_root}")
    if not args.no_report_figures:
        print(f"Saved selected report figures under: {figure_dir}")


if __name__ == "__main__":
    main()
