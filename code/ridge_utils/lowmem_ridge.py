import numpy as np


def _zscore_cols(arr):
    mean = arr.mean(axis=0, dtype=np.float32, keepdims=True)
    std = arr.std(axis=0, dtype=np.float32, keepdims=True)
    std[std < 1e-6] = 1.0
    return (arr - mean) / std


def _column_corr(y_true, y_pred):
    z_true = _zscore_cols(y_true)
    z_pred = _zscore_cols(y_pred)
    corrs = (z_true * z_pred).mean(axis=0)
    corrs = np.nan_to_num(corrs, nan=0.0, posinf=0.0, neginf=0.0)
    return corrs.astype(np.float32, copy=False)


def prepare_dual_ridge_kernels(X_train, X_test, holdout_frac=0.1, min_val_rows=20):
    """Precompute kernel matrices reused across all voxel chunks.

    Uses a contiguous holdout at the end of training for alpha selection.
    """
    n_train = X_train.shape[0]
    n_val = max(int(n_train * holdout_frac), min_val_rows)
    n_val = min(n_val, n_train - 1)
    n_fit = n_train - n_val

    X_fit = X_train[:n_fit]
    X_val = X_train[n_fit:]

    K_fit = np.dot(X_fit, X_fit.T).astype(np.float32, copy=False)
    K_val_fit = np.dot(X_val, X_fit.T).astype(np.float32, copy=False)

    K_full = np.dot(X_train, X_train.T).astype(np.float32, copy=False)
    K_test_full = np.dot(X_test, X_train.T).astype(np.float32, copy=False)

    return {
        "n_fit": n_fit,
        "K_fit": K_fit,
        "K_val_fit": K_val_fit,
        "K_full": K_full,
        "K_test_full": K_test_full,
    }


def ridge_corrs_dual_single_alpha(Y_train, Y_test, alphas, prep):
    """Select one alpha on training holdout, then score test correlations per voxel."""
    n_fit = prep["n_fit"]
    K_fit = prep["K_fit"]
    K_val_fit = prep["K_val_fit"]
    K_full = prep["K_full"]
    K_test_full = prep["K_test_full"]

    Y_fit = Y_train[:n_fit]
    Y_val = Y_train[n_fit:]

    eye_fit = np.eye(K_fit.shape[0], dtype=np.float32)
    best_alpha = None
    best_score = -np.inf

    for alpha in alphas:
        Kreg = K_fit + np.float32(alpha) * eye_fit
        coef = np.linalg.solve(Kreg, Y_fit)
        pred_val = np.dot(K_val_fit, coef)
        score = float(np.mean(_column_corr(Y_val, pred_val)))
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)

    eye_full = np.eye(K_full.shape[0], dtype=np.float32)
    Kreg_full = K_full + np.float32(best_alpha) * eye_full
    coef_full = np.linalg.solve(Kreg_full, Y_train)
    pred_test = np.dot(K_test_full, coef_full)
    corrs = _column_corr(Y_test, pred_test)

    return corrs, best_alpha