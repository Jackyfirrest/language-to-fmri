import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr, splu


def _as_csr32(X):
    if sparse.issparse(X):
        return X.tocsr().astype(np.float32, copy=False)
    return sparse.csr_matrix(np.asarray(X, dtype=np.float32))


def _zscore_cols(arr):
    mean = arr.mean(axis=0, dtype=np.float32, keepdims=True)
    std = arr.std(axis=0, dtype=np.float32, keepdims=True)
    std[std < 1e-6] = 1.0
    return (arr - mean) / std


def _column_corr(y_true, y_pred):
    z_true = _zscore_cols(np.asarray(y_true, dtype=np.float32))
    z_pred = _zscore_cols(np.asarray(y_pred, dtype=np.float32))
    corrs = (z_true * z_pred).mean(axis=0)
    return np.nan_to_num(corrs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _fit_sparse_ridge(X, Y, alpha, exact_feature_cutoff=4000, tol=1e-4, maxiter=300):
    X = _as_csr32(X)
    Y = np.asarray(Y, dtype=np.float32)
    n_samples, n_features = X.shape

    if n_features <= exact_feature_cutoff:
        XtX = (X.T @ X).tocsc()
        XtX = XtX + sparse.eye(n_features, format="csc", dtype=np.float32) * np.float32(alpha)
        XtY = X.T @ Y
        XtY = np.asarray(XtY, dtype=np.float32)
        lu = splu(XtX)
        W = lu.solve(XtY)
        return np.asarray(W, dtype=np.float32)

    # Fallback for very wide feature spaces: ridge-regularized sparse least squares.
    W = np.empty((n_features, Y.shape[1]), dtype=np.float32)
    damp = float(np.sqrt(alpha))
    for j in range(Y.shape[1]):
        sol = lsmr(X, Y[:, j], damp=damp, atol=tol, btol=tol, maxiter=maxiter)[0]
        W[:, j] = sol.astype(np.float32, copy=False)
    return W


def choose_alpha_sparse(X_train, Y_train, alphas, holdout_frac=0.1, sample_voxels=16):
    X_train = _as_csr32(X_train)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    n_rows = X_train.shape[0]
    n_val = max(1, int(n_rows * holdout_frac))
    n_val = min(n_val, n_rows - 1)
    n_fit = n_rows - n_val

    X_fit = X_train[:n_fit]
    X_val = X_train[n_fit:]
    Y_fit = Y_train[:n_fit]
    Y_val = Y_train[n_fit:]

    n_vox = Y_train.shape[1]
    sample_voxels = min(sample_voxels, n_vox)
    sample_idx = np.linspace(0, n_vox - 1, sample_voxels, dtype=int)

    best_alpha = float(alphas[0])
    best_score = -np.inf

    for alpha in alphas:
        W = _fit_sparse_ridge(X_fit, Y_fit[:, sample_idx], alpha)
        pred = X_val @ W
        score = float(np.mean(_column_corr(Y_val[:, sample_idx], pred)))
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)

    return best_alpha


def ridge_corrs_sparse(X_train, Y_train, X_test, Y_test, alpha):
    X_train = _as_csr32(X_train)
    X_test = _as_csr32(X_test)
    Y_train = np.asarray(Y_train, dtype=np.float32)
    Y_test = np.asarray(Y_test, dtype=np.float32)

    W = _fit_sparse_ridge(X_train, Y_train, alpha)
    pred = X_test @ W
    corrs = _column_corr(Y_test, pred)
    return corrs, W