"""Outils statistiques : regressions a erreurs groupees par match, AUC, bootstrap par match."""
import numpy as np
import pandas as pd


def zs(s):
    s = pd.Series(s, dtype=float)
    sd = s.std()
    return (s - s.mean()) / (sd if sd > 0 else 1.0)


def logit(y, X, groups):
    """Regression logistique (IRLS), erreurs robustes groupees. Retourne noms, coefficients, erreurs-types."""
    Xm = np.column_stack([np.ones(len(y))] + [np.asarray(X[c], float) for c in X.columns])
    y = np.asarray(y, float)
    b = np.zeros(Xm.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-np.clip(Xm @ b, -30, 30)))
        step = np.linalg.solve(Xm.T @ (Xm * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(b)), Xm.T @ (y - p))
        b = b + step
        if np.abs(step).max() < 1e-7:
            break
    p = 1 / (1 + np.exp(-np.clip(Xm @ b, -30, 30)))
    bread = np.linalg.inv(Xm.T @ (Xm * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(b)))
    meat = np.zeros_like(bread)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        s = (Xm[groups == g] * (y - p)[groups == g][:, None]).sum(0)
        meat += np.outer(s, s)
    se = np.sqrt(np.abs(np.diag(bread @ meat @ bread)))
    return ["const"] + list(X.columns), b, se


def or_ci(names, b, se, var):
    i = names.index(var)
    return float(np.exp(b[i])), float(np.exp(b[i] - 1.96 * se[i])), float(np.exp(b[i] + 1.96 * se[i]))


def ols_cl(y, X, groups):
    """Moindres carres, erreurs robustes groupees. Retourne noms, coefficients, erreurs-types."""
    Xm = np.column_stack([np.ones(len(y))] + [np.asarray(X[c], float) for c in X.columns])
    y = np.asarray(y, float)
    bread = np.linalg.inv(Xm.T @ Xm + 1e-9 * np.eye(Xm.shape[1]))
    b = bread @ Xm.T @ y
    e = y - Xm @ b
    meat = np.zeros_like(bread)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        s = (Xm[groups == g] * e[groups == g][:, None]).sum(0)
        meat += np.outer(s, s)
    se = np.sqrt(np.abs(np.diag(bread @ meat @ bread)))
    return ["const"] + list(X.columns), b, se


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def half_label(match_ids):
    """Deux moities de matchs (rang pair / impair des identifiants) pour tester la replication."""
    ids = sorted(pd.unique(match_ids))
    rank = {m: i % 2 for i, m in enumerate(ids)}
    return pd.Series(match_ids).map(rank).values


def pct(x, q=(2.5, 97.5)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return tuple(np.percentile(x, q)) if len(x) > 30 else (np.nan, np.nan)
