"""Couche de fiabilite : quelle confiance accorder a une position extrapolee, donc a une hauteur de ligne ?

Un joueur dont is_detected est faux a une position estimee. Son erreur typique (sigma) croit avec la duree du trou d'extrapolation ;
on simule des positions plausibles et on rapporte l'incertitude de la ligne (4 defenseurs de champ les plus bas)."""
import numpy as np

from . import config as cfg


def gap_lengths(D):
    """Longueur (en images) du trou d'extrapolation qui contient chaque image ; 0 si detecte."""
    n, m = D.shape
    G = np.zeros((n, m), dtype=np.int32)
    for j in range(m):
        miss = (D[:, j] == 0).astype(int)
        idx = np.flatnonzero(np.diff(np.r_[0, miss, 0]))
        for s, e in zip(idx[::2], idx[1::2]):
            G[s:e, j] = e - s
    return G


def sigma_of(gap_frames):
    """Ecart-type de l'erreur (m) d'une position selon la duree du trou, en images a 10 Hz."""
    return np.interp(np.asarray(gap_frames) / cfg.FPS, cfg.SIG_T, cfg.SIG_V)


def radius90(gap_frames):
    """Rayon a 90 % de l'erreur de position (m)."""
    return cfg.R90 * sigma_of(gap_frames)


def line_of(dist_to_goal):
    """Hauteur de ligne = moyenne des 4 distances au but les plus petites (joueurs de champ)."""
    return float(np.sort(dist_to_goal)[:4].mean())


def line_uncertainty(dist_to_goal, gaps, k=100, rng=None):
    """Incertitude (ecart-type, m) de la hauteur de ligne : k tirages de positions plausibles pour les 10 joueurs de champ."""
    rng = rng or np.random.default_rng(0)
    d = np.asarray(dist_to_goal, float)
    sd = sigma_of(gaps)
    sims = d[None, :] + rng.normal(size=(k, len(d))) * sd[None, :]
    lines = np.sort(sims, axis=1)[:, :4].mean(axis=1)
    return float(lines.std())
