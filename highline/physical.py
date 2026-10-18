"""Cout physique apres une perte de balle : part du temps en course rapide (> 5,5 m/s) des 4 defenseurs les plus bas de A dans les 8 s qui suivent.

Vitesse sur 0,5 s calculee uniquement entre images detectees (les images extrapolees sont exclues)."""
import numpy as np
import pandas as pd

from . import config as cfg
from . import data as dat
from .stats import half_label, ols_cl, zs


def hsr_after_loss(rows, matches=None):
    out = []
    for mid in (matches or sorted(rows.match_id.unique())):
        sub = rows[rows.match_id == mid]
        if sub.empty:
            continue
        T = dat.load_match(mid)
        X, Y, D, team, grp = T["X"], T["Y"], T["D"], T["team"], T["grp"]
        for r in sub.itertuples():
            f = int(r.frame)
            if f + 85 >= T["n"]:
                continue
            ac = np.where((team == r.opp) & (grp != "GK") & np.isfinite(X[f]))[0]
            if len(ac) != 10:
                continue
            deep = ac[np.argsort(-(r.sgn * X[f, ac]))][:4]
            k = np.arange(f + 5, f + 81)
            hs = val = 0
            for j in deep:
                ok = (D[k, j] == 1) & (D[k - 5, j] == 1) & np.isfinite(X[k, j]) & np.isfinite(X[k - 5, j])
                v = np.hypot(X[k, j] - X[k - 5, j], Y[k, j] - Y[k - 5, j]) / 0.5
                hs += int((ok & (v > cfg.HSR_MS)).sum())
                val += int(ok.sum())
            out.append(dict(match_id=mid, frame=f, hsr_share=(100.0 * hs / val) if val else np.nan, valid=val / (4.0 * len(k))))
    return pd.DataFrame(out)


def analyse(rows, rep):
    rep.say("COUT PHYSIQUE : MONTER LA LIGNE COUTE-T-IL DES COURSES RAPIDES DE REPLI ? (4 defenseurs les plus bas de A, 8 s apres une perte, images detectees seulement)")
    h = hsr_after_loss(rows)
    e = rows.merge(h, on=["match_id", "frame"]).dropna(subset=["hsr_share", "line_A", "start", "n_press", "ay"]).copy()
    e = e[e.valid >= 0.3].copy()
    if len(e) < 100:
        rep.say("   trop peu de pertes exploitables.")
        return None
    for c in ["line_A", "start", "n_press", "ay", "valid"]:
        e["f_" + c] = zs(e[c])
    e["half"] = half_label(e.match_id.values)
    feats = ["f_line_A", "f_start", "f_n_press", "f_ay", "f_valid"]

    def eff(s):
        names, b, se = ols_cl(s.hsr_share.values, s[feats], s.match_id.values)
        i = names.index("f_line_A")
        return b[i], b[i] - 1.96 * se[i], b[i] + 1.96 * se[i]
    full = eff(e)
    hh = [eff(e[e.half == k])[0] if (e.half == k).sum() > 100 else np.nan for k in (0, 1)]
    strict = eff(e[e.valid >= 0.8]) if (e.valid >= 0.8).sum() > 200 else (np.nan,) * 3
    rep.say(f"{len(e)} pertes (>= 30 % d'images detectees) ; part moyenne du temps en course rapide : {e.hsr_share.mean():.1f} % ; 1 ecart-type de ligne = {e.line_A.std():.1f} m")
    rep.say(f"Effet d'un ecart-type de hauteur de ligne de A (points de %) : {full[0]:+.2f} [{full[1]:+.2f};{full[2]:+.2f}] ; moities {hh[0]:+.2f} / {hh[1]:+.2f} ; "
            f"part detectee >= 0,8 ({(e.valid >= 0.8).sum()} pertes) : {strict[0]:+.2f} [{strict[1]:+.2f};{strict[2]:+.2f}]")
    ok = full[1] > 0 and np.isfinite(hh).all() and np.sign(hh[0]) == np.sign(hh[1]) > 0 and strict[1] > 0
    verdict = "GO" if ok else ("A VOIR" if full[1] > 0 else "NO-GO")
    rep.say(f"VERDICT COUT PHYSIQUE : {verdict}")
    rep.say("Lecture : cout d'EFFORT local mesure sur les 8 s qui suivent la perte ; ni charge de match, ni risque de blessure. Les defenseurs hors champ de la camera sont exclus (selection possible).")
    return dict(n=len(e), full=full, halves=hh, strict=strict, mean=float(e.hsr_share.mean()), sd_line=float(e.line_A.std()), verdict=verdict)
