"""Table de moments defensifs : une ligne = une equipe qui DEFEND (l'adversaire a le ballon), toutes les 2 s.

Colonnes : line (hauteur de ligne = moyenne des 4 joueurs de champ les plus bas, m depuis son but), mid, att, d_dm, d_ma, length, width, area, stretch,
ball_x (m depuis son but), ball_ay, n_det (joueurs de champ detectes sur 10), line_sd (incertitude de la ligne, m), n_det_back4,
shot5/10/15/20 (l'adversaire tire dans les 5/10/15/20 s), puis own15, recov15, rec_x, age, free (outcomes), score_diff, minute (contexte)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from . import config as cfg
from . import data as dat
from . import reliability as rel
from .stats import half_label

HORIZONS = {"shot5": 50, "shot10": 100, "shot15": 150, "shot20": 200}
PRESS = {"no_pressure": 0, "low_pressure": 1, "medium_pressure": 2, "high_pressure": 3, "very_high_pressure": 4}


def match_table(mid, step=cfg.STEP):
    T = dat.load_match(mid)
    ev = dat.load_events(mid)
    X, Y, D, PER, BX, BY, team, grp = T["X"], T["Y"], T["D"], T["PER"], T["BX"], T["BY"], T["team"], T["grp"]
    n = T["n"]
    gx = dat.own_goal_sign(T)
    G = rel.gap_lengths(D)
    poss = dat.possession_array(ev, T["f0"], n)
    sf, st = dat.shot_frames(ev, T["f0"])
    rng = np.random.default_rng(int(mid) % (2 ** 32))
    rows = []
    for tm in np.unique(team):
        out_c = np.where((team == tm) & (grp != "GK"))[0]
        for t in range(100, n - 260, step):
            per = PER[t]
            if per == 0 or np.isnan(poss[t]) or poss[t] == tm or PER[t + 250] != per or PER[t - 100] != per:
                continue
            g = gx.get((per, tm))
            if g is None or not np.isfinite(g) or not np.isfinite(BX[t]):
                continue
            on = out_c[np.isfinite(X[t, out_c])]
            if len(on) != 10:
                continue
            sgn = 1.0 if g < 0 else -1.0
            xr = sgn * (X[t, on] - g)
            yy = Y[t, on]
            order = np.argsort(xr)
            srt = xr[order]
            try:
                area = ConvexHull(np.column_stack([xr, yy])).volume
            except Exception:
                area = np.nan
            row = dict(match_id=mid, team=int(tm), t=t, per=int(per), line=srt[:4].mean(), mid=srt[4:7].mean(), att=srt[7:].mean(),
                       length=srt[-1] - srt[0], width=yy.max() - yy.min(), area=area,
                       stretch=float(np.hypot(xr - xr.mean(), yy - yy.mean()).mean()),
                       ball_x=sgn * (BX[t] - g), ball_ay=abs(BY[t]), n_det=int(np.nansum(D[t, on] == 1)))
            row["d_dm"], row["d_ma"] = row["mid"] - row["line"], row["att"] - row["mid"]
            row["line_sd"] = rel.line_uncertainty(xr, G[t, on], k=100, rng=rng)
            row["n_det_back4"] = int(np.nansum(D[t, on][order[:4]] == 1))
            for k, h in HORIZONS.items():
                row[k] = int(((sf > t) & (sf <= t + h) & (st != tm)).any())
            rows.append(row)
    return pd.DataFrame(rows)


def add_outcomes(df):
    """own15 (tir de l'equipe en 15 s), recov15, rec_x (distance au but a la recuperation), age (s depuis qu'elle a eu le ballon), free (porteur peu presse)."""
    parts = []
    for mid, sub in df.groupby("match_id"):
        T = dat.load_match(mid)
        ev = dat.load_events(mid)
        n = T["n"]
        poss = dat.possession_array(ev, T["f0"], n)
        sf, st = dat.shot_frames(ev, T["f0"])
        gx = dat.own_goal_sign(T)
        BX = T["BX"]
        own, rec, rx, age, free, epv0, org0 = [], [], [], [], [], [], []
        pp = ev[ev.event_type == "player_possession"].dropna(subset=["frame_start"]).copy()
        pp["fs"] = pp.frame_start.astype(int) - T["f0"]
        pp = pp.sort_values("fs")
        pfs = pp.fs.values
        ppress = pp.overall_pressure_start.map(PRESS).values if "overall_pressure_start" in pp else np.full(len(pp), np.nan)
        pepv = pd.to_numeric(pp["possession_epv_for_start"], errors="coerce").values if "possession_epv_for_start" in pp else np.full(len(pp), np.nan)
        porg = pp.organised_defense.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float).values if "organised_defense" in pp else np.full(len(pp), np.nan)
        for r in sub.itertuples():
            t, tm = int(r.t), int(r.team)
            g = gx.get((r.per, tm))
            sgn = 1.0 if (g is not None and g < 0) else -1.0
            own.append(int(((sf > t) & (sf <= t + 150) & (st == tm)).any()))
            k = int(np.searchsorted(pfs, t, side="right")) - 1
            free.append(float(ppress[k] <= 1) if k >= 0 and np.isfinite(ppress[k]) else np.nan)
            epv0.append(float(pepv[k]) if k >= 0 else np.nan)
            org0.append(float(porg[k]) if k >= 0 else np.nan)
            back = np.where(poss[max(0, t - 300):t + 1][::-1] == tm)[0]
            age.append(float(back[0]) / 10.0 if len(back) else 30.0)
            idx = np.where(poss[t + 1:t + 151] == tm)[0]
            rec.append(int(len(idx) > 0))
            if len(idx) and g is not None and np.isfinite(BX[t + 1 + idx[0]]):
                rx.append(float(sgn * (BX[t + 1 + idx[0]] - g)))
            else:
                rx.append(np.nan)
        s = sub.copy()
        s["own15"], s["recov15"], s["rec_x"], s["age"], s["free"] = own, rec, rx, age, free
        s["epv0"], s["org0"] = epv0, org0
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def _score_lookup(mid):
    T = dat.load_match(mid)
    ev = dat.load_events(mid)
    e = ev.dropna(subset=["team_score", "opponent_team_score", "frame_start", "team_id", "minute_start"]).copy()
    e["fs"] = e.frame_start.astype(int) - T["f0"]
    e = e.sort_values("fs")
    return (e.fs.values, e.team_id.values.astype(int), (e.team_score - e.opponent_team_score).values.astype(float), e.minute_start.values.astype(float))


def lookup(ctx, team, frame):
    """Ecart de score du point de vue de `team` et minute de jeu, lus sur le dernier evenement connu avant `frame`."""
    fs, tid, dif, mn = ctx
    team, frame = np.asarray(team), np.asarray(frame)
    k = np.searchsorted(fs, frame, side="right") - 1
    ok = k >= 0
    k = np.clip(k, 0, None)
    sc = np.where(tid[k] == team, dif[k], -dif[k])
    return np.where(ok, sc, np.nan), np.where(ok, mn[k], np.nan)


def add_context(df, col_team, col_frame):
    """Ajoute score_diff (avance de l'equipe `col_team`), minute, score_state et min_bin."""
    out = df.copy()
    out["score_diff"], out["minute"] = np.nan, np.nan
    for mid in out.match_id.unique():
        m = (out.match_id == mid).values
        s, mn = lookup(_score_lookup(mid), out.loc[m, col_team].values.astype(int), out.loc[m, col_frame].values.astype(int))
        out.loc[m, "score_diff"], out.loc[m, "minute"] = s, mn
    out["score_state"] = pd.cut(out.score_diff, [-np.inf, -0.5, 0.5, np.inf], labels=["derriere", "egalite", "devant"])
    out["min_bin"] = pd.cut(out.minute, [-1, 30, 60, 500], labels=["0-30", "30-60", "60+"])
    return out


def residual_line(df, by_team_match=True):
    """Hauteur de ligne moins ce qu'elle est d'habitude pour la meme position de balle (ajustement quadratique par equipe et par match)."""
    res = pd.Series(np.nan, index=df.index)
    groups = df.groupby(["match_id", "team"]) if by_team_match else [(0, df)]
    for _, g in groups:
        if len(g) < 100:
            continue
        A = np.column_stack([np.ones(len(g)), g.ball_x, g.ball_x ** 2, g.ball_ay])
        beta = np.linalg.lstsq(A, g.line.values, rcond=None)[0]
        res[g.index] = g.line.values - A @ beta
    return res


#: colonnes qui doivent exister dans le cache ; sinon il est reconstruit (cache ecrit par une version anterieure)
REQUIRED = ["epv0", "org0", "score_diff", "minute"]


def build(matches=None, cache="cache/frames.csv", refresh=False, verbose=True):
    """Construit (ou relit) la table complete de moments defensifs pour les matchs demandes.

    Le cache depend de la reference de but (data.goal_reference()) : le nom du fichier doit la porter (voir pipeline)."""
    ids = list(matches or cfg.ALL_MATCHES)
    cache = Path(cache)
    if cache.exists() and not refresh:
        df = pd.read_csv(cache)
        if set(ids) <= set(df.match_id.unique()) and all(c in df.columns for c in REQUIRED):
            return df[df.match_id.isin(ids)].reset_index(drop=True)
    parts = []
    for mid in ids:
        r = match_table(mid)
        if verbose:
            print(f"[frames] match {mid} : {len(r)} moments", flush=True)
        parts.append(r)
    df = pd.concat(parts, ignore_index=True)
    df = add_outcomes(df)
    df = add_context(df, "team", "t")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df
