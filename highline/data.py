"""Chargement des donnees ouvertes SkillCorner (tracking, evenements dynamiques). Telechargement a la demande dans un dossier local (non versionne)."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as cfg

_DIR = Path("./data")
_GOAL_REF = cfg.GOAL_REFERENCE


def set_dir(path):
    global _DIR
    _DIR = Path(path)
    _DIR.mkdir(parents=True, exist_ok=True)
    load_match.cache_clear()


def get_dir():
    return _DIR


def set_goal_reference(mode):
    """'line' (ligne de but, par defaut) ou 'gk_median' (position mediane du gardien, premiere version)."""
    global _GOAL_REF
    if mode not in ("line", "gk_median"):
        raise ValueError("reference de but : 'line' ou 'gk_median'")
    _GOAL_REF = mode


def goal_reference():
    return _GOAL_REF


def fetch(url, dest):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=900) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    return dest


def grp_of(acr):
    if acr == "GK":
        return "GK"
    if acr in cfg.DEF_ROLES:
        return "D"
    if acr in cfg.MID_ROLES:
        return "M"
    return "F"


@lru_cache(maxsize=2)
def load_match(mid):
    """Tableaux image x joueur : X, Y, D (1 detecte, 0 extrapole, NaN absent), balle, periode, equipe et poste de chaque colonne."""
    meta = json.load(open(fetch(f"{cfg.RAW}/{mid}/{mid}_match.json", _DIR / f"{mid}_match.json")))
    pl = pd.DataFrame([{"player_id": p["id"], "team_id": p["team_id"], "grp": grp_of(p["player_role"]["acronym"])} for p in meta["players"]])
    cols = sorted(pl.player_id)
    cidx = {p: i for i, p in enumerate(cols)}
    path = fetch(f"{cfg.MEDIA}/{mid}/{mid}_tracking_extrapolated.jsonl", _DIR / f"{mid}_trk.jsonl")
    recs = []
    with open(path, "rb") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("period") is None:
                continue
            recs.append(r)
    f0 = min(r["frame"] for r in recs)
    n = max(r["frame"] for r in recs) - f0 + 1
    X = np.full((n, len(cols)), np.nan)
    Y = np.full((n, len(cols)), np.nan)
    D = np.full((n, len(cols)), np.nan)
    BX = np.full(n, np.nan)
    BY = np.full(n, np.nan)
    PER = np.zeros(n, dtype=int)
    for r in recs:
        i = r["frame"] - f0
        PER[i] = r["period"]
        b = r.get("ball_data") or {}
        if b.get("x") is not None:
            BX[i], BY[i] = b["x"], b["y"]
        for p in r["player_data"]:
            j = cidx.get(p["player_id"])
            if j is not None and p.get("x") is not None:
                X[i, j], Y[i, j] = p["x"], p["y"]
                D[i, j] = 1.0 if p.get("is_detected") else 0.0
    ind = pl.set_index("player_id")
    team = np.array([int(ind.team_id[c]) for c in cols])
    grp = np.array([ind.grp[c] for c in cols])
    return dict(f0=f0, n=n, cols=cols, cidx=cidx, X=X, Y=Y, D=D, BX=BX, BY=BY, PER=PER, team=team, grp=grp,
                half_len=float(meta.get("pitch_length") or 105.0) / 2)


def own_goal_sign(T, reference=None):
    """Par periode et equipe : abscisse du but defendu.

    reference='line' (defaut) : ligne de but, +/- demi-longueur du terrain ; le signe vient de la position mediane du gardien.
    reference='gk_median' : position mediane du gardien (premiere version : decalee de 10 a 18 m de la ligne de but, car le gardien
    n'est pas observe la plupart du temps et se tient souvent haut). Les distances 'a son but' du reste du code en dependent."""
    ref = reference or _GOAL_REF
    gx = {}
    for per in np.unique(T["PER"][T["PER"] > 0]):
        m = T["PER"] == per
        for t in np.unique(T["team"]):
            c = np.where((T["team"] == t) & (T["grp"] == "GK"))[0]
            med = np.nanmedian(T["X"][m][:, c]) if len(c) else np.nan
            if ref == "line" and np.isfinite(med):
                med = float(np.sign(med)) * T.get("half_len", 52.5)
            gx[(per, t)] = med
    return gx


def load_events(mid):
    path = fetch(f"{cfg.RAW}/{mid}/{mid}_dynamic_events.csv", _DIR / f"{mid}_dyn.csv")
    return pd.read_csv(path, low_memory=False)


def possession_array(ev, f0, n):
    """Equipe en possession a chaque image (a partir des evenements player_possession)."""
    p = ev[ev.event_type == "player_possession"].dropna(subset=["frame_start", "team_id"]).sort_values("frame_start")
    poss = np.full(n, np.nan)
    fs = p.frame_start.astype(int).values - f0
    tm = p.team_id.values.astype(float)
    for i, f in enumerate(fs):
        if 0 <= f < n:
            end = fs[i + 1] if i + 1 < len(fs) else n
            poss[max(f, 0):min(end, n)] = tm[i]
    return poss


def shot_frames(ev, f0):
    p = ev[(ev.event_type == "player_possession") & (ev.get("end_type") == "shot")].dropna(subset=["frame_end", "team_id"])
    return (p.frame_end.astype(int).values - f0), p.team_id.values.astype(int)
