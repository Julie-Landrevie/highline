"""Body Pose (2 matchs) : couche OPTIONNELLE et EXPLORATOIRE. Necessite `pip install highline[pose]` (huggingface_hub) et ~600 Mo par match.

Hypotheses enregistrees d'avance (voir docs/HYPOTHESES_BODYPOSE.md) : H1 plus de joueurs proches de la balle tournes vers leur but, plus de tirs adverses ;
H2 plus d'adversaires proches face a l'avant, plus de tirs adverses. Avec 2 matchs, le test ne peut pas trancher : l'ecart minimal detectable est rapporte."""
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from . import data as dat
from .stats import ols_cl, zs

POSE_STEP = 5
MAX_ERR_TORSO_CM, MAX_ERR_HEAD_CM = 20.0, 30.0
SHOULDER_WIDTH = (0.20, 0.60)
HEAD_VEC_MIN = 0.03
_DIR = Path("./hp_data")


def set_dir(path):
    global _DIR
    _DIR = Path(path)
    _DIR.mkdir(parents=True, exist_ok=True)


def available():
    try:
        import huggingface_hub  # noqa: F401
        return True
    except ImportError:
        return any(_DIR.glob("*_pose_vectors.pkl"))


def _lookup(j):
    return {k.lower(): v for k, v in j.items() if isinstance(v, dict) and v.get("xyz") is not None}


def _pt(d, name, max_err):
    v = d.get(name)
    if v is None:
        return None
    if v.get("p90_mae_cm") is not None and v["p90_mae_cm"] > max_err:
        return None
    return v["xyz"][0], v["xyz"][1]


def _mean(pts):
    pts = [p for p in pts if p is not None]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def pose_row(p):
    """Vecteurs (plan du terrain) d'un joueur : epaules G-D, hanches G-D, oreilles -> nez. None si pas de pose."""
    j = p.get("joints")
    if not j:
        return None
    d = _lookup(j)
    out = {"px": p.get("x"), "py": p.get("y")}
    ls, rs = _pt(d, "lshoulder", MAX_ERR_TORSO_CM), _pt(d, "rshoulder", MAX_ERR_TORSO_CM)
    if ls and rs:
        vx, vy = ls[0] - rs[0], ls[1] - rs[1]
        if SHOULDER_WIDTH[0] <= math.hypot(vx, vy) <= SHOULDER_WIDTH[1]:
            out.update(sh_x=vx, sh_y=vy)
    lh, rh = _pt(d, "lhip", MAX_ERR_TORSO_CM), _pt(d, "rhip", MAX_ERR_TORSO_CM)
    if lh and rh:
        vx, vy = lh[0] - rh[0], lh[1] - rh[1]
        if 0.10 <= math.hypot(vx, vy) <= 0.50:
            out.update(hip_x=vx, hip_y=vy)
    front = _mean([_pt(d, "nose", MAX_ERR_HEAD_CM), _pt(d, "leye", MAX_ERR_HEAD_CM), _pt(d, "reye", MAX_ERR_HEAD_CM)])
    back = _mean([_pt(d, "lear", MAX_ERR_HEAD_CM), _pt(d, "rear", MAX_ERR_HEAD_CM)])
    if front and back:
        vx, vy = front[0] - back[0], front[1] - back[1]
        if math.hypot(vx, vy) >= HEAD_VEC_MIN:
            out.update(hd_x=vx, hd_y=vy)
    return out if len(out) > 2 else None


def read_pose(mid):
    cache = _DIR / f"{mid}_pose_vectors.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    from huggingface_hub import hf_hub_download
    print(f"[{mid}] Body Pose : telechargement / cache HF (~600 Mo)...", flush=True)
    zp = hf_hub_download(repo_id=cfg.HF_REPO_ID, filename=f"raw/{mid}.jsonl.zip", repo_type="dataset")
    frame_re = re.compile(rb'^\{\s*"frame"\s*:\s*(\d+)')
    rows = []
    with zipfile.ZipFile(zp) as z:
        inner = [n for n in z.namelist() if n.endswith(".jsonl")][0]
        with z.open(inner) as fh:
            for raw in fh:
                m = frame_re.match(raw)
                if not m or int(m.group(1)) % POSE_STEP != 0:
                    continue
                rec = json.loads(raw)
                for p in rec.get("player_data", []):
                    r = pose_row(p)
                    if r is not None:
                        r.update(pose_frame=rec["frame"], player_id=p["player_id"])
                        rows.append(r)
    df = pd.DataFrame(rows)
    df["T"] = (df.pose_frame * 2 // 5).astype(int)          # frame de tracking correspondante (pose a 25 Hz)
    df.to_pickle(cache)
    return df


def pose_features(d, pose_dir=None):
    """Pour chaque moment (colonnes match_id, frame, sgn, team, opp) : part des 3 joueurs de `opp` les plus proches de la balle tournes vers leur but (a_turned)
    et part des 3 joueurs de `team` les plus proches face a l'avant (b_forward). `sgn` = direction d'attaque de `team`."""
    if pose_dir:
        set_dir(pose_dir)
    res = []
    for mid in cfg.POSE_MATCHES:
        sub = d[d.match_id == mid]
        if sub.empty:
            continue
        T = dat.load_match(mid)
        V = read_pose(mid)
        V = V[V.player_id.isin(T["cidx"]) & V.sh_x.notna()].copy()
        V["ci"] = V.player_id.map(T["cidx"])
        V["fi"] = V["T"] - T["f0"]
        X, Y = T["X"], T["Y"]
        ok = (V.fi.values >= 3) & (V.fi.values < T["n"] - 3)
        V = V[ok].copy()
        fi, ci = V.fi.values, V.ci.values
        vx, vy = (X[fi + 3, ci] - X[fi - 3, ci]) / 0.6, (Y[fi + 3, ci] - Y[fi - 3, ci]) / 0.6
        sp = np.hypot(vx, vy)
        # direction "devant" = perpendiculaire a la ligne des epaules ; sens verifie par la direction de course (>= 3 m/s)
        fx, fy = V.sh_y.values, -V.sh_x.values
        yaw = np.arctan2(fy, fx)
        diff = np.abs(np.degrees((yaw - np.arctan2(vy, vx) + np.pi) % (2 * np.pi) - np.pi))
        if (sp >= 3).sum() > 30 and np.nanmedian(diff[sp >= 3]) > 120:
            yaw = yaw + np.pi
        V["yaw"] = yaw
        by_fi = {k: g for k, g in V.groupby("fi")}
        for r in sub.itertuples():
            fw = 0.0 if r.sgn > 0 else np.pi
            near = {}
            for side, tm in (("B", r.team), ("A", r.opp)):
                cs = np.where((T["team"] == tm) & (T["grp"] != "GK") & np.isfinite(X[int(r.frame)]))[0]
                dist = np.hypot(X[int(r.frame), cs] - T["BX"][int(r.frame)], Y[int(r.frame), cs] - T["BY"][int(r.frame)])
                near[side] = cs[np.argsort(dist)][:3]
            shares = {}
            for side in ("B", "A"):
                vals = []
                for c in near[side]:
                    best = None
                    for dfi in (0, 1, -1, 2, -2, 3, -3):
                        g = by_fi.get(int(r.frame) + dfi)
                        if g is not None:
                            m = g[g.ci == c]
                            if len(m):
                                best = float(m.yaw.iloc[0])
                                break
                    if best is not None:
                        vals.append(abs(np.degrees((best - fw + np.pi) % (2 * np.pi) - np.pi)) <= 60)
                shares[side] = (np.mean(vals) if len(vals) >= 2 else np.nan, len(vals))
            res.append(dict(match_id=mid, frame=int(r.frame), b_forward=shares["B"][0], b_n=shares["B"][1], a_turned=shares["A"][0], a_n=shares["A"][1]))
    return pd.DataFrame(res)


def analyse(d, rep, pose_dir=None):
    """Effet des deux variables de pose sur le tir adverse / le tir de l'equipe dans le bloc installe, avec l'ecart minimal detectable (2,8 x erreur-type)."""
    rep.say("BODY POSE (2 matchs ; exploratoire)")
    if not available():
        rep.say("   non evalue (huggingface_hub absent et aucun cache de pose). Installer avec : pip install -e .[pose]")
        return None
    try:
        ds = d[d.match_id.isin(cfg.POSE_MATCHES)].copy()
        recs = []
        for mid, g in ds.groupby("match_id"):
            T = dat.load_match(mid)
            gx = dat.own_goal_sign(T)
            teams = [int(t) for t in pd.unique(T["team"])]
            for r in g.itertuples():
                gg = gx.get((r.per, r.team))
                if gg is None or not np.isfinite(gg):
                    continue
                other = [t for t in teams if t != r.team]
                if other:
                    recs.append(dict(match_id=mid, frame=int(r.t), sgn=-(1.0 if gg < 0 else -1.0), team=int(other[0]), opp=int(r.team)))
        pf = pose_features(pd.DataFrame(recs), pose_dir)
        m = ds.assign(frame=ds.t.astype(int)).merge(pf, on=["match_id", "frame"], how="inner")
        m = m[(m.a_n >= 2) & (m.b_n >= 2)].copy()
        rep.say(f"{len(m)} moments avec >= 2 joueurs de T et >= 2 adversaires a buste valide ; tirs adverses en 15 s : {int(m.opp15.sum())} ; tirs de T : {int(m.own15.sum())}")
        m["f_turn"], m["f_fwd"] = zs(m.a_turned), zs(m.b_forward)
        m["blk"] = m.match_id.astype(str) + "_" + (m.frame // 1500).astype(str)
        ctrl = ["f_res", "f_bx", "f_bx2", "f_ay", "libre"]
        rep.say(f"{'resultat':>8s} {'variable':>8s} {'effet (points par ecart-type)':>34s} {'ecart minimal detectable':>26s} {'match 1925299':>14s} {'match 1996435':>14s}")
        sig, rows = False, []
        for y in ("opp15", "own15"):
            for v in ("f_turn", "f_fwd"):
                names, b, se = ols_cl(m[y].values, m[[v] + ctrl], m.blk.values)
                i = names.index(v)
                per = []
                for mid in cfg.POSE_MATCHES:
                    s = m[m.match_id == mid]
                    n2, b2, _ = ols_cl(s[y].values, s[[v] + ctrl], s.blk.values)
                    per.append(100 * b2[n2.index(v)])
                lo, hi = 100 * (b[i] - 1.96 * se[i]), 100 * (b[i] + 1.96 * se[i])
                sig |= bool((lo > 0 or hi < 0) and np.sign(per[0]) == np.sign(per[1]))
                rows.append(dict(y=y, var=v[2:], eff=100 * b[i], lo=lo, hi=hi, mde=100 * 2.8 * se[i], per=per))
                rep.say(f"{y:>8s} {v[2:]:>8s} {100 * b[i]:+8.2f} [{lo:+.2f};{hi:+.2f}] {'':>8s} {100 * 2.8 * se[i]:>24.2f} {per[0]:+14.2f} {per[1]:+14.2f}")
        rep.say(f"VERDICT POSE : {'pose utile' if sig else 'exploratoire (ne pas annoncer comme resultat)'}")
        rep.say("Lecture : l'ecart minimal detectable est la plus petite difference que 2 matchs peuvent mettre en evidence ; s'il est du meme ordre que l'effet de la ligne (1 a 2 points), le test ne peut rien conclure.")
        return dict(rows=rows, n=len(m), shots=int(m.opp15.sum()), verdict="utile" if sig else "exploratoire")
    except Exception as ex:                                               # la pose ne doit jamais faire echouer le reste
        rep.say(f"   pose non evaluee ({type(ex).__name__}: {ex})")
        return None
