"""Un moment reel, pour le VOIR : positions des 22 joueurs et de la balle pendant quelques secondes, avec ce que la camera a vraiment observe.

Sert a deux choses : la vue « un moment » de l'outil HTML (outil.build) et le rendu 3D (render3d.py). Rien n'est invente : les positions sont celles du fournisseur ;
les disques d'incertitude sont ceux de la couche de fiabilite (rayon a 90 % selon la duree depuis la derniere observation). Aucune identite de joueur n'est conservee.

Coordonnees de sortie : x = distance a la ligne de but de l'equipe qui defend (m), y = ecart a l'axe du terrain (m), l'equipe qui defend defend le but en x = 0."""
import numpy as np

from . import config as cfg
from . import data as dat
from . import frames as fr
from . import reliability as rel


def select(df, prefer_pose=True, min_res_sd=1.0, max_line_sd=3.0, max_line_m=55.0):
    """Choisit un moment illustratif de facon deterministe : bloc installe, balle a >= 35 m, porteur libre, ligne plus haute que d'habitude (>= 1 ecart-type),
    5 a 8 joueurs de champ observes dont au plus 3 des 4 plus bas, balle entre 50 et 80 m de son but, ligne entre 30 et 55 m (on ecarte les pressings dans la surface adverse) ; parmi eux, la ligne la plus haute. Retourne (match_id, equipe, image) ou None."""
    d = df.copy()
    d["res"] = fr.residual_line(d, True)
    z = d.res / d.res.std()
    ok = (d.age >= 5) & (d.ball_x >= 35) & (d.free == 1) & (z >= min_res_sd) & d.n_det.between(5, 8) & (d.n_det_back4 <= 3) & (d.line_sd <= max_line_sd) & d.line.between(30, max_line_m) & d.ball_x.between(50, 80)
    c = d[ok]
    if prefer_pose and len(c[c.match_id.isin(cfg.POSE_MATCHES)]):
        c = c[c.match_id.isin(cfg.POSE_MATCHES)]
    if c.empty:
        return None
    r = c.loc[c.res.idxmax()]
    return int(r.match_id), int(r.team), int(r.t)


def extract(match_id, team, t, back=30, fwd=30, step=2):
    """Fenetre [t - back, t + fwd] images (10 Hz), une image sur `step` : positions, observe ou non, rayon d'incertitude a 90 % (m), ligne et son incertitude."""
    T = dat.load_match(match_id)
    per = int(T["PER"][t])
    g = dat.own_goal_sign(T).get((per, team))
    sgn = 1.0 if g < 0 else -1.0
    idx = [f for f in range(t - back, t + fwd + 1, step) if 0 <= f < T["n"] and T["PER"][f] == per]
    cols = [j for j in range(len(T["cols"])) if np.isfinite(T["X"][idx, j]).any()]
    G = rel.gap_lengths(T["D"][:, cols])
    players = []
    for k, j in enumerate(cols):
        x = sgn * (T["X"][idx, j] - g)
        y = sgn * T["Y"][idx, j]
        players.append(dict(side="d" if T["team"][j] == team else "a", gk=int(T["grp"][j] == "GK"),
                            x=[None if not np.isfinite(v) else round(float(v), 2) for v in x], y=[None if not np.isfinite(v) else round(float(v), 2) for v in y],
                            det=[int(T["D"][f, j] == 1) for f in idx], r=[round(float(rel.radius90(G[f, k])), 2) for f in idx]))
    ball = dict(x=[None if not np.isfinite(T["BX"][f]) else round(float(sgn * (T["BX"][f] - g)), 2) for f in idx],
                y=[None if not np.isfinite(T["BY"][f]) else round(float(sgn * T["BY"][f]), 2) for f in idx])
    line, sd = [], []
    deff = [p for p in players if p["side"] == "d" and not p["gk"]]
    for i, f in enumerate(idx):
        d_ = np.array([p["x"][i] for p in deff if p["x"][i] is not None])
        sg = np.array([rel.sigma_of(0) if p["det"][i] else p["r"][i] / cfg.R90 for p in deff if p["x"][i] is not None])
        if len(d_) < 4:
            line.append(None)
            sd.append(None)
            continue
        line.append(round(float(np.sort(d_)[:4].mean()), 2))
        sims = d_[None, :] + np.random.default_rng(i).normal(size=(100, len(d_))) * sg[None, :]
        sd.append(round(float(np.sort(sims, axis=1)[:, :4].mean(axis=1).std()), 2))
    return dict(meta=dict(match_id=int(match_id), frame=int(t), fps=10 // step, n=len(idx), pitch=[2 * T.get("half_len", 52.5), 68.0], reference=dat.goal_reference()),
                players=players, ball=ball, line=line, sd=sd)
