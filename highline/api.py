"""API publique : lire une hauteur de ligne avec son incertitude, puis la situation de decision, en quelques lignes.

    from highline.api import load_table, line_decision

    table = load_table("out/decision_cells.json")            # ecrit par `python reproduce.py`
    r = line_decision(table, dist_to_goal=[18, 20, 22, 25, 30, 34, 38, 41, 45, 50],   # 10 joueurs de champ, m de la ligne de but
                      gap_s=[0, 0, 4, 0, 0, 12, 0, 0, 0, 0],                           # secondes depuis la derniere detection (0 = observe)
                      habitual_line_m=19.0,                                            # hauteur habituelle pour cette position de balle
                      ball_x=55.0, carrier="libre", w=1.0)
    print(r["message"])

Conventions : metres ; distance a la ligne de but du defenseur ; hauteur de ligne = moyenne des 4 plus bas (joueurs de champ).
Fournisseur de tracking quelconque : seuls des positions et, si on les connait, les durees d'extrapolation sont necessaires.
Le resultat est une ASSOCIATION observee en A-League 2024/25 : pas une recommandation causale."""
import json
from pathlib import Path

import numpy as np

from . import config as cfg
from . import reliability as rel
from .stats import pct

__all__ = ["load_table", "observe_line", "fit_habit", "decide", "line_decision"]


def load_table(path_or_dict):
    """Table de decision exportee par le pipeline (decision_cells.json) ; accepte aussi le dict deja charge."""
    t = path_or_dict if isinstance(path_or_dict, dict) else json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    if "cells" not in t or "sd_line" not in t:
        raise ValueError("table de decision invalide : colonnes 'cells' et 'sd_line' attendues")
    return t


def observe_line(dist_to_goal, gap_s=None, k=300, seed=0):
    """Hauteur de ligne (4 plus bas) et son incertitude, a partir des 10 distances au but et de la duree d'extrapolation de chaque joueur."""
    d = np.asarray(dist_to_goal, float)
    if d.shape != (10,):
        raise ValueError("il faut les 10 joueurs de champ de l'equipe qui defend")
    gaps = np.zeros(10) if gap_s is None else np.asarray(gap_s, float)
    if gaps.shape != (10,):
        raise ValueError("gap_s doit avoir 10 valeurs (secondes depuis la derniere detection ; 0 = observe)")
    rng = np.random.default_rng(seed)
    sd = rel.sigma_of(gaps * cfg.FPS)
    sims = d[None, :] + rng.normal(size=(k, 10)) * sd[None, :]
    lines = np.sort(sims, axis=1)[:, :4].mean(axis=1)
    deepest = np.argsort(d)[:4]
    line = rel.line_of(d)
    s = float(lines.std())
    level = "high" if s < 0.5 else ("medium" if s < 1.5 else "low")
    return dict(line_m=line, sd_m=s, p5_m=float(np.percentile(lines, 5)), p95_m=float(np.percentile(lines, 95)), n_observed=int((gaps == 0).sum()),
                deepest4_observed=int((gaps[deepest] == 0).sum()), reliability=level, _sims=lines)


def fit_habit(line_m, ball_x, ball_ay=None):
    """Ajuste la hauteur HABITUELLE d'une equipe selon la position de la balle (meme ajustement quadratique que le pipeline) et retourne une fonction."""
    bx = np.asarray(ball_x, float)
    by = np.zeros_like(bx) if ball_ay is None else np.abs(np.asarray(ball_ay, float))
    A = np.column_stack([np.ones(len(bx)), bx, bx ** 2, by])
    beta = np.linalg.lstsq(A, np.asarray(line_m, float), rcond=None)[0]
    return lambda x, ay=0.0: float(beta @ np.array([1.0, x, x * x, abs(ay)]))


def _find(table, key):
    for c in table["cells"]:
        if tuple(c["key"]) == tuple(key):
            return c
    return None


def decide(table, carrier="tous", zone="tous", score="tous", minute="tous", w=1.0, residual_m=None):
    """Gain, cout et bilan d'un bloc plus haut dans une situation. residual_m = ecart a la hauteur habituelle (m) ; sinon : effet d'UN ecart-type.

    Retourne cost / gain / net (points de probabilite de tir en 15 s), intervalles a 95 % (bootstrap par match), w_star et une etiquette."""
    key = (carrier, zone, score, minute)
    c = _find(table, key)
    if c is None:
        ok = sorted({" / ".join(x for x in cc["key"] if x != "tous") or "ensemble" for cc in table["cells"]})
        raise ValueError(f"situation {key} absente (moins de {table.get('min_cell', 400)} moments). Situations disponibles : {ok[:12]} ...")
    scale = 1.0 if residual_m is None else float(residual_m) / table["sd_line"]
    dr = np.asarray(c["draws"], float)
    net_d = (dr[:, 1] - w * dr[:, 0]) * scale
    lo, hi = pct(net_d)
    cost, gain = c["cost"] * scale, c["gain"] * scale
    net = (c["gain"] - w * c["cost"]) * scale
    up = scale > 0
    if lo > 0:
        tag = "higher_pays" if up else "lower_pays"
    elif hi < 0:
        tag = "higher_costs" if up else "lower_costs"
    else:
        tag = "undetermined"
    wstar = c["gain"] / c["cost"] if (c["cost"] > 0 and c["gain"] > 0) else None
    return dict(key=key, n=c["n"], cost=cost, gain=gain, net=net, net_ci=(lo, hi), w_star=wstar, tag=tag, per_sd=residual_m is None)


_MSG = {
    "fr": dict(
        up="monter paie", down="monter coute", und="pas tranchable",
        txt=("Ligne a {line:.1f} m (incertitude {sd:.1f} m ; {n4} des 4 plus bas observes), {rel:+.1f} m par rapport a l'habitude : bilan {net:+.2f} point [{lo:+.2f} ; {hi:+.2f}] de probabilite de tir en 15 s "
             "(gain {gain:+.2f}, cout {cost:+.2f}) -> {tag}."),
        warn_low="Incertitude de la ligne elevee : la hauteur peut etre fausse de plusieurs metres (positions extrapolees).",
        warn_sign="Le signe de l'ecart a l'habitude n'est assure que dans {p:.0f} % des tirages de positions plausibles."),
    "en": dict(
        up="going up pays", down="going up costs", und="cannot tell",
        txt=("Line at {line:.1f} m (uncertainty {sd:.1f} m; {n4} of the 4 deepest observed), {rel:+.1f} m versus usual: net {net:+.2f} points [{lo:+.2f} ; {hi:+.2f}] of shot probability within 15 s "
             "(gain {gain:+.2f}, cost {cost:+.2f}) -> {tag}."),
        warn_low="High line uncertainty: the height may be off by several metres (extrapolated positions).",
        warn_sign="The sign of the gap to the usual height holds in only {p:.0f}% of plausible position draws."),
}


def line_decision(table, dist_to_goal, habitual_line_m, ball_x, carrier="libre", gap_s=None, score="tous", minute="tous", w=1.0, lang="fr", seed=0):
    """Tout en un : hauteur de ligne observee + incertitude + ecart a l'habitude + bilan de la situation + avertissements de fiabilite."""
    if ball_x < 35:
        raise ValueError("hors perimetre : le tableau ne couvre que les moments ou la balle est a plus de 35 m du but de l'equipe qui defend")
    zone = "adversaire" if ball_x >= 70 else "milieu"
    o = observe_line(dist_to_goal, gap_s, seed=seed)
    resid = o["line_m"] - habitual_line_m
    d = decide(table, carrier, zone, score, minute, w, residual_m=resid)
    sims = o.pop("_sims")
    p_sign = float(np.mean(np.sign(sims - habitual_line_m) == np.sign(resid))) if resid != 0 else 0.5
    M = _MSG[lang]
    tag = M["up"] if d["tag"] in ("higher_pays", "lower_costs") else (M["down"] if d["tag"] in ("higher_costs", "lower_pays") else M["und"])
    msg = M["txt"].format(line=o["line_m"], sd=o["sd_m"], n4=o["deepest4_observed"], rel=resid, net=d["net"], lo=d["net_ci"][0], hi=d["net_ci"][1], gain=d["gain"], cost=d["cost"], tag=tag)
    warnings = []
    if o["reliability"] == "low":
        warnings.append(M["warn_low"])
    if p_sign < 0.9:
        warnings.append(M["warn_sign"].format(p=100 * p_sign))
    return dict(observation=o, residual_m=resid, decision=d, p_sign=p_sign, warnings=warnings, message=" ".join([msg] + warnings))
