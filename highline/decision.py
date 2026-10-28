"""Table de decision de la ligne haute : cout (tirs concedes), gain (tirs de l'equipe) et bilan, par situation.

Unite : moment ou l'equipe T defend, bloc installe (adversaire en possession depuis >= 5 s), balle a >= 35 m de son but.
Effets par ecart-type de hauteur de ligne RESIDUELLE (ligne moins son habitude pour la meme position de balle), en points de probabilite.
bilan(w) = gain - w x cout ; w* = gain / cout = poids d'un tir concede au-dela duquel garder la ligne basse est preferable."""
import itertools

import numpy as np
import pandas as pd

from . import config as cfg
from . import frames as fr
from .stats import auc, half_label, logit, ols_cl, pct, zs

ZONE_LABELS = {"milieu": "milieu (35-70 m)", "adversaire": "chez l'adversaire (> 70 m)"}
DIM_ORDER = ["porteur", "zone", "score", "minute"]
LEVELS = {"porteur": ["libre", "presse"], "zone": ["milieu", "adversaire"], "score": ["derriere", "egalite", "devant"], "minute": ["0-30", "30-60", "60+"]}
PAIRS = [(("score", "derriere"), ("score", "devant")), (("minute", "0-30"), ("minute", "60+")),
         (("porteur", "libre"), ("porteur", "presse")), (("zone", "milieu"), ("zone", "adversaire"))]


def prepare(df, strict=False):
    """Population du tableau de decision. strict=True exige score, minute et porteur connus (tests d'interaction et modele)."""
    tab = df.copy().reset_index(drop=True)
    tab["opp15"] = tab["shot15"]
    tab["res"] = fr.residual_line(tab, True)
    need = ["res", "ball_x", "ball_ay"] + (["score_diff", "minute", "free"] if strict else [])
    d = tab.dropna(subset=need)
    d = d[(d.age >= 5) & (d.ball_x >= 35)].copy().reset_index(drop=True)
    d["net15"] = d.own15 - d.opp15
    d["f_res"] = zs(d.res)
    d["f_bx"], d["f_bx2"], d["f_ay"] = zs(d.ball_x), zs(d.ball_x ** 2), zs(d.ball_ay)
    dg = np.hypot(d.ball_x, d.ball_ay)
    d["f_dg"], d["f_dg2"] = zs(dg), zs(dg ** 2)
    d["half"] = half_label(d.match_id.values)
    d["score_c"] = d.score_diff.clip(-2, 2)
    d["f_min"] = zs(d.minute)
    d["f_age"] = zs(d.age.clip(upper=30))
    d["libre"] = d.free.astype(float)
    d["adv"] = (d.ball_x >= 70).astype(float)
    for a in ("libre", "adv", "score_c", "f_min"):
        d["x_" + a] = d.f_res * d[a]
    return d


CTRL = ["f_bx", "f_bx2", "f_ay", "f_dg", "f_dg2"]


def _masks(d):
    m = {("porteur", "libre"): (d.free == 1).values, ("porteur", "presse"): (d.free == 0).values,
         ("zone", "milieu"): ((d.ball_x >= 35) & (d.ball_x < 70)).values, ("zone", "adversaire"): (d.ball_x >= 70).values}
    for s in LEVELS["score"]:
        m[("score", s)] = (d.score_state.astype(str) == s).values
    for s in LEVELS["minute"]:
        m[("minute", s)] = (d.min_bin.astype(str) == s).values
    return m


def cells(d, nb=300, rng=None, min_n=cfg.MIN_CELL):
    """Toutes les situations (croisement de porteur, zone, score, minute ; 'tous' = pas de filtre) d'au moins min_n moments, avec bootstrap par match."""
    rng = rng or np.random.default_rng(204)
    M = _masks(d)
    A = np.column_stack([np.ones(len(d)), d[["f_res"] + CTRL].values])
    Yg = np.column_stack([d.opp15.values.astype(float), d.own15.values.astype(float)])
    ids, half = d.match_id.values, d.half.values
    idx_by = {m: np.where(ids == m)[0] for m in np.unique(ids)}
    out = []
    for combo in itertools.product(*[["tous"] + LEVELS[k] for k in DIM_ORDER]):
        mask = np.ones(len(d), bool)
        for k, lab in zip(DIM_ORDER, combo):
            if lab != "tous":
                mask &= M[(k, lab)]
        if mask.sum() >= min_n:
            out.append(dict(key=combo, mask=mask, n=int(mask.sum())))

    def coefs(ix):
        if len(ix) < 50:
            return np.array([np.nan, np.nan])
        return 100 * np.linalg.lstsq(A[ix], Yg[ix], rcond=None)[0][1]     # [cout, gain] en points par ecart-type

    for c in out:
        ix = np.where(c["mask"])[0]
        c["point"] = coefs(ix)
        c["halves"] = [coefs(np.where(c["mask"] & (half == h))[0]) for h in (0, 1)]
        c["draws"] = np.full((nb, 2), np.nan)
    ms = list(idx_by)
    for b in range(nb):
        ix = np.concatenate([idx_by[m] for m in rng.choice(ms, len(ms))])
        for c in out:
            c["draws"][b] = coefs(ix[c["mask"][ix]])
    return out


def summarize(cell, w=1.0):
    """Cout, gain, bilan(w) avec IC95 % bootstrap, w* et etiquette de lecture."""
    cost, gain = cell["point"]
    dr = cell["draws"]
    lo_c, hi_c = pct(dr[:, 0])
    lo_g, hi_g = pct(dr[:, 1])
    net = gain - w * cost
    lo, hi = pct(dr[:, 1] - w * dr[:, 0])
    tag = "monter PAIE" if lo > 0 else ("monter COUTE" if hi < 0 else "pas tranchable")
    wstar = gain / cost if (cost > 0 and gain > 0) else np.nan
    return dict(cost=cost, cost_ci=(lo_c, hi_c), gain=gain, gain_ci=(lo_g, hi_g), net=net, net_ci=(lo, hi), wstar=wstar, tag=tag)


def cells_payload(cell_list, thin=100, seed=1):
    """Cellules serialisables (point cout / gain par ecart-type et tirages bootstrap amincis) : pour l'outil HTML et pour highline.api."""
    rng = np.random.default_rng(seed)
    out = []
    for c in cell_list:
        dr = c["draws"][~np.isnan(c["draws"]).any(axis=1)]
        if len(dr) > thin:
            dr = dr[rng.choice(len(dr), thin, replace=False)]
        out.append(dict(key=list(c["key"]), n=c["n"], cost=round(float(c["point"][0]), 3), gain=round(float(c["point"][1]), 3),
                        draws=[[round(float(a), 3), round(float(b), 3)] for a, b in dr]))
    return out


def export_cells(cell_list, sd_line, path, goal_reference="line", thin=100):
    """Ecrit la table de decision en JSON (lisible par highline.api.load_table). Aucune donnee brute."""
    import json
    payload = dict(version=1, dims=LEVELS, goal_reference=goal_reference, sd_line=round(float(sd_line), 3), min_cell=cfg.MIN_CELL,
                   unit="points de probabilite de tir en 15 s par ecart-type de hauteur de ligne residuelle", cells=cells_payload(cell_list, thin))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    return path


def label(key):
    parts = [f"{k} : {v}" for k, v in zip(DIM_ORDER, key) if v != "tous"]
    return " / ".join(parts) if parts else "ensemble"


def one_dim(cell_list):
    """Cellules a au plus une dimension filtree (ensemble + marges)."""
    return [c for c in cell_list if sum(v != "tous" for v in c["key"]) <= 1]


def diff_tests(cell_list):
    """Difference de bilan(w=1) entre strates extremes de chaque dimension : IC bootstrap et signe par moitie. Retourne une liste de dicts."""
    idx = {c["key"]: c for c in cell_list}

    def key1(dim, lab):
        return tuple(lab if k == dim else "tous" for k in DIM_ORDER)
    res = []
    for (d1, l1), (d2, l2) in PAIRS:
        c1, c2 = idx.get(key1(d1, l1)), idx.get(key1(d2, l2))
        if c1 is None or c2 is None:
            res.append(dict(dim=d1, a=l1, b=l2, ok=None))
            continue
        dv = (c1["draws"][:, 1] - c1["draws"][:, 0]) - (c2["draws"][:, 1] - c2["draws"][:, 0])
        pt = (c1["point"][1] - c1["point"][0]) - (c2["point"][1] - c2["point"][0])
        lo, hi = pct(dv)
        hs = [(c1["halves"][h][1] - c1["halves"][h][0]) - (c2["halves"][h][1] - c2["halves"][h][0]) for h in (0, 1)]
        ok = bool((lo > 0 or hi < 0) and np.isfinite(hs).all() and np.sign(hs[0]) == np.sign(hs[1]) == np.sign(pt))
        res.append(dict(dim=d1, a=l1, b=l2, diff=pt, ci=(lo, hi), halves=hs, ok=ok))
    return res


def verdict(cell_list, diffs):
    est = [x["dim"] for x in diffs if x.get("ok")]
    anyt = any(summarize(c)["tag"] != "pas tranchable" for c in one_dim(cell_list) if c["key"] != ("tous",) * 4)
    return ("GO" if est else ("A VOIR" if anyt else "NO-GO")), est


def report_table(cell_list, diffs, rep):
    rep.say("Par situation (effet d'un bloc plus haut de 1 ecart-type, points de probabilite [IC95 % bootstrap par match]) ; bilan(w) = gain - w x cout ;")
    rep.say("w* = poids d'un tir concede (par rapport a un tir cree) au-dela duquel garder la ligne basse est preferable (si cout > 0 et gain > 0).")
    rep.say(f"{'situation':>34s} {'n':>6s} | {'cout':>20s} {'gain':>20s} | {'bilan w=1':>20s} {'bilan w=2':>20s} | {'w*':>5s} | lecture (w=1 / w=2)")
    for c in one_dim(cell_list):
        s1, s2 = summarize(c, 1.0), summarize(c, 2.0)
        w = f"{s1['wstar']:5.1f}" if np.isfinite(s1["wstar"]) else "   - "
        rep.say(f"{label(c['key']):>34s} {c['n']:6d} | {s1['cost']:+5.2f} [{s1['cost_ci'][0]:+.2f};{s1['cost_ci'][1]:+.2f}] {s1['gain']:+5.2f} [{s1['gain_ci'][0]:+.2f};{s1['gain_ci'][1]:+.2f}] | "
                f"{s1['net']:+5.2f} [{s1['net_ci'][0]:+.2f};{s1['net_ci'][1]:+.2f}] {s2['net']:+5.2f} [{s2['net_ci'][0]:+.2f};{s2['net_ci'][1]:+.2f}] | {w} | {s1['tag']} / {s2['tag']}")
    rep.say("\nDifference de bilan(w=1) entre strates extremes (IC95 % bootstrap par match) et signe dans les deux moities de matchs :")
    for x in diffs:
        if x.get("ok") is None:
            rep.say(f"   {x['dim']} : {x['a']} contre {x['b']} : strate trop petite, non evaluee")
        else:
            rep.say(f"   {x['dim']} : {x['a']} moins {x['b']} : {x['diff']:+.2f} [{x['ci'][0]:+.2f};{x['ci'][1]:+.2f}] ; moities {x['halves'][0]:+.2f} / {x['halves'][1]:+.2f} -> {'ETABLIE' if x['ok'] else 'non etablie'}")
    v, est = verdict(cell_list, diffs)
    rep.say(f"\nVERDICT TABLE DE DECISION : {v}   (dimensions avec difference etablie : {', '.join(est) if est else 'aucune'})")
    rep.say("Lecture : le bilan n'est PAS un optimum ; il dit dans quel sens la balance penche dans chaque situation. Association, 20 matchs.")
    return v


# ------------------------------------------------------------------ interactions score / minute
def interaction_test(d, rep):
    """Le score et la minute modifient-ils le bilan a porteur et zone egaux ? (net15 ~ f_res + controles + interactions)."""
    rep.say("N : LE SCORE ET LE MOMENT CHANGENT-ILS LE BILAN DE LA LIGNE HAUTE A SITUATION EGALE ?")
    ctrl = CTRL + ["libre", "adv", "score_c", "f_min", "f_age"]
    feats = ["f_res"] + ctrl + ["x_libre", "x_adv", "x_score_c", "x_f_min"]
    rep.say(f"{len(d)} moments, {d.match_id.nunique()} matchs. 1 but d'ecart de score = 1 unite ; 1 ecart-type de minute = {d.minute.std():.0f} min ; 1 ecart-type de ligne = {d.res.std():.1f} m.")

    def coefs(s, y):
        names, b, se = ols_cl(s[y].values, s[feats], s.match_id.values)
        return {n: (100 * x, 100 * (x - 1.96 * e), 100 * (x + 1.96 * e)) for n, x, e in zip(names, b, se)}
    rep.say(f"{'resultat':>10s} {'variable':>12s} {'tout':>24s} {'moitie 0':>9s} {'moitie 1':>9s}")
    res = {}
    for y in ("net15", "opp15", "own15"):
        full = coefs(d, y)
        h = [coefs(d[d.half == k], y) for k in (0, 1)]
        for v in ("f_res", "x_libre", "x_adv", "x_score_c", "x_f_min"):
            e = full[v]
            rep.say(f"{y:>10s} {v:>12s} {e[0]:+7.2f} [{e[1]:+.2f};{e[2]:+.2f}] {h[0][v][0]:+9.2f} {h[1][v][0]:+9.2f}")
        res[y] = (full, h)
    full, h = res["net15"]
    ok = {}
    for v in ("x_score_c", "x_f_min"):
        e = full[v]
        ok[v] = (e[1] > 0 or e[2] < 0, np.sign(h[0][v][0]) == np.sign(h[1][v][0]) == np.sign(e[0]))
    est = [v for v in ok if ok[v][0] and ok[v][1]]
    one = [v for v in ok if ok[v][0]]
    verdict_ = "GO" if est else ("A VOIR" if one else "NO-GO")
    rep.say(f"VERDICT INTERACTIONS : {verdict_}   (net15 : score {full['x_score_c'][0]:+.2f} [{full['x_score_c'][1]:+.2f};{full['x_score_c'][2]:+.2f}] ; "
            f"minute {full['x_f_min'][0]:+.2f} [{full['x_f_min'][1]:+.2f};{full['x_f_min'][2]:+.2f}])")
    rep.say("Lecture : score_c = avance de T (positif = T mene). x_score_c < 0 = bilan moins bon quand T mene ; x_f_min > 0 = meilleur en fin de match. "
            "Un NO-GO n'etablit pas l'absence d'effet : 20 matchs.")
    return verdict_, full


# ------------------------------------------------------------------ modele valide hors echantillon
def _oof(d, feats, y, folds):
    p = np.full(len(d), np.nan)
    for f in folds:
        te = d.match_id.isin(f).values
        names, b, _ = logit(d.loc[~te, y].values, d.loc[~te, feats], d.loc[~te, "match_id"].values)
        Xt = np.column_stack([np.ones(te.sum())] + [d.loc[te, c].values for c in feats])
        p[te] = 1 / (1 + np.exp(-np.clip(Xt @ b, -30, 30)))
    return p


def model_validation(d, rep, nb=300, rng=None):
    """La situation, puis la hauteur de ligne, puis ses interactions : AUC hors echantillon (5 blocs de matchs), IC bootstrap des differences."""
    rng = rng or np.random.default_rng(305)
    rep.say("M : UN MODELE DE LA SITUATION PREDIT-IL MIEUX AVEC LA HAUTEUR DE LIGNE ET SES INTERACTIONS ? (hors echantillon, 5 blocs de matchs)")
    s0 = ["f_bx", "f_bx2", "f_ay", "f_dg", "f_dg2", "libre", "score_c", "f_min", "f_age"]
    s1 = s0 + ["f_res"]
    s2 = s1 + ["x_libre", "x_adv", "x_score_c", "x_f_min", "adv"]
    ids = d.match_id.unique()
    folds = np.array_split(rng.permutation(ids), min(5, len(ids)))
    ms = list(ids)
    idx_by = {m: np.where(d.match_id.values == m)[0] for m in ms}
    out = {}
    rep.say(f"{'cible':>6s} {'modele':>22s} {'AUC':>7s} | {'dAUC S1-S0 [IC95 %]':>24s} {'dAUC S2-S1 [IC95 %]':>24s}")
    for y in ("opp15", "own15"):
        P = {k: _oof(d, f, y, folds) for k, f in (("S0", s0), ("S1", s1), ("S2", s2))}
        yy = d[y].values
        d01, d12 = [], []
        for _ in range(nb):
            ix = np.concatenate([idx_by[m] for m in rng.choice(ms, len(ms))])
            a = {k: auc(yy[ix], P[k][ix]) for k in P}
            d01.append(a["S1"] - a["S0"])
            d12.append(a["S2"] - a["S1"])
        a = {k: auc(yy, P[k]) for k in P}
        c01, c12 = pct(d01), pct(d12)
        for k, lab in (("S0", "situation seule"), ("S1", "+ hauteur de ligne"), ("S2", "+ interactions")):
            tail = f"| {a['S1'] - a['S0']:+.4f} [{c01[0]:+.4f};{c01[1]:+.4f}]" if k == "S1" else (
                f"| {'':>24s} {a['S2'] - a['S1']:+.4f} [{c12[0]:+.4f};{c12[1]:+.4f}]" if k == "S2" else "")
            rep.say(f"{y:>6s} {lab:>22s} {a[k]:7.4f} {tail}")
        out[y] = (a["S1"] - a["S0"], c01, a["S2"] - a["S1"], c12)
    first = any(v[0] >= 0.005 and v[1][0] > 0 for v in out.values())
    both = any(v[0] >= 0.005 and v[1][0] > 0 and v[2] >= 0.005 and v[3][0] > 0 for v in out.values())
    v = "GO" if both else ("A VOIR" if first else "NO-GO")
    rep.say(f"VERDICT MODELE : {v}   (la hauteur de ligne ajoute a la situation : {first} ; les interactions ajoutent : {both})")
    rep.say("Lecture : un gain d'AUC faible est normal pour des evenements rares (2 % de tirs) ; le modele sert a verifier que la structure du bilan n'est pas un artefact d'echantillon.")
    return v
