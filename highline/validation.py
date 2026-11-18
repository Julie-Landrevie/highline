"""T1 : le verdict par situation fait-il mieux, hors echantillon, qu'une regle unique de hauteur de ligne ?

REGLE ECRITE ET COMMITEE AVANT TOUT RESULTAT (voir aussi docs/PROTOCOLE.json, "validation").

Pourquoi. La video dit « un staff choisit sa hauteur de bloc avec des moyennes ». L'heterogeneite de l'effet (porteur libre contre presse)
est deja estimee et repliquee par moities de matchs ; ce test demande autre chose : separer libre et presse ameliore-t-il la prediction sur
des matchs jamais vus, par rapport a une pente unique ?

Unite : moment defensif de `decision.prepare(df, strict=True)` (bloc installe, balle >= 35 m de son but). Resultat principal : net15 = tir de l'equipe
moins tir adverse dans les 15 s, modele lineaire (meme estimateur que les cellules du tableau de decision).
Modeles (memes controles de position de balle, memes effets principaux `libre` et `adv`) :
  G  : une pente de hauteur de ligne residuelle (f_res) ;
  S1 : G + f_res x libre  (la pente depend du porteur) ;
  S2 : S1 + f_res x adv   (la pente depend aussi de la zone ; secondaire).
Validation : on laisse de cote un match a la fois (un pli par match). Mesure : reduction relative de l'erreur quadratique, 100 x (SSE_G - SSE_S1) / SSE_G
(positive = S1 fait mieux), IC95 % par bootstrap sur les MATCHS (pas sur les moments : plusieurs moments d'une meme possession se ressemblent).
Verdict (principal, S1 contre G) :
  GO     si la borne basse de l'IC est > 0 ;
  A VOIR si la reduction est > 0 mais l'IC contient 0 ;
  NO-GO  si la reduction est <= 0.
Information, sans verdict : S2 contre G ; perte logarithmique (regression logistique) de tir adverse et de tir de l'equipe, S1 contre G ; test de
permutation (le porteur est tire au hasard parmi les moments d'un meme match ; on compte la part des tirages qui font au moins aussi bien).
Les modeles ne sont pas ajustes apres avoir vu le resultat. Limite connue : les variables standardisees (zs) utilisent la moyenne et l'ecart-type de
tout l'echantillon, sans les resultats (fuite negligeable).
"""
import numpy as np

from .decision import CTRL

BASE = CTRL + ["libre", "adv"]
FEATS = dict(G=["f_res"] + BASE, S1=["f_res"] + BASE + ["x_libre"], S2=["f_res"] + BASE + ["x_libre", "x_adv"])
NB_DEFAULT, NPERM_DEFAULT, SEED = 2000, 200, 4242


def _X(d, feats):
    return np.column_stack([np.ones(len(d))] + [d[c].values.astype(float) for c in feats])


def _fit_logit(X, y, iters=25, ridge=1e-6):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    return b


def _lomo(X, y, ids, kind="ols"):
    """Un pli par match : retourne (matchs, n, perte totale du match). OLS : somme des carres ; logit : somme de la perte logarithmique."""
    ms = np.unique(ids)
    n, loss = [], []
    for m in ms:
        te = ids == m
        if kind == "ols":
            b = np.linalg.lstsq(X[~te], y[~te], rcond=None)[0]
            loss.append(float(((y[te] - X[te] @ b) ** 2).sum()))
        else:
            b = _fit_logit(X[~te], y[~te])
            p = np.clip(1 / (1 + np.exp(-np.clip(X[te] @ b, -30, 30))), 1e-9, 1 - 1e-9)
            loss.append(float(-(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)).sum()))
        n.append(int(te.sum()))
    return ms, np.array(n), np.array(loss)


def _rel_gain(lg, ls):
    return 100 * (lg.sum() - ls.sum()) / lg.sum()


def _boot(lg, ls, nb, rng):
    k = len(lg)
    out = np.empty(nb)
    for i in range(nb):
        ix = rng.integers(0, k, k)
        out[i] = _rel_gain(lg[ix], ls[ix])
    return np.percentile(out, [2.5, 97.5])


def _verdict(point, lo):
    return "GO" if lo > 0 else ("A VOIR" if point > 0 else "NO-GO")


def situation_vs_rule(d, rep=None, nb=NB_DEFAULT, nperm=NPERM_DEFAULT, seed=SEED):
    """d : sortie de decision.prepare(df, strict=True). Retourne dict(verdict, gain, ci, ...) ; ecrit le detail dans rep si fourni."""
    say = rep.say if rep is not None else (lambda s: None)
    rng = np.random.default_rng(seed)
    d = d.reset_index(drop=True)
    ids = d.match_id.values
    y = d.net15.values.astype(float)
    say("T1 : LE VERDICT PAR SITUATION FAIT-IL MIEUX QU'UNE REGLE UNIQUE, HORS ECHANTILLON ? (un match laisse de cote a la fois)")
    say(f"{len(d)} moments, {len(np.unique(ids))} matchs ; resultat principal : net15 (tir de l'equipe moins tir adverse, 15 s), erreur quadratique.")
    _, n, lg = _lomo(_X(d, FEATS["G"]), y, ids)
    _, _, l1 = _lomo(_X(d, FEATS["S1"]), y, ids)
    _, _, l2 = _lomo(_X(d, FEATS["S2"]), y, ids)
    g1, g2 = _rel_gain(lg, l1), _rel_gain(lg, l2)
    c1, c2 = _boot(lg, l1, nb, rng), _boot(lg, l2, nb, rng)
    v = _verdict(g1, c1[0])
    say(f"  S1 (pente selon le porteur) contre G (une pente) : reduction de l'erreur {g1:+.3f} % [{c1[0]:+.3f} ; {c1[1]:+.3f}]")
    say(f"  S2 (porteur et zone) contre G, information : {g2:+.3f} % [{c2[0]:+.3f} ; {c2[1]:+.3f}]")
    sec = {}
    for tgt in ("opp15", "own15"):
        yy = d[tgt].values.astype(float)
        _, _, a = _lomo(_X(d, FEATS["G"]), yy, ids, "logit")
        _, _, b = _lomo(_X(d, FEATS["S1"]), yy, ids, "logit")
        gg, cc = _rel_gain(a, b), _boot(a, b, nb, rng)
        sec[tgt] = (gg, tuple(cc))
        say(f"  perte logarithmique, {tgt}, S1 contre G, information : {gg:+.3f} % [{cc[0]:+.3f} ; {cc[1]:+.3f}]")
    perm = []
    for _ in range(nperm):
        dp = d.copy()
        dp["libre"] = d.groupby("match_id")["libre"].transform(lambda s: rng.permutation(s.values))
        dp["x_libre"] = dp.f_res * dp.libre
        d2 = d.copy()
        d2["x_libre"] = dp["x_libre"]
        _, _, lp = _lomo(_X(d2, FEATS["S1"]), y, ids)
        perm.append(_rel_gain(lg, lp))
    p = (1 + sum(x >= g1 for x in perm)) / (1 + len(perm)) if perm else float("nan")
    say(f"  permutation du porteur dans chaque match ({nperm} tirages), information : p = {p:.3f} (part des tirages au moins aussi bons)")
    txt = {"GO": "separer libre et presse ameliore la prediction hors echantillon.",
           "A VOIR": "la separation va dans le bon sens hors echantillon mais n'est pas etablie avec ces matchs.",
           "NO-GO": "la separation n'ameliore pas la prediction hors echantillon : ne pas affirmer que le verdict par situation fait mieux que la moyenne."}[v]
    say(f"VERDICT T1 : {v}   ({txt})")
    return dict(verdict=v, gain=float(g1), ci=(float(c1[0]), float(c1[1])), s2=(float(g2), (float(c2[0]), float(c2[1]))), secondary=sec, perm_p=float(p), n=int(n.sum()))
