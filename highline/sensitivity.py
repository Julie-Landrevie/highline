"""T2 : le verdict de la ligne haute tient-il quand on change une hypothese ?

REGLE ECRITE ET COMMITEE AVANT TOUT RESULTAT (voir docs/PROTOCOLE.json, "sensitivity").

Quantites suivies (bilan w = 1, points de probabilite par ecart-type de hauteur de ligne residuelle) : ensemble, porteur libre, porteur presse,
et la difference libre - presse. Base = definition du tableau de decision (bloc installe >= 5 s, balle >= 35 m, controles de position du ballon).
Variantes (une hypothese a la fois) :
  age du bloc installe : 3 s, 8 s, 12 s ;
  distance minimale du ballon a son but : 40 m, 45 m (plus strict que 35 m) ;
  controles de position : minimaux (ballon seulement), sans distance au but, plus l'age du bloc ;
  un match retire (20 variantes ; estimation ponctuelle seulement).
Regle de verdict :
  ROBUSTE si, dans toutes les variantes, le signe des quatre quantites est celui de la base ET l'IC95 % (bootstrap par match) de la difference libre - presse exclut 0 ;
  A VOIR  si les signes sont conserves partout mais l'IC de la difference contient 0 dans au moins une variante ;
  FRAGILE si un signe change dans au moins une variante (on dit laquelle).
Aussi rapporte, sans verdict : le poids w* d'un tir concede (gain / cout) au-dela duquel garder la ligne basse est preferable, et le bilan pour w = 0,5 ; 1 ; 2 ; 3.
Non couvert ici : la definition de « libre » / « presse » (niveau de pression du fournisseur) et l'echelle des erreurs de position, fixees en amont dans frames.py.
Les « points par ecart-type » changent un peu d'une variante a l'autre : l'ecart-type est celui des moments retenus.
"""
import numpy as np

from . import decision as dec

NB_DEFAULT, SEED = 500, 515
AGES, BALLS = (3, 8, 12), (40, 45)
CTRL_VARIANTS = {"controles minimaux (ballon)": ["f_bx", "f_bx2"], "sans distance au but": ["f_bx", "f_bx2", "f_ay"], "controles + age du bloc": dec.CTRL + ["f_age"]}
QUANT = ("ensemble", "libre", "presse", "libre - presse")


def _net(c):
    return c[1] - c[0]


def evaluate(d, ctrl=None, nb=NB_DEFAULT, seed=SEED):
    """Bilan (w = 1) pour l'ensemble, le porteur libre, le porteur presse et la difference libre - presse : estimation et IC95 % bootstrap par match."""
    ctrl = dec.CTRL if ctrl is None else ctrl
    rng = np.random.default_rng(seed)
    A = np.column_stack([np.ones(len(d)), d[["f_res"] + ctrl].values])
    Y = np.column_stack([d.opp15.values.astype(float), d.own15.values.astype(float)])
    masks = {"ensemble": np.ones(len(d), bool), "libre": (d.free == 1).values, "presse": (d.free == 0).values}     # pression inconnue (NaN) : ni libre ni presse, comme dans le tableau de decision
    ids = d.match_id.values
    idx_by = {m: np.where(ids == m)[0] for m in np.unique(ids)}

    def net(ix):
        if len(ix) < 50:
            return np.nan
        return _net(100 * np.linalg.lstsq(A[ix], Y[ix], rcond=None)[0][1])
    pt = {k: net(np.where(m)[0]) for k, m in masks.items()}
    pt["libre - presse"] = pt["libre"] - pt["presse"]
    ms = list(idx_by)
    dr = {k: np.empty(nb) for k in QUANT}
    for b in range(nb):
        ix = np.concatenate([idx_by[m] for m in rng.choice(ms, len(ms))])
        for k, m in masks.items():
            dr[k][b] = net(ix[m[ix]])
        dr["libre - presse"][b] = dr["libre"][b] - dr["presse"][b]
    return {k: (float(pt[k]), *[float(x) for x in np.nanpercentile(dr[k], [2.5, 97.5])]) for k in QUANT}


def leave_one_match_out(d, ctrl=None):
    """Difference libre - presse (point) quand on retire un match : retourne (min, max, nombre de matchs qui changent le signe)."""
    ctrl = dec.CTRL if ctrl is None else ctrl
    base = np.sign(_diff_point(d, ctrl))
    vals = [_diff_point(d[d.match_id != m], ctrl) for m in d.match_id.unique()]
    return min(vals), max(vals), int(sum(np.sign(v) != base for v in vals))


def _diff_point(d, ctrl):
    A = np.column_stack([np.ones(len(d)), d[["f_res"] + ctrl].values])
    Y = np.column_stack([d.opp15.values.astype(float), d.own15.values.astype(float)])
    n = lambda m: _net(100 * np.linalg.lstsq(A[m], Y[m], rcond=None)[0][1])
    return n((d.free == 1).values) - n((d.free == 0).values)


def verdict(base, variants, lomo_flips=0):
    """base et variants (dict nom -> resultat de evaluate). Retourne (verdict, liste des variantes qui changent un signe)."""
    sg = {k: np.sign(base[k][0]) for k in QUANT}
    flips = [n for n, r in variants.items() if any(np.sign(r[k][0]) != sg[k] for k in QUANT)]
    if lomo_flips:
        flips.append(f"un match retire ({lomo_flips} matchs changent le signe)")
    if flips:
        return "FRAGILE", flips
    ci0 = [n for n, r in list(variants.items()) + [("base", base)] if r["libre - presse"][1] <= 0 <= r["libre - presse"][2]]
    return ("A VOIR", ci0) if ci0 else ("ROBUSTE", [])


def run(df, rep, nb=NB_DEFAULT, seed=SEED):
    say = rep.say
    say("T2 : OU LE VERDICT BASCULE-T-IL QUAND ON CHANGE UNE HYPOTHESE ? (bilan w = 1, points par ecart-type de ligne residuelle ; IC95 % bootstrap par match)")
    variants, sizes = {}, {}
    d0 = dec.prepare(df, strict=False)
    base = evaluate(d0, nb=nb, seed=seed)
    sizes["base"] = len(d0)
    for a in AGES:
        d = dec.prepare(df, strict=False, min_age=a)
        variants[f"age du bloc >= {a} s"], sizes[f"age du bloc >= {a} s"] = evaluate(d, nb=nb, seed=seed), len(d)
    for x in BALLS:
        d = dec.prepare(df, strict=False, min_ball_x=x)
        variants[f"balle >= {x} m"], sizes[f"balle >= {x} m"] = evaluate(d, nb=nb, seed=seed), len(d)
    for name, ctrl in CTRL_VARIANTS.items():
        variants[name], sizes[name] = evaluate(d0, ctrl=ctrl, nb=nb, seed=seed), len(d0)
    lo, hi, flips = leave_one_match_out(d0)
    say(f"{'variante':>30s} {'n':>6s} | {'ensemble':>20s} {'libre':>20s} {'presse':>20s} {'libre - presse':>22s}")
    for name, r in [("base", base)] + list(variants.items()):
        f = lambda k: f"{r[k][0]:+5.2f} [{r[k][1]:+.2f};{r[k][2]:+.2f}]"
        say(f"{name:>30s} {sizes[name]:6d} | {f('ensemble'):>20s} {f('libre'):>20s} {f('presse'):>20s} {f('libre - presse'):>22s}")
    say(f"{'un match retire (20)':>30s} {len(d0):6d} | difference libre - presse : de {lo:+.2f} a {hi:+.2f} ; matchs qui changent le signe : {flips}")
    A = np.column_stack([np.ones(len(d0)), d0[["f_res"] + dec.CTRL].values])
    Y = np.column_stack([d0.opp15.values.astype(float), d0.own15.values.astype(float)])
    say("Poids d'un tir concede (w) : bilan de la base pour w = 0,5 ; 1 ; 2 ; 3 et w* (gain / cout ; '-' si le cout ou le gain n'est pas positif)")
    for name, m in (("ensemble", np.ones(len(d0), bool)), ("libre", (d0.free == 1).values), ("presse", (d0.free == 0).values)):
        cost, gain = 100 * np.linalg.lstsq(A[m], Y[m], rcond=None)[0][1]
        ws = f"{gain / cost:.1f}" if cost > 0 and gain > 0 else "-"
        say(f"{name:>14s} : " + " ; ".join(f"w={w}: {gain - w * cost:+.2f}" for w in (0.5, 1, 2, 3)) + f" ; w* = {ws}")
    v, why = verdict(base, variants, flips)
    say(f"VERDICT SENSIBILITE : {v}" + (f"   ({'; '.join(why)})" if why else ""))
    say("Lecture : ROBUSTE = aucun signe ne change et la difference libre - presse reste distincte de 0 ; A VOIR = signes conserves mais intervalle large dans certaines variantes ; "
        "FRAGILE = un signe change : a dire dans les limites avec la variante responsable.")
    return dict(verdict=v, base=base, variants=variants, lomo=(lo, hi, flips), why=why)
