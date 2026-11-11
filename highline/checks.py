"""Controles de credibilite : les chiffres du tableau de decision mesurent-ils ce qu'ils annoncent ?

Regles fixees AVANT de regarder les resultats (GO / A VOIR / NO-GO) ; les analyses marquees (post hoc) ne sont pas dans les hypotheses de depart.

C1  Alignement tracking / evenements. Pour chaque engagement sur le porteur, la distance entre les deux joueurs lue dans le tracking est comparee a
    celle que donne le fichier d'evenements (`interplayer_distance_start`). GO si la mediane de l'ecart est <= 0,10 m et si >= 95 % des ecarts
    sont <= 0,5 m ; A VOIR si >= 90 % sont <= 0,5 m ; NO-GO sinon. Decalages de -3 a +3 images testes : le meilleur doit etre 0.
C2  Etiquette des tirs. Chaque possession qui se termine par un tir (`end_type == "shot"`) doit etre marquee `lead_to_shot` par le fournisseur.
    GO si l'accord est >= 98 %.
C3  Reference de but (post hoc, decouverte en faisant C1/C2). Notre hauteur de ligne est comparee a `last_defensive_line_height_start` du
    fournisseur (distance a la ligne de but du defenseur de champ le plus bas). GO si la mediane de l'ecart est <= 1 m avec la reference "ligne de but".
C4  Placebo. Dans chaque equipe-match, la hauteur de ligne residuelle est decalee circulairement : elle n'a plus aucun lien avec les tirs. On applique les
    MEMES intervalles que le tableau de decision (bootstrap par match). GO si, sur les trois resultats (tir adverse, tir de l'equipe, bilan), au plus 10 % des
    intervalles excluent 0 (taux nominal 5 %). Un intervalle analytique groupe par match et une permutation simple sont rapportes en complement, ainsi que le
    gain d'AUC du modele contre celui d'une ligne placebo.
C5  Faux positifs attendus. Nombre d'intervalles du tableau de decision, nombre de ceux qui excluent 0, nombre attendu par hasard (5 %) : information, pas de verdict.
C6  Contexte fournisseur (post hoc). Le bilan de la ligne haute survit-il au controle de la valeur de possession du fournisseur (`possession_epv_for_start`) ?
    STABLE si, pour l'ensemble, le porteur libre et le porteur presse, le signe du bilan est inchange et si le changement est <= max(0,3 point ; 30 % du bilan de base).
    (xloss de SkillCorner n'existe que sur les engagements, donc pas sur chaque moment : l'EPV de la possession est le contexte disponible partout.)

C7  Observation de la ligne (post hoc, inspire de la comparaison « avec / sans camera » de `backline`, nowayfootball). Trois lectures :
    (a) couverture : part des moments ou les 4 defenseurs les plus bas (positions estimees comprises pour les classer) sont tous detectes, par bande de hauteur de ligne ;
    (b) selection : a position de balle egale (cases de 5 m x 2 bandes laterales), ecart de hauteur de ligne entre moments ou les 4 sont detectes et les autres ; bootstrap par match.
        Information, pas de verdict : l'ecart reflete surtout ce que la camera voit (part observee selon le tercile de ligne), pas une erreur d'estimation ;
    (c) robustesse : le bilan de la ligne haute, recalcule sur les seuls moments ou les 4 plus bas sont detectes. STABLE si le signe est inchange pour l'ensemble,
        le porteur libre et le porteur presse (moins de moments : intervalles plus larges, a lire comme tels).

Toutes les analyses sont rapportees quel que soit le resultat.
"""
import numpy as np
import pandas as pd

from . import config as cfg
from . import data as dat
from . import decision as dec
from scipy.stats import t as student_t

from .stats import auc, pct, zs

FP_MAX = 0.10
ALPHA = 0.05
OFFSETS = (-3, -2, -1, 0, 1, 2, 3)


# ---------------------------------------------------------------------------------------------------------------- C1, C2, C3
_SCAN = {}


def _scan(mid):
    """Une seule lecture du tracking par match pour C1 (engagements) et C3 (ligne fournisseur) ; resultat garde en memoire."""
    if mid in _SCAN:
        return _SCAN[mid]
    T = dat.load_match(mid)
    ev = dat.load_events(mid)
    X, Y, D = T["X"], T["Y"], T["D"]
    eng = []
    e = ev[ev.event_type == "on_ball_engagement"].dropna(subset=["frame_start", "player_id", "player_in_possession_id", "interplayer_distance_start"])
    for r in e.itertuples():
        a, b = T["cidx"].get(int(r.player_id)), T["cidx"].get(int(r.player_in_possession_id))
        if a is None or b is None:
            continue
        f = int(r.frame_start) - T["f0"]
        rec = dict(match_id=mid, reported=float(r.interplayer_distance_start), seen=bool(D[f, a] == 1 and D[f, b] == 1) if 0 <= f < T["n"] else False)
        for k in OFFSETS:
            g = f + k
            ok = 0 <= g < T["n"] and np.isfinite([X[g, a], Y[g, a], X[g, b], Y[g, b]]).all()
            rec[f"m{k}"] = float(np.hypot(X[g, a] - X[g, b], Y[g, a] - Y[g, b])) if ok else np.nan
        eng.append(rec)
    p = ev[ev.event_type == "player_possession"].dropna(subset=["frame_start", "last_defensive_line_height_start", "team_id"])
    g_line, g_gk = dat.own_goal_sign(T, "line"), dat.own_goal_sign(T, "gk_median")
    offs = [abs(g_line[k]) - abs(g_gk[k]) for k in g_line if np.isfinite(g_line[k]) and np.isfinite(g_gk[k])]
    lines = []
    for r in p.itertuples():
        f = int(r.frame_start) - T["f0"]
        if not 0 <= f < T["n"] or T["PER"][f] == 0:
            continue
        for tm in np.unique(T["team"]):
            if tm == r.team_id:
                continue
            c = np.where((T["team"] == tm) & (T["grp"] != "GK") & np.isfinite(X[f]))[0]
            gl, gg = g_line.get((T["PER"][f], tm)), g_gk.get((T["PER"][f], tm))
            if len(c) < 8 or gl is None or not np.isfinite(gl) or not np.isfinite(gg):
                continue
            sgn = 1.0 if gl < 0 else -1.0
            deep_line = np.sort(sgn * (X[f, c] - gl))
            deep_gk = np.sort(sgn * (X[f, c] - gg))
            lines.append(dict(h=float(r.last_defensive_line_height_start), line=deep_line[0], gk=deep_gk[0], l4=deep_line[:4].mean()))
    _SCAN[mid] = (eng, lines, offs)
    return _SCAN[mid]


def alignment(ids, rep):
    """C1 : le tracking et le fichier d'evenements decrivent-ils le meme instant ?"""
    rep.say("C1 : ALIGNEMENT TRACKING / EVENEMENTS (distance porteur-presseur lue dans le tracking contre `interplayer_distance_start`)")
    rows = []
    for mid in ids:
        rows += _scan(mid)[0]
    df = pd.DataFrame(rows)
    if df.empty:
        rep.say("   non evalue (aucun engagement exploitable)")
        return dict(verdict="n/d")
    err = {k: (df[f"m{k}"] - df.reported).abs() for k in OFFSETS}
    e0 = err[0].dropna()
    med, w05, o2 = float(e0.median()), float((e0 <= 0.5).mean()), float((e0 > 2).mean())
    best = min(OFFSETS, key=lambda k: err[k].median())
    rep.say(f"{len(e0)} engagements dans {df.match_id.nunique()} matchs ; ecart median {med:.3f} m ; {100 * w05:.1f} % a moins de 0,5 m ; {100 * o2:.1f} % a plus de 2 m")
    for lab, mask in (("deux joueurs observes", df.seen), ("au moins un extrapole", ~df.seen)):
        s = err[0][mask].dropna()
        if len(s):
            rep.say(f"   {lab:<24s} n = {len(s):5d} ; ecart median {s.median():.3f} m ; {100 * (s <= 0.5).mean():.1f} % a moins de 0,5 m")
    rep.say("   ecart median selon le decalage (images) : " + " ; ".join(f"{k:+d} : {err[k].median():.3f}" for k in OFFSETS) + f"   -> meilleur decalage : {best:+d}")
    v = "GO" if (med <= 0.10 and w05 >= 0.95 and best == 0) else ("A VOIR" if w05 >= 0.90 and best == 0 else "NO-GO")
    rep.say(f"VERDICT C1 : {v}")
    return dict(verdict=v, n=int(len(e0)), median=med, within05=w05, best=best)


def label_check(ids, rep):
    """C2 : nos tirs (fin de possession `shot`) sont-ils ceux que le fournisseur relie a un tir ?"""
    rep.say("C2 : ETIQUETTE DES TIRS (possessions qui finissent par un tir, contre `lead_to_shot` du fournisseur)")
    n_shot = n_flag = n_lead = 0
    for mid in ids:
        ev = dat.load_events(mid)
        p = ev[ev.event_type == "player_possession"]
        s = p[p.end_type == "shot"]
        n_shot += len(s)
        n_flag += int((s.get("lead_to_shot").map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0) == 1).sum()) if "lead_to_shot" in p else 0
        n_lead += int((p.get("lead_to_shot").map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0) == 1).sum()) if "lead_to_shot" in p else 0
    if n_shot == 0:
        rep.say("   non evalue (aucun tir)")
        return dict(verdict="n/d")
    share = n_flag / n_shot
    rep.say(f"{n_shot} possessions finissant par un tir dans {len(ids)} matchs ; {100 * share:.1f} % marquees `lead_to_shot` par le fournisseur ; "
            f"{n_lead} possessions marquees au total (les passes qui precedent un tir le sont aussi : chaque tir n'est compte qu'une fois chez nous)")
    v = "GO" if share >= 0.98 else ("A VOIR" if share >= 0.9 else "NO-GO")
    rep.say(f"VERDICT C2 : {v}")
    return dict(verdict=v, n=n_shot, share=share)


def line_convention(ids, rep):
    """C3 (post hoc) : notre hauteur de ligne contre `last_defensive_line_height_start`, avec les deux references de but."""
    rep.say("C3 (post hoc) : REFERENCE DE BUT ET HAUTEUR DE LIGNE (defenseur de champ le plus bas, contre `last_defensive_line_height_start` du fournisseur)")
    rows, offs = [], []
    for mid in ids:
        _, lines, o = _scan(mid)
        rows += lines
        offs += o
    df = pd.DataFrame(rows)
    if df.empty:
        rep.say("   non evalue")
        return dict(verdict="n/d")
    d_line, d_gk = (df.h - df.line).abs(), (df.h - df.gk).abs()
    r4 = float(np.corrcoef(df.h, df.l4)[0, 1])
    rep.say(f"{len(df)} possessions ; ecart au defenseur le plus bas du fournisseur : reference 'ligne de but' : mediane {d_line.median():.2f} m, p90 {d_line.quantile(.9):.2f} m ; "
            f"reference 'position mediane du gardien' (premiere version) : mediane {d_gk.median():.2f} m, p90 {d_gk.quantile(.9):.2f} m")
    rep.say(f"   correlation entre notre moyenne des 4 plus bas et la hauteur fournisseur : {r4:.3f} ; "
            f"ecart moyen entre la ligne de but et la position mediane du gardien : {np.mean(offs):.1f} m (de {np.min(offs):.1f} a {np.max(offs):.1f} m sur {len(offs)} equipes-periodes)")
    v = "GO" if d_line.median() <= 1.0 else "NO-GO"
    rep.say(f"VERDICT C3 : {v}   (une reference decalee de {np.mean(offs):.0f} m deplacerait les seuils de zone '35 m' et '70 m' d'autant ; la reference par defaut est la ligne de but)")
    return dict(verdict=v, median_line=float(d_line.median()), median_gk=float(d_gk.median()), offset_mean=float(np.mean(offs)), corr4=r4)


# ---------------------------------------------------------------------------------------------------------------- C4
def _cluster_se(A, Y, starts):
    """Moindres carres ; erreur-type robuste groupee (par match) du coefficient de la colonne 1 de A, pour chaque colonne de Y."""
    inv = np.linalg.inv(A.T @ A + 1e-9 * np.eye(A.shape[1]))
    B = inv @ (A.T @ Y)
    E = Y - A @ B
    se = np.empty(Y.shape[1])
    for k in range(Y.shape[1]):
        S = np.add.reduceat(A * E[:, k][:, None], starts, axis=0)
        se[k] = np.sqrt(abs((inv @ (S.T @ S) @ inv)[1, 1]))
    return B[1], se


def shift_within(values, blocks, rng):
    """Decalage circulaire aleatoire de `values` dans chaque bloc (conserve la loi et l'autocorrelation, rompt le lien avec les tirs)."""
    out = np.asarray(values, float).copy()
    for ix in blocks:
        if len(ix) >= 8:
            out[ix] = np.roll(values[ix], int(rng.integers(len(ix) // 4, max(len(ix) // 4 + 1, 3 * len(ix) // 4))))
    return out


def permute_within(values, blocks, rng):
    out = np.asarray(values, float).copy()
    for ix in blocks:
        out[ix] = values[rng.permutation(ix)]
    return out


def placebo_effect(d, rep, nperm=200, seed=7):
    """C4 : une hauteur de ligne sans lien avec les tirs produit-elle des effets 'significatifs' ? (taux de faux positifs des intervalles groupes par match)."""
    rep.say("C4 : PLACEBO (hauteur de ligne residuelle decalee ou permutee dans chaque match ; aucun lien reel avec les tirs)")
    dd = d.sort_values(["match_id", "team"], kind="stable").reset_index(drop=True)
    names = ["tir adverse (cout)", "tir de l'equipe (gain)", "bilan (gain - cout)"]
    A = np.column_stack([np.ones(len(dd)), dd.f_res.values] + [dd[c].values for c in dec.CTRL])
    Y = np.column_stack([dd.opp15, dd.own15, dd.net15]).astype(float)
    mid = dd.match_id.values
    starts = np.r_[0, np.flatnonzero(mid[1:] != mid[:-1]) + 1]
    b, se = _cluster_se(A, Y, starts)
    real, real_t = 100 * b, b / se
    team_blocks = list(dd.groupby(["match_id", "team"]).indices.values())
    match_blocks = list(dd.groupby("match_id").indices.values())
    rng = np.random.default_rng(seed)
    res = {}
    crit = float(student_t.ppf(0.975, max(1, dd.match_id.nunique() - 1)))
    rep.say(f"{len(dd)} moments, {dd.match_id.nunique()} matchs, {nperm} tirages par schema ; effet reel (points par ecart-type) : "
            + " ; ".join(f"{n} {v:+.2f} (t = {t:+.1f})" for n, v, t in zip(names, real, real_t)))
    rep.say(f"Intervalle analytique groupe par match, valeur critique de Student a {dd.match_id.nunique() - 1} ddl = {crit:.2f} (1,96 sous-estimerait l'incertitude avec peu de matchs).")
    rep.say(f"{'schema':>22s} {'resultat':>24s} {'effet placebo moyen':>20s} {'[2,5 ; 97,5] %':>18s} {'intervalles excluant 0':>24s} {'p (placebo >= reel)':>20s}")
    for lab, fn, blocks in (("decalage circulaire", shift_within, team_blocks), ("permutation", permute_within, match_blocks)):
        eff, tt = np.empty((nperm, 3)), np.empty((nperm, 3))
        for i in range(nperm):
            A2 = A.copy()
            A2[:, 1] = fn(dd.f_res.values, blocks, rng)
            bb, ss = _cluster_se(A2, Y, starts)
            eff[i], tt[i] = 100 * bb, bb / ss
        fp = (np.abs(tt) > crit).mean(axis=0)
        pv = (1 + (np.abs(eff) >= np.abs(real)).sum(axis=0)) / (1 + nperm)
        for k, n in enumerate(names):
            lo, hi = np.percentile(eff[:, k], [2.5, 97.5])
            rep.say(f"{lab:>22s} {n:>24s} {eff[:, k].mean():+20.2f} {lo:+8.2f};{hi:+7.2f} {100 * fp[k]:>22.1f} % {pv[k]:>20.3f}")
        res[lab] = dict(fp=fp.tolist(), p=pv.tolist())
    return dict(real=real.tolist(), **res)


def placebo_bootstrap(d, rep, nperm=60, nb=100, seed=13):
    """C4 : MEMES intervalles que le tableau de decision (bootstrap par match, percentiles) appliques a une ligne placebo : quel taux excluent 0 ?"""
    rep.say("C4 (intervalles du tableau) : INTERVALLES BOOTSTRAP PAR MATCH SUR UNE LIGNE PLACEBO (decalage circulaire dans chaque equipe-match)")
    dd = d.reset_index(drop=True)
    A = np.column_stack([np.ones(len(dd)), dd.f_res.values] + [dd[c].values for c in dec.CTRL])
    Y = np.column_stack([dd.opp15.values, dd.own15.values]).astype(float)
    by_match = {m: np.where(dd.match_id.values == m)[0] for m in dd.match_id.unique()}
    ms = list(by_match)
    blocks = list(dd.groupby(["match_id", "team"]).indices.values())
    rng = np.random.default_rng(seed)
    names = ["tir adverse (cout)", "tir de l'equipe (gain)", "bilan (gain - cout)"]
    hits = np.zeros(3)
    for _ in range(nperm):
        A[:, 1] = shift_within(dd.f_res.values, blocks, rng)
        dr = []
        for _ in range(nb):
            ix = np.concatenate([by_match[m] for m in rng.choice(ms, len(ms))])
            dr.append(100 * np.linalg.lstsq(A[ix], Y[ix], rcond=None)[0][1])
        dr = np.asarray(dr)
        for k, col in enumerate((dr[:, 0], dr[:, 1], dr[:, 1] - dr[:, 0])):
            lo, hi = np.percentile(col, [2.5, 97.5])
            hits[k] += lo > 0 or hi < 0
    fp = hits / nperm
    rep.say(f"{nperm} lignes placebo x {nb} tirages bootstrap par match ({len(ms)} matchs) : part des intervalles qui excluent 0 (5 % attendu) : "
            + " ; ".join(f"{n} {100 * f:.1f} %" for n, f in zip(names, fp)))
    v = "GO" if (fp <= FP_MAX).all() else "A VOIR"
    rep.say(f"VERDICT C4 (intervalles) : {v}   (GO si au plus {100 * FP_MAX:.0f} % pour les 3 resultats ; un bootstrap par match sur peu de matchs est instable, a lire avec prudence)")
    return dict(verdict=v, fp=fp.tolist())


def placebo_model(ds, rep, nperm=30, seed=11):
    """C4 (modele) : le gain d'AUC apporte par la hauteur de ligne depasse-t-il celui d'une ligne placebo ?"""
    rep.say("C4 (modele) : GAIN D'AUC DE LA HAUTEUR DE LIGNE CONTRE UNE LIGNE PLACEBO (memes blocs de matchs que le modele M)")
    s0 = ["f_bx", "f_bx2", "f_ay", "f_dg", "f_dg2", "libre", "score_c", "f_min", "f_age"]
    s1 = s0 + ["f_res"]
    ds = ds.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    ids = ds.match_id.unique()
    folds = np.array_split(rng.permutation(ids), min(5, len(ids)))
    blocks = list(ds.groupby(["match_id", "team"]).indices.values())
    out = {}
    rep.say(f"{'cible':>6s} {'gain reel':>10s} {'placebo moyen':>14s} {'placebo p95':>12s} {'p (placebo >= reel)':>20s}")
    for y in ("opp15", "own15"):
        yy = ds[y].values
        base = auc(yy, dec._oof(ds, s0, y, folds))
        real = auc(yy, dec._oof(ds, s1, y, folds)) - base
        pl = []
        for _ in range(nperm):
            sh = ds.copy()
            sh["f_res"] = shift_within(ds.f_res.values, blocks, rng)
            pl.append(auc(yy, dec._oof(sh, s1, y, folds)) - base)
        pl = np.asarray(pl)
        p = (1 + (pl >= real).sum()) / (1 + nperm)
        rep.say(f"{y:>6s} {real:+10.4f} {pl.mean():+14.4f} {np.percentile(pl, 95):+12.4f} {p:>20.3f}")
        out[y] = dict(real=float(real), placebo_mean=float(pl.mean()), p=float(p))
    rep.say("Lecture : un gain d'AUC reel dans la queue du placebo (p petit) est de l'information ; sinon il peut venir de la capacite du modele.")
    return out


# ---------------------------------------------------------------------------------------------------------------- C5
def multiplicity(cell_list, diffs, rep):
    """C5 : combien d'intervalles ont ete regardes, et combien excluraient 0 par hasard ?"""
    tested = sig = 0
    for c in cell_list:
        s = dec.summarize(c, 1.0)
        for lo, hi in (s["cost_ci"], s["gain_ci"], s["net_ci"]):
            if np.isfinite(lo):
                tested += 1
                sig += int(lo > 0 or hi < 0)
    nd = sum(1 for x in diffs if x.get("ok") is not None)
    exp = ALPHA * (tested + nd)
    rep.say(f"C5 : FAUX POSITIFS ATTENDUS. {tested} intervalles (cout, gain, bilan de {len(cell_list)} situations) et {nd} differences entre strates ont ete examines ; "
            f"{sig} intervalles excluent 0 ; sans aucun effet reel, environ {exp:.0f} le feraient par hasard (indicatif : les situations se recouvrent, "
            f"les intervalles ne sont pas independants). Les conclusions du README reposent sur les differences entre strates et la replication par moities, pas sur chaque intervalle.")
    return dict(tested=tested, nd=nd, significant=sig, expected=exp)


# ---------------------------------------------------------------------------------------------------------------- C6
def _effects(d, mask, cols, nb, rng, by_match):
    A = np.column_stack([np.ones(len(d)), d.f_res.values] + [d[c].values for c in cols])
    Y = np.column_stack([d.opp15.values, d.own15.values]).astype(float)

    def coefs(ix):
        if len(ix) < 50:
            return np.array([np.nan, np.nan])
        return 100 * np.linalg.lstsq(A[ix], Y[ix], rcond=None)[0][1]
    point = coefs(np.where(mask)[0])
    ms = list(by_match)
    draws = np.array([coefs((lambda ix: ix[mask[ix]])(np.concatenate([by_match[m] for m in rng.choice(ms, len(ms))]))) for _ in range(nb)])
    return point, draws


def context_control(d, rep, nb=300, seed=21):
    """C6 (post hoc) : le bilan resiste-t-il au controle de la valeur de possession du fournisseur ?"""
    rep.say("C6 (post hoc) : CONTEXTE DU FOURNISSEUR (valeur de possession `possession_epv_for_start` ajoutee aux controles)")
    if "epv0" not in d or d.epv0.notna().sum() < 400:
        rep.say("   non evalue (colonne absente ou trop peu de valeurs)")
        return dict(verdict="n/d")
    d = d.copy().reset_index(drop=True)
    d["f_epv"] = zs(d.epv0.fillna(d.epv0.median()))
    rng = np.random.default_rng(seed)
    by_match = {m: np.where(d.match_id.values == m)[0] for m in d.match_id.unique()}
    strata = {"ensemble": np.ones(len(d), bool), "porteur libre": (d.free == 1).values, "porteur presse": (d.free == 0).values}
    variants = (("base", dec.CTRL, None), ("+ EPV", dec.CTRL + ["f_epv"], None))
    if "org0" in d and d.org0.notna().sum() >= 400:
        org = (d.org0 == 1).values
        rep.say(f"Accord avec la notion fournisseur de defense organisee : {100 * d.org0.mean():.0f} % de nos moments 'bloc installe' sont marques `organised_defense` ({int(d.org0.notna().sum())} valeurs).")
    else:
        org = None
    rep.say(f"{'situation':>16s} {'variante':>12s} {'n':>6s} | {'cout':>20s} {'gain':>20s} {'bilan w=1':>22s}")
    res, stable = {}, True
    for sname, smask in strata.items():
        net, ci = {}, {}
        for vname, cols, _ in variants + ((("organisee seule", dec.CTRL, org),) if org is not None else ()):
            m = smask & (org if vname == "organisee seule" else True)
            if m.sum() < cfg.MIN_CELL:
                continue
            pt, dr = _effects(d, m, cols, nb, rng, by_match)
            lc, hc = pct(dr[:, 0])
            lg, hg = pct(dr[:, 1])
            ln, hn = pct(dr[:, 1] - dr[:, 0])
            rep.say(f"{sname:>16s} {vname:>12s} {int(m.sum()):6d} | {pt[0]:+5.2f} [{lc:+.2f};{hc:+.2f}] {pt[1]:+5.2f} [{lg:+.2f};{hg:+.2f}] {pt[1] - pt[0]:+6.2f} [{ln:+.2f};{hn:+.2f}]")
            net[vname], ci[vname] = pt[1] - pt[0], (ln, hn)
        if "base" in net and "+ EPV" in net:
            b0, b1 = net["base"], net["+ EPV"]
            ok = bool(np.sign(b0) == np.sign(b1) and abs(b1 - b0) <= max(0.3, 0.3 * abs(b0)))
            stable &= ok
            res[sname] = dict(base=b0, epv=b1, ok=ok)
            if "organisee seule" in net:
                res[sname].update(org=net["organisee seule"], org_ci=ci["organisee seule"], base_ci=ci["base"])
    v = "STABLE" if stable and res else "A VOIR"
    rep.say(f"VERDICT C6 : {v}   (signe du bilan inchange et changement <= max(0,3 point ; 30 %) pour l'ensemble, le porteur libre et le porteur presse, valeur de possession ajoutee)")
    weak = [k for k, r in res.items() if "org" in r and (np.sign(r["org"]) != np.sign(r["base"]) or not (r["org_ci"][0] > 0 or r["org_ci"][1] < 0))]
    if weak:
        rep.say("ATTENTION (post hoc, hors regle de verdict) : restreint aux moments que le fournisseur juge en defense organisee, le bilan perd son intervalle excluant 0 ou change de signe pour : "
                + ", ".join(f"{k} ({res[k]['base']:+.2f} -> {res[k]['org']:+.2f})" for k in weak)
                + ". Le resultat vaut surtout pour les phases de pression active ; a dire dans les limites.")
    return dict(verdict=v, strata=res, weak_when_organised=weak)


def line_observation(d, rep, nb=300, seed=31):
    """C7 (post hoc) : la hauteur de ligne est-elle biaisee quand des defenseurs ne sont pas observes, et le bilan tient-il sur les moments observes ?"""
    rep.say("C7 (post hoc) : OBSERVATION DES 4 DEFENSEURS LES PLUS BAS (detectes par la camera) ET HAUTEUR DE LIGNE")
    if "n_det_back4" not in d or d.n_det_back4.notna().sum() < 400:
        rep.say("   non evalue (colonne `n_det_back4` absente)")
        return dict(verdict="n/d")
    d = d.copy().reset_index(drop=True)
    seen = (d.n_det_back4 == 4).values
    bands = [("ligne < 25 m", d.line < 25), ("ligne 25-40 m", (d.line >= 25) & (d.line < 40)), ("ligne >= 40 m", d.line >= 40), ("ensemble", d.line.notna())]
    rep.say(f"(a) couverture, moments 'bloc installe' ({len(d)} moments, {d.match_id.nunique()} matchs) :")
    cov = {}
    for name, m in bands:
        m = m.values
        if m.sum() == 0:
            continue
        cov[name] = dict(n=int(m.sum()), back4=float(seen[m].mean()), all10=float((d.n_det.values[m] >= 10).mean()))
        rep.say(f"   {name:>14s} : n = {cov[name]['n']:6d} ; 4 plus bas tous detectes : {100 * cov[name]['back4']:5.1f} % ; 10 joueurs de champ detectes : {100 * cov[name]['all10']:5.1f} %")
    cell = (np.floor(d.ball_x.values / 5).astype(int) * 2 + (d.ball_ay.values >= 11)).astype(int)

    def gap(ix):
        g, w = [], []
        for c in np.unique(cell[ix]):
            k = ix[cell[ix] == c]
            a, b = k[seen[k]], k[~seen[k]]
            if len(a) >= 30 and len(b) >= 30:
                g.append(d.line.values[a].mean() - d.line.values[b].mean())
                w.append(len(a))
        return np.average(g, weights=w) if w else np.nan
    rng = np.random.default_rng(seed)
    by_match = {m: np.where(d.match_id.values == m)[0] for m in d.match_id.unique()}
    ms = list(by_match)
    allix = np.arange(len(d))
    pt = gap(allix)
    draws = np.array([gap(np.concatenate([by_match[m] for m in rng.choice(ms, len(ms))])) for _ in range(nb)])
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    rep.say(f"(b) a position de balle egale, hauteur de ligne (tous detectes) moins (au moins un estime) : {pt:+.2f} m [{lo:+.2f} ; {hi:+.2f}] (bootstrap par match ; > 0 : la ligne est plus haute quand les 4 sont observes)")
    rk = pd.Series(d.line.values).groupby(cell).rank(pct=True).values
    terc = np.digitize(rk, [1 / 3, 2 / 3])
    p_obs = [float(seen[terc == k].mean()) if (terc == k).any() else np.nan for k in range(3)]
    rep.say(f"    Lecture : c'est de la SELECTION par la camera, pas une erreur d'estimation demontree. A position de balle egale, la part de moments ou les 4 plus bas sont observes est de "
            f"{100 * p_obs[0]:.0f} % / {100 * p_obs[1]:.0f} % / {100 * p_obs[2]:.0f} % pour une ligne basse / moyenne / haute : observer, c'est voir surtout des lignes proches du ballon, donc plus hautes. "
            "Les positions dans les trous sont ancrees aux deux bouts : leur erreur ne se mesure pas directement (voir docs/rapport_full.txt, fiabilite).")
    rep.say("(c) bilan de la ligne haute (points par ecart-type, w = 1), tous moments puis moments ou les 4 plus bas sont detectes :")
    strata = {"ensemble": np.ones(len(d), bool), "porteur libre": (d.free == 1).values, "porteur presse": (d.free == 0).values}
    res, stable = {}, True
    for sname, smask in strata.items():
        row = {}
        for vname, m in (("tous", smask), ("observes", smask & seen)):
            if m.sum() < cfg.MIN_CELL:
                continue
            p, dr = _effects(d, m, dec.CTRL, nb, rng, by_match)
            ln, hn = pct(dr[:, 1] - dr[:, 0])
            row[vname] = (p[1] - p[0], ln, hn, int(m.sum()))
            rep.say(f"   {sname:>16s} {vname:>9s} n = {int(m.sum()):6d} : {p[1] - p[0]:+5.2f} [{ln:+.2f} ; {hn:+.2f}]")
        if "tous" in row and "observes" in row:
            stable &= bool(np.sign(row["tous"][0]) == np.sign(row["observes"][0]))
            res[sname] = row
    v = "STABLE" if stable and res else "A VOIR"
    rep.say(f"VERDICT C7 (robustesse) : {v}   (signe du bilan inchange sur les moments observes pour l'ensemble, le porteur libre et le porteur presse)")
    return dict(verdict=v, bias=dict(gap=float(pt), ci=(float(lo), float(hi)), p_obs_by_line_tercile=p_obs), coverage=cov, strata=res)


# ---------------------------------------------------------------------------------------------------------------- orchestrateur
def run(ids, d, ds, cell_list, diffs, rep, nb=300, nperm=200, nperm_model=30, nperm_boot=60, nb_boot=100):
    """Toutes les verifications ; retourne un dict de verdicts pour le tableau du README."""
    rep.say("CONTROLES DE CREDIBILITE (regles dans l'en-tete de highline/checks.py)")
    out = {}
    for name, fn in (("C1", lambda: alignment(ids, rep)), ("C2", lambda: label_check(ids, rep)), ("C3", lambda: line_convention(ids, rep)),
                     ("C4", lambda: dict(placebo_bootstrap(d, rep, nperm=nperm_boot, nb=nb_boot), analytic=placebo_effect(d, rep, nperm=nperm))), ("C4m", lambda: placebo_model(ds, rep, nperm=nperm_model)),
                     ("C5", lambda: multiplicity(cell_list, diffs, rep)), ("C6", lambda: context_control(d, rep, nb=nb)), ("C7", lambda: line_observation(d, rep, nb=nb))):
        try:
            out[name] = fn()
        except Exception as ex:                                  # un controle ne doit jamais faire echouer le reste
            rep.say(f"   {name} non evalue ({type(ex).__name__}: {ex})")
            out[name] = dict(verdict="n/d")
        rep.say("")
    return out
