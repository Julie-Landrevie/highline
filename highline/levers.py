"""Leviers apres une perte de balle : contre-presser (joueurs de l'equipe qui perd a moins de 8 m de la balle) ou se replier (joueurs entre la balle et leur but).

Unite : perte de balle d'une equipe A = recuperation de l'adversaire B dans sa propre moitie, A ayant 10 joueurs de champ.
Issues : box15 (B entre dans la surface en 15 s en gardant le ballon), shot15 (tir de B en 15 s), lost8 (A reprend le ballon en 8 s), direct (B progresse d'au moins 10 m en 2,5 s)."""
import numpy as np
import pandas as pd

from . import config as cfg
from . import data as dat
from . import frames as fr
from .stats import half_label, logit, pct, zs

DIRECT_PHASES = {"quick_break", "direct", "transition"}


def recoveries(mid):
    T = dat.load_match(mid)
    ev = dat.load_events(mid)
    X, Y, BX, BY, PER, team, grp = T["X"], T["Y"], T["BX"], T["BY"], T["PER"], T["team"], T["grp"]
    n = T["n"]
    gx = dat.own_goal_sign(T)
    poss = dat.possession_array(ev, T["f0"], n)
    sf, st = dat.shot_frames(ev, T["f0"])
    pp = ev[(ev.event_type == "player_possession")].dropna(subset=["frame_start", "team_id"]).copy()
    pp["fs"] = pp.frame_start.astype(int) - T["f0"]
    chg = np.where((poss[1:] != poss[:-1]) & ~np.isnan(poss[1:]) & ~np.isnan(poss[:-1]))[0] + 1
    rows = []
    for f in chg:
        if f + 160 >= n or PER[f] == 0 or PER[f + 160] != PER[f] or not np.isfinite(BX[f:f + 160]).all():
            continue
        b, a = int(poss[f]), int(poss[f - 1])
        gb, ga = gx.get((PER[f], b)), gx.get((PER[f], a))
        if gb is None or ga is None or not np.isfinite([gb, ga]).all():
            continue
        sb, sa = (1.0 if gb < 0 else -1.0), (1.0 if ga < 0 else -1.0)
        pr = sb * BX[f:f + 161]
        if pr[0] >= 0:
            continue
        ac = np.where((team == a) & (grp != "GK") & np.isfinite(X[f]))[0]
        bc = np.where((team == b) & (grp != "GK") & np.isfinite(X[f]))[0]
        if len(ac) != 10 or len(bc) < 8:
            continue
        xa = np.sort(sa * (X[f, ac] - ga))
        dist_ball = np.hypot(X[f, ac] - BX[f], Y[f, ac] - BY[f])
        inbox = (pr >= 36) & (np.abs(BY[f:f + 161]) <= 20.16) & (poss[f:f + 161] == b)
        box = 0
        if inbox.any():
            k = int(np.argmax(inbox))
            box = int((poss[f:f + k + 1] == b).all())
        cand = pp[(pp.team_id == b) & (pp.fs >= f) & (pp.fs <= f + 30)]
        phase = cand.iloc[0].team_in_possession_phase_type if len(cand) and "team_in_possession_phase_type" in cand else None
        rows.append(dict(match_id=mid, frame=int(f), team=b, opp=a, sgn=sb, start=float(pr[0]), direct=int((pr[25] - pr[0]) >= 10),
                         n_opp_ahead=int((sb * X[f, ac] > pr[0]).sum()), n_press=int((dist_ball < 8).sum()), ay=float(abs(BY[f])),
                         line_A=float(xa[:4].mean()), d_dm_A=float(xa[4:7].mean() - xa[:4].mean()),
                         box15=box, shot15=int(((sf > f) & (sf <= f + 150) & (st == b)).any()), lost8=int((poss[f + 1:f + 81] == a).any()),
                         phase=phase, n_det_A=int(np.nansum(T["D"][f, ac] == 1))))
    return rows


def build_rows(matches=None, cache="cache/losses.csv", refresh=False, verbose=True):
    import os
    ids = list(matches or cfg.ALL_MATCHES)
    if os.path.exists(cache) and not refresh:
        df = pd.read_csv(cache)
        if set(ids) <= set(df.match_id.unique()):
            return df[df.match_id.isin(ids)].reset_index(drop=True)
    rows = []
    for mid in ids:
        r = recoveries(mid)
        if verbose:
            print(f"[pertes] match {mid} : {len(r)} pertes", flush=True)
        rows += r
    df = pd.DataFrame(rows)
    df = fr.add_context(df, "opp", "frame")
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def provider_overlap(rows):
    """Controle de non-recoupement : accord entre mon critere 'direct' et les phases quick_break / direct / transition du fournisseur."""
    ok = rows.phase.notna()
    if not ok.any():
        return None
    qb = rows.phase.isin(DIRECT_PHASES)
    return dict(n=int(ok.sum()), mine_in_provider=float(qb[rows.direct == 1].mean()), provider_in_mine=float(rows[qb].direct.mean()), n_provider=int(qb.sum()))


def _fit(s, o, feats):
    names, b, se = logit(s[o].values, s[feats], s.match_id.values)
    return dict(zip(names, b)), dict(zip(names, se))


def analyse(rows, rep, nb=300, rng=None, min_events=cfg.MIN_EVENTS):
    """Contre-pressing contre repli : OR par ecart-type, difference D, echange d'un joueur, strates (zone de recuperation, score, moment, ligne de A)."""
    rng = rng or np.random.default_rng(204)
    rep.say("LEVIERS : CONTRE-PRESSER OU SE REPLIER ? (pertes de balle d'une equipe A ; OR par ecart-type)")
    e = rows.dropna(subset=["start", "n_opp_ahead", "n_press", "line_A", "d_dm_A", "ay"]).copy().reset_index(drop=True)
    sd = {c: e[c].std() for c in ["n_press", "n_opp_ahead"]}
    for c in ["start", "n_opp_ahead", "n_press", "ay"]:
        e["f_" + c] = zs(e[c])
    e["half"] = half_label(e.match_id.values)
    e["z_rec"] = pd.qcut(e.start, 3, labels=["profonde", "mediane", "haute"])
    e["l_A"] = np.where(e.line_A >= e.line_A.median(), "ligne haute", "ligne basse")
    feats = ["f_n_press", "f_n_opp_ahead", "f_start", "f_ay"]
    rep.say(f"{len(e)} pertes ; 1 ecart-type = {sd['n_press']:.2f} joueur(s) a < 8 m (contre-pressing), {sd['n_opp_ahead']:.2f} joueur(s) entre la balle et le but (repli)")
    rho = e[["n_press", "n_opp_ahead"]].corr(method="spearman").iloc[0, 1]
    rep.say(f"Correlation de rang contre-pressing / repli = {rho:+.2f}")
    ov = provider_overlap(e)
    if ov:
        rep.say(f"Non-recoupement avec les phases fournisseur : {ov['mine_in_provider']:.0%} de mes pertes 'directes' sont quick_break/direct/transition chez le fournisseur ; "
                f"{ov['provider_in_mine']:.0%} des phases fournisseur correspondantes sont 'directes' selon mon critere (n = {ov['n_provider']}).")
    strata = {"ensemble": None}
    for dim, col, labs in (("zone de recuperation", "z_rec", ["profonde", "mediane", "haute"]), ("score de A", "score_state", ["derriere", "egalite", "devant"]),
                           ("moment", "min_bin", ["0-30", "30-60", "60+"]), ("hauteur de ligne de A", "l_A", ["ligne basse", "ligne haute"])):
        for lab in labs:
            strata[(dim, lab)] = (col, lab)

    def sub_of(s, key):
        return s if strata[key] is None else s[s[strata[key][0]].astype(str) == strata[key][1]]

    def delta(s):
        b, _ = _fit(s, "box15", feats)
        return b["f_n_press"] - b["f_n_opp_ahead"]

    point, boots, halves = {}, {k: [] for k in strata}, {}
    for k in strata:
        s = sub_of(e, k)
        if s.box15.sum() < min_events or s.box15.nunique() < 2:
            point[k] = None
            continue
        bb, se = _fit(s, "box15", feats)
        br, _ = _fit(s, "lost8", feats)
        point[k] = dict(n=len(s), ev=int(s.box15.sum()), bp=bb["f_n_press"], br=bb["f_n_opp_ahead"], sep=se["f_n_press"], ser=se["f_n_opp_ahead"],
                        rp=br["f_n_press"], rr=br["f_n_opp_ahead"], D=bb["f_n_press"] - bb["f_n_opp_ahead"])
        hh = []
        for h in (0, 1):
            sh = s[s.half == h]
            hh.append(delta(sh) if sh.box15.sum() >= 10 and sh.box15.nunique() == 2 else np.nan)
        halves[k] = hh
    groups = {m: g for m, g in e.groupby("match_id")}
    mids = list(groups)
    swap_b = []

    def swap(s):
        b, _ = _fit(s, "box15", feats)
        return float(np.exp(b["f_n_press"] / sd["n_press"] - b["f_n_opp_ahead"] / sd["n_opp_ahead"]))
    for _ in range(nb):
        eb = pd.concat([groups[m] for m in rng.choice(mids, len(mids))], ignore_index=True)
        for k in strata:
            if point[k] is None:
                continue
            try:
                boots[k].append(delta(sub_of(eb, k)))
            except Exception:
                pass
        try:
            swap_b.append(swap(eb))
        except Exception:
            pass
    rep.say("D = ln OR(contre-pressing) - ln OR(repli) sur l'entree dans la surface ; D < 0 : le contre-pressing la reduit plus ; D > 0 : le repli la reduit plus.")
    rep.say(f"{'situation':>34s} {'n':>5s} {'evt':>4s} | {'OR press.':>18s} {'OR repli':>18s} | {'D [IC95 %]':>22s} {'moit.0':>7s} {'moit.1':>7s} | {'lecture':>26s} | regain 8 s : OR press. / repli")
    pref = {}
    for k in strata:
        lab = "ensemble" if k == "ensemble" else f"{k[0]} : {k[1]}"
        p = point[k]
        if p is None:
            rep.say(f"{lab:>34s}   (trop peu d'evenements)")
            continue
        lo, hi = pct(boots[k])
        h0, h1 = halves[k]
        stable = np.isfinite(h0) and np.isfinite(h1) and np.sign(h0) == np.sign(h1) == np.sign(p["D"])
        if lo > 0 and stable:
            lec, pref[k] = "REPLI mieux (tranche)", "repli"
        elif hi < 0 and stable:
            lec, pref[k] = "CONTRE-PRESSING mieux (tranche)", "pressing"
        else:
            lec, pref[k] = "indistinct", None
        rep.say(f"{lab:>34s} {p['n']:5d} {p['ev']:4d} | {np.exp(p['bp']):.2f} [{np.exp(p['bp'] - 1.96 * p['sep']):.2f};{np.exp(p['bp'] + 1.96 * p['sep']):.2f}] "
                f"{np.exp(p['br']):.2f} [{np.exp(p['br'] - 1.96 * p['ser']):.2f};{np.exp(p['br'] + 1.96 * p['ser']):.2f}] | {p['D']:+.2f} [{lo:+.2f};{hi:+.2f}] {h0:+7.2f} {h1:+7.2f} | {lec:>26s} | "
                f"{np.exp(p['rp']):.2f} / {np.exp(p['rr']):.2f}")
    sw0, (sw_lo, sw_hi) = swap(e), pct(swap_b)
    rep.say(f"Echange d'UN joueur (de 'derriere la balle' a 'pres de la balle') : OR de l'entree dans la surface = {sw0:.2f} [{sw_lo:.2f};{sw_hi:.2f}]  (< 1 : le contre-pressing est le meilleur usage du joueur)")
    dims = {}
    for k, v in pref.items():
        if k != "ensemble" and v:
            dims.setdefault(k[0], set()).add(v)
    reversal = [d for d, s in dims.items() if len(s) == 2]
    verdict = "GO" if reversal else ("A VOIR" if pref.get("ensemble") else "NO-GO")
    rep.say(f"VERDICT LEVIERS : {verdict}   (inversion entre strates dans : {', '.join(reversal) if reversal else 'aucune'} ; levier prefere dans l'ensemble : {pref.get('ensemble') or 'aucun'})")
    rep.say("Lecture : D est calcule par ecart-type et l'ecart-type du repli est plus grand que celui du contre-pressing ; la comparaison par joueur (echange ci-dessus) est la plus parlante. "
            "Association : on presse quand on le peut.")
    return dict(point=point, pref=pref, boots=boots, strata=strata, swap=(sw0, sw_lo, sw_hi), verdict=verdict, rho=rho, overlap=ov)
