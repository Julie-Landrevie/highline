"""T2 : la sensibilite retrouve des signes connus et le verdict suit sa regle ecrite."""
import numpy as np
import pandas as pd

from highline import sensitivity as S


def _synth(seed=0, n_matches=16, per=900):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        f = rng.normal(size=per)
        libre = rng.integers(0, 2, per)
        opp = (rng.random(per) < 0.03 + 0.012 * f * (libre == 1)).astype(float)       # monter coute quand le porteur est libre
        own = (rng.random(per) < 0.014 + 0.012 * f * (libre == 0)).astype(float)      # monter rapporte quand il est presse
        rows.append(pd.DataFrame(dict(match_id=m, f_res=f, free=libre, opp15=opp, own15=own,
                                      **{c: rng.normal(size=per) for c in ("f_bx", "f_bx2", "f_ay", "f_dg", "f_dg2", "f_age")})))
    return pd.concat(rows, ignore_index=True)


def _avec_pression_inconnue(seed=3):
    d = _synth(seed)
    rng = np.random.default_rng(seed)
    d.loc[rng.random(len(d)) < 0.1, "free"] = np.nan          # moments sans niveau de pression : ni libre ni presse
    d["ball_x"] = rng.uniform(36, 69, len(d))
    d["score_state"], d["min_bin"], d["half"] = "egalite", "0-30", (d.match_id % 2).values
    return d


def test_les_nombres_sont_ceux_du_tableau_de_decision():
    """La base de la sensibilite doit etre exactement la ligne « porteur : libre / presse / ensemble » du tableau de decision."""
    from highline import decision as dec
    d = _avec_pression_inconnue()
    cells = {c["key"]: c for c in dec.cells(d, nb=2, min_n=400)}
    r = S.evaluate(d, nb=5)
    for name, key in (("ensemble", ("tous",) * 4), ("libre", ("libre", "tous", "tous", "tous")), ("presse", ("presse", "tous", "tous", "tous"))):
        assert abs(r[name][0] - dec.summarize(cells[key])["net"]) < 1e-9, name


def test_retrouve_les_signes_connus():
    r = S.evaluate(_synth(), nb=100)
    assert r["libre"][0] < 0 < r["presse"][0] and r["libre - presse"][0] < 0
    assert r["libre - presse"][2] < 0                      # l'IC exclut 0


def test_le_retrait_d_un_match_ne_change_pas_le_signe_ici():
    lo, hi, flips = S.leave_one_match_out(_synth())
    assert hi < 0 and flips == 0


def test_verdict_suit_la_regle():
    def res(e, l, p, dl, lo, hi):
        return dict(ensemble=(e, 0, 0), libre=(l, 0, 0), presse=(p, 0, 0), **{"libre - presse": (dl, lo, hi)})
    base = res(1, -1, 1.4, -2.3, -3, -1)
    ok = {"a": res(0.8, -0.6, 1.1, -1.7, -2.9, -0.4)}
    wide = {"a": res(0.8, -0.6, 1.1, -1.7, -3.5, 0.2)}
    flip = {"a": res(0.8, 0.2, 1.1, -0.9, -3.5, 0.2)}
    assert S.verdict(base, ok)[0] == "ROBUSTE"
    assert S.verdict(base, wide)[0] == "A VOIR"
    assert S.verdict(base, flip)[0] == "FRAGILE" and S.verdict(base, ok, lomo_flips=2)[0] == "FRAGILE"
