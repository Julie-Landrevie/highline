"""T1 : le test hors echantillon detecte une heterogeneite connue et ne la voit pas quand elle n'existe pas."""
import numpy as np
import pandas as pd

from highline import validation as V


def _synth(hetero, n_matches=16, per=700, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        k = per
        f = rng.normal(size=k)
        libre = rng.integers(0, 2, k).astype(float)
        adv = rng.integers(0, 2, k).astype(float)
        base = {c: rng.normal(size=k) for c in V.CTRL}
        slope = np.where(libre == 1, -0.05 * hetero, 0.05 * hetero)
        net = rng.normal(scale=0.2, size=k) + slope * f
        rows.append(pd.DataFrame(dict(match_id=m, f_res=f, libre=libre, adv=adv, net15=net, opp15=(rng.random(k) < 0.03).astype(float),
                                      own15=(rng.random(k) < 0.05).astype(float), **base)))
    d = pd.concat(rows, ignore_index=True)
    d["x_libre"], d["x_adv"] = d.f_res * d.libre, d.f_res * d.adv
    return d


def test_detecte_une_heterogeneite_connue():
    r = V.situation_vs_rule(_synth(1.0), nb=200, nperm=10)
    assert r["verdict"] == "GO" and r["gain"] > 0 and r["ci"][0] > 0


def test_ne_voit_rien_quand_il_n_y_a_rien():
    r = V.situation_vs_rule(_synth(0.0), nb=200, nperm=10)
    assert r["verdict"] in ("NO-GO", "A VOIR") and r["ci"][0] <= 0


def test_les_controles_sont_ceux_du_tableau_de_decision():
    from highline import decision
    assert V.CTRL == decision.CTRL
