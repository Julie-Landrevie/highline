"""Controles de credibilite : sur des donnees synthetiques ou l'effet vrai est connu."""
import numpy as np
import pandas as pd

from highline import checks
from highline import decision as dec
from highline import frames as fr
from highline.report import Report


def synthetic_frames(effect_cost=0.0, effect_gain=0.0, n_match=20, per_team=400, seed=0):
    """Table de moments pour decision.prepare : la hauteur de ligne residuelle agit (ou non) sur les tirs avec un effet connu (points de probabilite par ecart-type)."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_match):
        for team in (1, 2):
            bx = rng.uniform(36, 90, per_team)
            ay = rng.uniform(0, 30, per_team)
            line = 0.45 * bx + rng.normal(0, 3, per_team)
            rows.append(pd.DataFrame(dict(match_id=m, team=team, t=np.arange(per_team) * 20, ball_x=bx, ball_ay=ay, line=line, age=rng.uniform(5, 30, per_team),
                                          free=rng.integers(0, 2, per_team).astype(float), score_diff=rng.integers(-1, 2, per_team).astype(float), minute=rng.uniform(1, 90, per_team),
                                          epv0=rng.uniform(0.0, 0.1, per_team), org0=1.0)))
    df = pd.concat(rows, ignore_index=True)
    res = fr.residual_line(df, True)
    z = (res / res.std()).fillna(0)
    df["shot15"] = (rng.random(len(df)) < 0.05 + effect_cost / 100 * z).astype(int)
    df["own15"] = (rng.random(len(df)) < 0.05 + effect_gain / 100 * z).astype(int)
    df["score_state"] = pd.cut(df.score_diff, [-np.inf, -0.5, 0.5, np.inf], labels=["derriere", "egalite", "devant"])
    df["min_bin"] = pd.cut(df.minute, [-1, 30, 60, 500], labels=["0-30", "30-60", "60+"])
    return df


def test_effet_connu_retrouve():
    d = dec.prepare(synthetic_frames(effect_cost=2.0, effect_gain=1.0))
    cl = dec.cells(d, nb=30, rng=np.random.default_rng(1))
    allc = next(c for c in cl if c["key"] == ("tous",) * 4)
    assert abs(allc["point"][0] - 2.0) < 0.6 and abs(allc["point"][1] - 1.0) < 0.6      # cout et gain par ecart-type, en points


def test_placebo_sans_effet_calibre():
    d = dec.prepare(synthetic_frames(0.0, 0.0, seed=3))
    rep = Report(quiet=True)
    out = checks.placebo_bootstrap(d, rep, nperm=20, nb=30, seed=5)
    assert max(out["fp"]) <= 0.3                                # 5 % attendu ; marge pour 20 tirages seulement


def test_placebo_detecte_un_vrai_effet():
    d = dec.prepare(synthetic_frames(effect_cost=3.0, effect_gain=0.0, seed=4))
    rep = Report(quiet=True)
    out = checks.placebo_effect(d, rep, nperm=40, seed=2)
    assert out["real"][0] > 1.5 and out["decalage circulaire"]["p"][0] < 0.1     # l'effet reel est dans la queue du placebo


def test_decalage_circulaire_conserve_les_valeurs():
    v = np.arange(100.0)
    blocks = [np.arange(0, 50), np.arange(50, 100)]
    out = checks.shift_within(v, blocks, np.random.default_rng(0))
    assert sorted(out[:50]) == list(range(50)) and sorted(out[50:]) == list(range(50, 100)) and not np.array_equal(out, v)
    out2 = checks.permute_within(v, blocks, np.random.default_rng(0))
    assert sorted(out2) == sorted(v)


def test_faux_positifs_attendus():
    d = dec.prepare(synthetic_frames(1.0, 0.5, seed=6))
    cl = dec.cells(d, nb=40, rng=np.random.default_rng(1))
    rep = Report(quiet=True)
    m = checks.multiplicity(cl, dec.diff_tests(cl), rep)
    assert m["tested"] >= 3 * len(cl) - 3 and abs(m["expected"] - 0.05 * (m["tested"] + m["nd"])) < 1e-9 and m["significant"] <= m["tested"]


def test_controle_contexte_stable_sans_confusion():
    d = dec.prepare(synthetic_frames(effect_cost=2.0, effect_gain=1.0, seed=7))
    rep = Report(quiet=True)
    out = checks.context_control(d, rep, nb=30)
    assert out["verdict"] in ("STABLE", "A VOIR") and "ensemble" in out["strata"]
    assert out["strata"]["ensemble"]["ok"]                      # l'EPV est tire au hasard : il ne doit rien changer au bilan


def test_modele_placebo_s_execute():
    d = dec.prepare(synthetic_frames(1.5, 0.0, seed=8), strict=True)
    out = checks.placebo_model(d, Report(quiet=True), nperm=3)
    assert set(out) == {"opp15", "own15"} and 0 < out["opp15"]["p"] <= 1
