"""Tests unitaires sur donnees synthetiques (aucune donnee SkillCorner requise)."""
import numpy as np
import pandas as pd

from highline import config as cfg
from highline import frames, reliability as rel, stats


def test_zs_centre_reduit():
    z = stats.zs(np.arange(10))
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9


def test_logit_retrouve_le_coefficient():
    rng = np.random.default_rng(1)
    n = 6000
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.3 + 0.8 * x)))).astype(int)
    names, b, se = stats.logit(y, pd.DataFrame({"x": x}), rng.integers(0, 20, n))
    assert abs(b[1] - 0.8) < 0.15 and se[1] > 0


def test_ols_cl_retrouve_la_pente():
    rng = np.random.default_rng(2)
    x = rng.normal(size=3000)
    y = 2 * x + rng.normal(size=3000)
    names, b, se = stats.ols_cl(y, pd.DataFrame({"x": x}), rng.integers(0, 20, 3000))
    assert abs(b[1] - 2) < 0.1


def test_auc_bornes():
    y = np.array([0, 0, 1, 1])
    assert stats.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert stats.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert np.isnan(stats.auc(np.zeros(4), np.arange(4)))


def test_gap_lengths():
    D = np.array([[1], [0], [0], [0], [1], [0], [1]])
    assert rel.gap_lengths(D)[:, 0].tolist() == [0, 3, 3, 3, 0, 1, 0]


def test_sigma_croit_avec_le_trou():
    s = rel.sigma_of(np.array([0, 10, 20, 100, 1000]))
    assert np.all(np.diff(s) >= 0)
    assert abs(rel.radius90(np.array([0]))[0] - cfg.R90 * cfg.SIG_V[0]) < 1e-9


def test_ligne_et_incertitude():
    d = np.array([10, 12, 14, 16, 30, 40, 50, 60, 70, 80.0])
    assert rel.line_of(d) == 13.0
    assert rel.line_uncertainty(d, np.zeros(10)) < rel.line_uncertainty(d, np.full(10, 300))


def test_lookup_signe_et_ordre():
    # equipe 1 mene 1-0 apres l'image 100 ; equipe 2 voit donc -1
    ctx = (np.array([0, 100]), np.array([1, 1]), np.array([0.0, 1.0]), np.array([0.0, 20.0]))
    s1, m1 = frames.lookup(ctx, np.array([1, 1]), np.array([50, 150]))
    s2, _ = frames.lookup(ctx, np.array([2]), np.array([150]))
    assert s1.tolist() == [0.0, 1.0] and m1.tolist() == [0.0, 20.0] and s2.tolist() == [-1.0]
    s0, _ = frames.lookup(ctx, np.array([1]), np.array([-5]))
    assert np.isnan(s0[0])


def test_ligne_residuelle_centree():
    rng = np.random.default_rng(3)
    n = 400
    bx = rng.uniform(35, 100, n)
    df = pd.DataFrame({"match_id": 1, "team": 1, "ball_x": bx, "ball_ay": rng.uniform(0, 30, n)})
    df["line"] = 0.5 * bx + rng.normal(size=n)
    r = frames.residual_line(df)
    assert abs(r.mean()) < 1e-6 and r.std() < 1.2
