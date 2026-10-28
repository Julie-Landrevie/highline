"""Scenes construites a la main : on connait la reponse avant de lancer le code."""
import numpy as np
import pandas as pd
import pytest

from highline import data as dat
from highline import frames as fr

from synth import DEF_DIST, make_events, make_match, patch


def test_ligne_exacte_et_observation(monkeypatch):
    patch(monkeypatch, make_match(), make_events())
    t = fr.match_table(1)
    assert set(t.team) == {1}                                  # l'equipe 2 a la balle : seule l'equipe 1 defend
    assert np.allclose(t.line, np.mean(sorted(DEF_DIST)[:4]))   # (10 + 12 + 14 + 16) / 4 = 13 m de la ligne de but
    assert np.allclose(t.ball_x, 60.0)
    assert (t.n_det == 10).all() and (t.n_det_back4 == 4).all()
    assert t.line_sd.max() < 0.2                                # tous observes : incertitude de l'ordre de sigma(0) / 2


def test_joueur_extrapole_augmente_l_incertitude(monkeypatch):
    patch(monkeypatch, make_match(gap=(1, 90, 260)), make_events())      # le plus bas des defenseurs (10 m) n'est pas observe pendant 17 s
    t = fr.match_table(1)
    mid = t[(t.t >= 120) & (t.t <= 200)]
    assert (mid.n_det == 9).all() and (mid.n_det_back4 == 3).all()
    assert mid.line_sd.min() > 1.0                              # sigma(17 s) ~ 5 m sur 1 joueur des 4 : environ 1,3 m sur la moyenne
    assert np.allclose(mid.line, 13.0)                          # la position fournie est conservee


def test_etiquette_tir_dans_les_15_secondes(monkeypatch):
    patch(monkeypatch, make_match(), make_events(shot_end=250))
    t = fr.match_table(1)
    # tir adverse (equipe 2 tire, equipe 1 concede) a l'image 250 : vrai pour 100 <= t <= 249
    assert (t[t.t <= 240].shot15 == 1).all() and (t[t.t >= 260].shot15 == 0).all()


def test_build_complet_contexte(monkeypatch, tmp_path):
    patch(monkeypatch, make_match(), make_events())
    df = fr.build([1], cache=tmp_path / "f.csv", verbose=False)
    assert df.own15.sum() == 0                                  # l'equipe 1 ne tire jamais
    assert (df.free == 1).all()                                 # low_pressure : porteur libre
    assert np.allclose(df.epv0, 0.03) and (df.org0 == 1).all()
    assert (df.score_diff == 0).all()
    again = fr.build([1], cache=tmp_path / "f.csv", verbose=False)    # relu depuis le cache
    assert len(again) == len(df)


def test_cache_ancien_reconstruit(monkeypatch, tmp_path):
    patch(monkeypatch, make_match(), make_events())
    df = fr.build([1], cache=tmp_path / "f.csv", verbose=False)
    df.drop(columns=["epv0", "org0"]).to_csv(tmp_path / "old.csv", index=False)       # cache d'une version anterieure
    new = fr.build([1], cache=tmp_path / "old.csv", verbose=False)
    assert "epv0" in new.columns


def test_reference_de_but(monkeypatch):
    T = make_match()
    T["X"][:, 0] = -34.0                                        # gardien median a -34 m : decale de 18,5 m de la ligne de but
    monkeypatch.setattr(dat, "load_match", lambda m: T)
    g_line, g_gk = dat.own_goal_sign(T, "line"), dat.own_goal_sign(T, "gk_median")
    assert g_line[(1, 1)] == -52.5 and abs(g_gk[(1, 1)] + 34.0) < 1e-9
    assert g_line[(1, 2)] == 52.5                               # equipe 2 : gardien a +50, but defendu en +52,5
    with pytest.raises(ValueError):
        dat.set_goal_reference("autre")
