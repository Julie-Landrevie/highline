"""API publique, extraction d'un moment, rendu 3D, vue HTML d'un moment, pose (a partir de scenes synthetiques)."""
import json

import numpy as np
import pandas as pd
import pytest

from highline import api, moment, pose, render3d, tool
from highline import data as dat
from highline import decision as dec

from synth import make_match, make_events, patch


def fake_table(cost=1.0, gain=0.5, sd=8.0, n=900):
    rng = np.random.default_rng(0)
    return dict(sd_line=sd, min_cell=400, cells=[dict(key=["libre", "milieu", "tous", "tous"], n=n, cost=cost, gain=gain,
                                                      draws=[[cost + rng.normal() * 0.2, gain + rng.normal() * 0.1] for _ in range(100)])])


def test_observe_line_incertitude_et_fiabilite():
    d = [18, 20, 22, 25, 30, 34, 38, 41, 45, 50]
    o = api.observe_line(d)
    assert o["line_m"] == pytest.approx(21.25) and o["reliability"] == "high" and o["deepest4_observed"] == 4
    o2 = api.observe_line(d, gap_s=[10, 10, 10, 10, 0, 0, 0, 0, 0, 0])
    assert o2["sd_m"] > 1.5 and o2["reliability"] == "low" and o2["deepest4_observed"] == 0
    with pytest.raises(ValueError):
        api.observe_line(d[:5])


def test_decide_echelle_et_signe():
    t = fake_table(cost=2.0, gain=0.5)                          # bilan par ecart-type = -1,5 (monter coute)
    r = api.decide(t, "libre", "milieu", residual_m=8.0)        # un ecart-type exactement
    assert r["net"] == pytest.approx(-1.5) and r["tag"] == "higher_costs" and r["w_star"] == pytest.approx(0.25)
    r2 = api.decide(t, "libre", "milieu", residual_m=-8.0)      # ligne plus basse que d'habitude : le signe s'inverse
    assert r2["net"] == pytest.approx(1.5) and r2["tag"] == "lower_pays"
    with pytest.raises(ValueError):
        api.decide(t, "presse", "milieu")


def test_line_decision_message_et_avertissements():
    t = fake_table(cost=1.0, gain=1.2)
    d = [18, 20, 22, 25, 30, 34, 38, 41, 45, 50]
    r = api.line_decision(t, d, habitual_line_m=19.0, ball_x=55.0, carrier="libre", gap_s=[0, 0, 12, 12, 0, 0, 0, 0, 0, 0])
    assert "monter" in r["message"] or "pas tranchable" in r["message"]
    assert any("Incertitude" in w or "signe" in w for w in r["warnings"]) or r["observation"]["reliability"] != "low"
    en = api.line_decision(t, d, 19.0, 55.0, "libre", lang="en")
    assert "Line at" in en["message"]
    with pytest.raises(ValueError):
        api.line_decision(t, d, 19.0, ball_x=20.0)               # hors perimetre


def test_fit_habit():
    rng = np.random.default_rng(1)
    bx = rng.uniform(35, 90, 500)
    line = 0.5 * bx + 2 + rng.normal(0, 0.3, 500)
    f = api.fit_habit(line, bx)
    assert f(60.0) == pytest.approx(0.5 * 60 + 2, abs=0.3)


def test_export_et_chargement_table(tmp_path):
    cells = [dict(key=("libre", "milieu", "tous", "tous"), n=500, point=np.array([1.0, 0.4]), draws=np.random.default_rng(0).normal(size=(120, 2)), mask=None, halves=[None, None])]
    p = dec.export_cells(cells, 7.6, tmp_path / "t.json")
    t = api.load_table(p)
    assert t["sd_line"] == 7.6 and len(t["cells"][0]["draws"]) == 100 and t["goal_reference"] == "line"


def test_extraction_d_un_moment(monkeypatch):
    patch(monkeypatch, make_match(gap=(1, 90, 260)), make_events())
    m = moment.extract(1, 1, 150, back=30, fwd=30, step=2)
    assert m["meta"]["n"] == 31 and len(m["players"]) == 22
    assert all(abs(v - 13.0) < 1e-6 for v in m["line"] if v is not None)
    deepest = next(p for p in m["players"] if p["side"] == "d" and not p["gk"] and abs(p["x"][0] - 10.0) < 1e-6)
    assert sum(deepest["det"]) == 0 and min(deepest["r"]) > 5                      # non observe : disque de plus de 5 m
    assert min(sd for sd in m["sd"]) > 1.0
    assert json.dumps(m)                                                           # serialisable


def test_rendu_3d_image_fixe(monkeypatch, tmp_path):
    patch(monkeypatch, make_match(gap=(1, 90, 260)), make_events())
    m = moment.extract(1, 1, 150)
    out = render3d.render(m, str(tmp_path / "m"), lang="en", style="plain", gif=False, mp4=False)
    assert out and (tmp_path / "m.png").stat().st_size > 5000


def test_outil_html_avec_moment(monkeypatch, tmp_path):
    patch(monkeypatch, make_match(), make_events())
    m = moment.extract(1, 1, 150)
    cells = [dict(key=("tous",) * 4, n=500, point=np.array([1.0, 0.4]), draws=np.random.default_rng(0).normal(size=(50, 2)))]
    p = tool.build(cells, [], None, None, dict(all10=0.14, all4=0.36, sd_med=0.9, sd_p90=2.4, n=100), None, tmp_path / "o.html", sd_line=7.6, moment=m)
    html = open(p, encoding="utf-8").read()
    assert 'id="mvbox"' in html and '"moment":{' in html and "drawMoment" in html


def test_pose_seuil_de_qualite(monkeypatch):
    """Scene de pose synthetique : trois defenseurs les plus proches, un seul sous le seuil de 20 cm, tous sous le seuil de 30 cm."""
    mid = 1925299
    T = make_match()
    T["BX"][:] = -52.5 + 25.0                                   # balle a 25 m du but, au niveau de la ligne de 25 m
    T["X"][:, 5] = -52.5 + 25.0                                 # defenseur colonne 5 : sur la balle
    T["X"][:, 4] = -52.5 + 25.0
    T["Y"][:, 4], T["Y"][:, 5], T["Y"][:, 6] = 1.0, 2.0, 3.0
    T["X"][:, 6] = -52.5 + 25.0
    monkeypatch.setattr(dat, "load_match", lambda m: T)
    dat.set_goal_reference("line")
    rows = []
    for col, err in ((4, 5.0), (5, 25.0), (6, 25.0)):
        for f in range(100, 400, 2):
            vx, vy = 0.0, -0.4                                  # lacet pi : epaules orientees pour faire face a -x (vers leur but)
            rows.append(dict(player_id=T["cols"][col], sh_x=vx, sh_y=vy, e_sh=err, T=f, pose_frame=f * 5 // 2))
    V = pd.DataFrame(rows)
    monkeypatch.setattr(pose, "read_pose", lambda m: V)
    d = pd.DataFrame([dict(match_id=mid, frame=200, sgn=-1.0, team=2, opp=1)])
    strict = pose.pose_features(d, max_err_cm=20.0)
    loose = pose.pose_features(d, max_err_cm=30.0)
    assert strict.a_n.iloc[0] == 1 and loose.a_n.iloc[0] == 3 and loose.a_tot.iloc[0] == 3
    assert loose.a_turned.iloc[0] == 1.0                        # les 3 defenseurs font face a leur propre but
