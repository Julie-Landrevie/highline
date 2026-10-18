import numpy as np
import pandas as pd

from highline import decision as dec, tool


def synth(n=16000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"match_id": rng.integers(0, 20, n), "team": rng.integers(1, 3, n)})
    d["ball_x"] = rng.uniform(36, 100, n)
    d["ball_ay"] = rng.uniform(0, 30, n)
    d["age"] = rng.integers(5, 40, n)
    d["free"] = rng.integers(0, 2, n).astype(float)
    d["score_diff"] = rng.integers(-2, 3, n).astype(float)
    d["minute"] = rng.uniform(0, 95, n)
    d["shot15"] = (rng.random(n) < 0.08).astype(int)
    d["own15"] = (rng.random(n) < 0.08).astype(int)
    d["line"] = 0.4 * d.ball_x + rng.normal(size=n) * 3
    d["score_state"] = pd.cut(d.score_diff, [-np.inf, -0.5, 0.5, np.inf], labels=["derriere", "egalite", "devant"])
    d["min_bin"] = pd.cut(d.minute, [-1, 30, 60, 500], labels=["0-30", "30-60", "60+"])
    return d


def test_prepare_et_cellules():
    d = dec.prepare(synth(), strict=True)
    assert {"net15", "f_res", "x_libre"} <= set(d.columns)
    assert abs(d.f_res.mean()) < 0.2
    cl = dec.cells(d, nb=30, rng=np.random.default_rng(0))
    assert len(cl) > 0
    s = dec.summarize(cl[0])
    assert s is not None


def test_outil_html_se_construit(tmp_path):
    d = dec.prepare(synth(), strict=True)
    cl = dec.cells(d, nb=30, rng=np.random.default_rng(0))
    diffs = dec.diff_tests(cl)
    out = tmp_path / "t.html"
    tool.build(cl, diffs, {}, {}, {}, {}, out, sd_line=float(d.res.std()))
    html = out.read_text(encoding="utf-8")
    assert "<html" in html and "DATA" in html
