"""Rendu 3D : une scene synthetique doit produire une image, avec les deux aspects et sans donnees SkillCorner."""
import numpy as np
import pytest

from highline import render3d as R


def _scene(n=12):
    rng = np.random.default_rng(1)

    def traj(x0, y0):
        return [float(x0 + 0.3 * i + rng.normal(0, 0.1)) for i in range(n)], [float(y0 + rng.normal(0, 0.1)) for i in range(n)]
    players = []
    for k, (x, y) in enumerate(zip([30, 31, 29, 33, 40, 42, 38, 44, 48, 36], [-24, -8, 8, 24, -15, 0, 15, -5, 10, -30])):
        xs, ys = traj(x, y)
        det = [1 if k >= 4 else 0] * n
        players.append(dict(side="d", gk=False, x=xs, y=ys, det=det, r=[0.0 if d else 3.0 for d in det]))
    xs, ys = traj(2, 0)
    players.append(dict(side="d", gk=True, x=xs, y=ys, det=[1] * n, r=[0.0] * n))
    for x, y in zip([45, 50, 55, 52, 60, 48, 58, 65, 40, 70], [-20, -10, 0, 12, 22, 5, -18, 15, -30, -4]):
        xs, ys = traj(x, y)
        players.append(dict(side="a", gk=False, x=xs, y=ys, det=[1] * n, r=[0.0] * n))
    return dict(meta=dict(pitch=[105, 68], n=n, fps=5), players=players, ball=dict(x=[50.0 + 0.3 * i for i in range(n)], y=[-8.0] * n),
                line=[31.0] * n, sd=[1.4] * n)


@pytest.mark.parametrize("look", ["charte", "realiste"])
def test_rendu_image_fixe(tmp_path, look):
    out = R.render(_scene(), str(tmp_path / "m"), lang="fr", style="plain", gif=False, mp4=False, look=look, size=400)
    assert out == [str(tmp_path / "m") + ".png"]
    assert (tmp_path / "m.png").stat().st_size > 5000


def test_un_joueur_non_detecte_reste_distinct():
    """Un joueur non detecte est dessine translucide : sa transparence ne doit jamais etre celle d'un joueur observe."""
    polys, cols = R._body(10.0, 0.0, (0.1, 0.5, 0.5), 0.42)
    assert all(c[3] == pytest.approx(0.42) for c in cols)
    polys, cols = R._body(10.0, 0.0, (0.1, 0.5, 0.5), 1.0)
    assert all(c[3] == pytest.approx(1.0) for c in cols)


def test_dix_joueurs_de_champ_par_equipe():
    """La scene de test, comme un vrai moment, compte 10 joueurs de champ par equipe (le gardien en plus)."""
    sc = _scene()
    for side in ("a", "d"):
        assert sum(1 for p in sc["players"] if p["side"] == side and not p["gk"]) == 10


def test_ballon_a_pentagones_qui_roule():
    """Le ballon montre des pentagones sombres sur fond clair, et le motif change quand il roule."""
    polys, c0 = R._ball(10.0, 0.0, R.BALL_R, np.eye(3), "#ffffff", "#1c1917")
    lum = np.array([sum(c[:3]) for c in c0])
    assert lum.max() > 2.0 and lum.min() < 0.6
    _, c1 = R._ball(10.0, 0.0, R.BALL_R, R._rot((0, 1, 0), 0.8), "#ffffff", "#1c1917")
    assert not np.allclose(np.array(c0), np.array(c1))


def test_format_video_16_9(tmp_path):
    from PIL import Image
    R.render(_scene(), str(tmp_path / "v"), lang="fr", style="plain", gif=False, mp4=False, look="realiste", fmt="video", size=300)
    w, h = Image.open(tmp_path / "v.png").size
    assert abs(w / h - 16 / 9) < 0.02


def test_aucun_joueur_de_champ_n_est_hors_cadre():
    """Regression : un attaquant tres loin de la camera (x d'origine > 87 m) etait coupe par un cadrage minimal fixe."""
    sc = _scene()
    far = [p for p in sc["players"] if p["side"] == "a" and not p["gk"]][0]
    far["x"] = [90.0 + 0.1 * i for i in range(len(far["x"]))]
    S = R.prepare(sc, 3)
    xmin = min(float(np.nanmin(q["x"])) for q in S["players"] if not q["gk"])
    assert S["xlo"] < xmin - 5
