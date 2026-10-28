"""Match synthetique construit a la main : positions connues, donc resultat attendu connu a l'avance (aucune donnee SkillCorner)."""
import numpy as np
import pandas as pd

PITCH_HALF = 52.5
DEF_DIST = [10.0, 12.0, 14.0, 16.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]   # distance a la ligne de but des 10 joueurs de champ de l'equipe 1 (qui defend)


def make_match(n=700, ball_dist=60.0, gap=None, f0=0):
    """Equipe 1 defend le but en x = -52,5 (gardien a -50) ; equipe 2 attaque vers -x et garde le ballon pendant tout le match.

    gap = (colonne, debut, fin) : ce joueur n'est pas observe (extrapole) entre ces images."""
    ids = list(range(1, 23))                       # 1..11 equipe 1 (1 = gardien), 12..22 equipe 2 (12 = gardien)
    team = np.array([1] * 11 + [2] * 11)
    grp = np.array(["GK"] + ["D"] * 10 + ["GK"] + ["M"] * 10)
    X = np.zeros((n, 22))
    Y = np.zeros((n, 22))
    X[:, 0], Y[:, 0] = -50.0, 0.0
    for k, dd in enumerate(DEF_DIST):
        X[:, 1 + k] = -PITCH_HALF + dd
        Y[:, 1 + k] = -27 + 6 * k
    bx = -PITCH_HALF + ball_dist
    X[:, 11], Y[:, 11] = 50.0, 0.0
    for k in range(10):
        X[:, 12 + k] = bx + 3 * ((k % 3) - 1)
        Y[:, 12 + k] = -20 + 4 * k
    D = np.ones((n, 22))
    if gap:
        col, a, b = gap
        D[a:b, col] = 0.0
    return dict(f0=f0, n=n, cols=ids, cidx={p: i for i, p in enumerate(ids)}, X=X, Y=Y, D=D, BX=np.full(n, bx), BY=np.zeros(n),
                PER=np.ones(n, dtype=int), team=team, grp=grp, half_len=PITCH_HALF)


def make_events(shot_end=250):
    """Equipe 2 a la possession pendant tout le match ; la derniere possession se termine par un tir a l'image `shot_end`."""
    base = dict(event_type="player_possession", team_id=2, overall_pressure_start="low_pressure", possession_epv_for_start=0.03, organised_defense=True,
                team_score=0, opponent_team_score=0, minute_start=10.0, team_in_possession_phase_type="build_up", last_defensive_line_height_start=13.0, lead_to_shot=False)
    rows = [dict(base, frame_start=0, frame_end=199, end_type="pass"), dict(base, frame_start=200, frame_end=shot_end, end_type="shot", lead_to_shot=True)]
    return pd.DataFrame(rows)


def patch(monkeypatch, T, ev, mid=1):
    from highline import data as dat
    monkeypatch.setattr(dat, "load_match", lambda m: T)
    monkeypatch.setattr(dat, "load_events", lambda m: ev)
    monkeypatch.setattr(dat, "set_goal_reference", dat.set_goal_reference)
    dat.set_goal_reference("line")
