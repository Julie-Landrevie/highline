# API publique / Public API

**FR.** Lire une hauteur de ligne avec son incertitude, puis le bilan de la situation, sans les données brutes : il suffit de la table `out/decision_cells.json` écrite par `python reproduce.py`.
**EN.** Read a line height with its uncertainty, then the situation's net balance, without the raw data: you only need `out/decision_cells.json`, written by `python reproduce.py`.

```python
from highline.api import load_table, line_decision

table = load_table("out/decision_cells.json")
r = line_decision(
    table,
    dist_to_goal=[18, 20, 22, 25, 30, 34, 38, 41, 45, 50],   # 10 outfield players, metres from their own goal line
    gap_s=[0, 0, 12, 0, 0, 0, 0, 0, 0, 0],                    # seconds since last observed (0 = observed); optional
    habitual_line_m=19.0,                                     # usual line height for this ball position (api.fit_habit)
    ball_x=55.0,                                              # metres from the defending team's goal line (>= 35)
    carrier="libre",                                          # "libre" (free) or "presse" (pressed)
    w=1.0,                                                    # weight of a conceded shot, in created shots
    lang="en",                                                # "fr" or "en"
)
print(r["message"])
```

Returned: `observation` (line, uncertainty, observed players, reliability level), `residual_m`, `decision` (gain, cost, net, 95 % interval, w*, tag), `p_sign` (share of plausible position draws that keep the sign of the gap to the usual height) and `warnings`.

**Conventions.** Metres; distance from the defending team's goal line; line height = mean of the 4 deepest outfield players. The numbers are an **association** observed in 20 A-League 2024/25 matches, not a causal recommendation. / Association observée sur 20 matchs de l'A-League 2024/25, pas une recommandation causale.

Other functions: `observe_line` (line and uncertainty), `fit_habit` (usual line from a team's history), `decide` (one situation of the table).

3D render of a real moment: `python -m highline.render3d --out out/moment3d` (`--lang en`, `--style plain`).
