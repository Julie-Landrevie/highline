"""Constantes du projet. Aucune donnee externe : uniquement le depot ouvert SkillCorner (tracking, evenements, Body Pose)."""
RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/refs/heads/master/data/matches"
MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches"
HF_REPO_ID = "SkillCorner/opendata-bodypose"

ALL_MATCHES = [2017461, 2016236, 2015213, 2013725, 2011166, 2010085, 2007721, 2007448, 2006363, 2006229,
               1996435, 1996436, 1986691, 1959846, 1953632, 1927964, 1925299, 1899585, 1886347, 1874553]
POSE_MATCHES = (1925299, 1996435)
FAST_MATCHES = [1925299, 1996435, 2017461, 2016236]      # mode rapide : 4 matchs (les 2 matchs a Body Pose + 2)

DEF_ROLES = {"CB", "LCB", "RCB", "LB", "RB", "LWB", "RWB"}
MID_ROLES = {"DM", "LDM", "RDM", "CM", "LCM", "RCM", "AM", "LM", "RM", "MC"}

FPS = 10
STEP = 20                 # un moment defensif toutes les 2 s
HORIZON = 150             # 15 s
HSR_MS = 5.5              # course rapide (m/s)
MIN_CELL = 400            # effectif minimal d'une situation du tableau de decision
MIN_EVENTS = 25           # nombre minimal d'evenements d'une strate de leviers

# Erreur de position d'un joueur extrapole (m) selon la duree du trou (s) : sigma ; rayon a 90 % = 2,146 sigma (erreur 2D)
SIG_T = [0, 1, 2, 5, 10, 30]
SIG_V = [0.10, 0.14, 0.44, 1.8, 4.3, 6.0]
R90 = 2.146
