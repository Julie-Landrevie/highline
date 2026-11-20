"""Rendu 3D d'un moment reel (image fixe, GIF, MP4) : ce que la camera voit, ce qu'elle ne voit pas, et la ligne qui en decoule.

    python -m highline.render3d --out out/moment3d              # choisit un moment (moment.select) et ecrit .png, .gif et .mp4 si ffmpeg est present
    python -m highline.render3d --match 1996435 --team 869 --frame 10440 --lang en --style plain

Lecture : defenseurs en vert d'eau, attaquants en gris, balle blanche. Un defenseur que la camera ne voit pas a cet instant est un DISQUE ROUILLE : sa position est estimee,
le disque est le rayon a 90 % de l'erreur probable (couche de fiabilite). Le mur noir est la hauteur de ligne (4 defenseurs de champ les plus bas), la bande rouille son incertitude.
Les positions sont celles du fournisseur (SkillCorner, A-League 2024/25) ; aucune n'est inventee ; entre deux images (5 Hz) elles sont interpolees pour la fluidite.

Styles : `linkedin` = carre 1200 x 1200, fond creme, titre, signature (polices DM Serif Display et Lato telechargees si possible, licence OFL ; sinon polices par defaut, avec un avertissement) ;
`plain` = meme scene sans titre ni signature (pour la video du concours).

Aspect (`--look`) : `charte` (defaut) = pelouse aux teintes cremes de la charte ; `realiste` = pelouse verte, lignes blanches. Les couleurs de lecture de la charte (vert d'eau, gris, rouille) sont gardees en `charte` ; en `realiste`, les defenseurs sont bleus et l'anneau bleu clair, car le vert d'eau se confond avec la pelouse.
Les joueurs sont des volumes (jambes, buste, tete) avec ombre au sol ; le ballon est une sphere a pentagones qui roule avec son deplacement (agrandi x2,5) ; un joueur non detecte est translucide. Les cages sont dessinees a la taille reglementaire (7,32 m x 2,44 m).
La hauteur des joueurs est un gabarit unique (1,8 m) : le tracking ne donne que (x, y)."""
import argparse
import json
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import font_manager as fm
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

C = dict(bg="#f6f1e7", ink="#1c1917", ink2="#57534e", muted="#8a8580", grid="#e4dccd", teal="#0d8f84", rust="#cc4a12", compare="#c9c0b1", card="#fbf8f2")
FONT_DIR = os.path.expanduser("~/.fonts_jl")
_FILES = {"title": "dmserifdisplay/DMSerifDisplay-Regular.ttf", "body": "lato/Lato-Regular.ttf", "bold": "lato/Lato-Bold.ttf", "black": "lato/Lato-Black.ttf"}

TXT = {
    "fr": dict(title="La ligne haute que l’on mesure\nrepose sur des joueurs que l’on ne voit pas",
               sub=("La ligne est la moyenne des 4 défenseurs les plus bas. Anneau vert : observé par la caméra ; disque rouille : position estimée, le disque est l’erreur probable à 90 %.{fact}"),
               fact=" Les 10 défenseurs de champ ne sont tous observés que dans {a} % des moments (20 matchs d’A-League).",
               obs="4 plus bas observés : {k} sur 4", unc="incertitude de la ligne : ± {s} m", line="ligne",
               src="Source : SkillCorner Open Data, A-League 2024/25\n(tracking de diffusion). Code : github.com/Julie-Landrevie/highline"),
    "en": dict(title="The high line we measure rests on\nplayers the camera cannot see",
               sub=("The line is the mean of the 4 deepest defenders. Green ring: observed by the camera; rust disc: estimated position, the disc is the probable error at 90%.{fact}"),
               fact=" All 10 outfield defenders are observed in only {a}% of moments (20 A-League matches).",
               obs="4 deepest observed: {k} of 4", unc="line uncertainty: ± {s} m", line="line",
               src="Source: SkillCorner Open Data, A-League 2024/25\n(broadcast tracking). Code: github.com/Julie-Landrevie/highline"),
}


def _fonts():
    """DM Serif Display + Lato (OFL, Google Fonts). Retourne (dict de FontProperties, avertissement ou None)."""
    out, warn = {}, None
    for k, rel in _FILES.items():
        try:
            os.makedirs(FONT_DIR, exist_ok=True)
            path = os.path.join(FONT_DIR, os.path.basename(rel))
            if not os.path.exists(path):
                if os.environ.get("HIGHLINE_DOWNLOAD_FONTS") != "1":      # aucun telechargement par defaut : polices locales ou polices du systeme
                    raise FileNotFoundError(path)
                urllib.request.urlretrieve("https://raw.githubusercontent.com/google/fonts/main/ofl/" + rel, path)   # option, polices libres (OFL) pour la mise en page des visuels
            out[k] = fm.FontProperties(fname=path)
        except Exception:
            out[k] = fm.FontProperties(family="serif" if k == "title" else "sans-serif", weight="bold" if k in ("bold", "black") else "normal")
            warn = "polices DM Serif Display / Lato absentes : polices par defaut utilisees (HIGHLINE_DOWNLOAD_FONTS=1 pour les telecharger)"
    return out, warn


def _interp(a, k):
    """Interpolation lineaire d'une liste (None = manquant) a k fois la frequence."""
    a = np.array([np.nan if v is None else v for v in a], float)
    n = len(a)
    xs = np.linspace(0, n - 1, (n - 1) * k + 1)
    out = np.interp(xs, np.arange(n), a)
    miss = np.isnan(a)
    if miss.any():                                       # un point manquant reste manquant autour de lui
        near = np.interp(xs, np.arange(n), miss.astype(float)) > 0
        out[near] = np.nan
    return out


def _disc(cx, cy, r, z=0.02, n=40):
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t), np.full(n, z)])


def _near(a, k, dtype=float):
    """Valeur de l'image la plus proche (pas d'interpolation : un joueur est observe ou non)."""
    n = len(a)
    idx = np.rint(np.linspace(0, n - 1, (n - 1) * k + 1)).astype(int)
    return np.array([np.nan if a[i] is None else a[i] for i in idx], dtype=dtype)


def prepare(m, k=3):
    """Series interpolees (k fois plus d'images) pour chaque joueur, la balle, la ligne et son incertitude.

    Repere d'affichage : x' = L - x et y' = - y (demi-tour du terrain). Le but defendu par l'equipe verte (x = 0 dans les donnees) est donc au FOND de l'image, la camera est
    derriere les attaquants. Les 4 plus bas (les plus proches de leur but, rang calcule sur le repere d'origine a chaque image) sont marques : ce sont eux, et eux seuls, qui
    portent un anneau (observe) ou un disque d'incertitude (estime)."""
    L = float(m["meta"]["pitch"][0])
    P = []
    for p in m["players"]:
        P.append(dict(side=p["side"], gk=p["gk"], depth=_interp(p["x"], k), x=L - _interp(p["x"], k), y=-_interp(p["y"], k),
                      det=_near(p["det"], k) >= 0.5, r=_near(p["r"], k)))
    n = (m["meta"]["n"] - 1) * k + 1
    for q in P:
        q["deep4"] = np.zeros(n, bool)
    cand = [q for q in P if q["side"] == "d" and not q["gk"]]
    for i in range(n):
        ok = [q for q in cand if np.isfinite(q["depth"][i])]
        if len(ok) >= 4:
            for q in sorted(ok, key=lambda q: q["depth"][i])[:4]:
                q["deep4"][i] = True
    bx = L - _interp(m["ball"]["x"], k)
    by_ = -_interp(m["ball"]["y"], k)
    step = np.hypot(np.diff(bx, prepend=bx[0]), np.diff(by_, prepend=by_[0]))
    bpath = np.cumsum(np.nan_to_num(step))                                      # chemin parcouru par le ballon (pour le faire rouler)
    xs = np.concatenate([q["x"][np.isfinite(q["x"])] for q in P if not q["gk"]] + [bx[np.isfinite(bx)]])
    xlo = float(max(0.0, np.floor(xs.min()) - 8.0)) if len(xs) else 22.0          # cadrage : on garde 8 m derriere le joueur ou le ballon le plus proche de la camera
    return dict(players=P, bx=bx, by=-_interp(m["ball"]["y"], k), line=L - _interp(m["line"], k), sd=_interp(m["sd"], k), n=n, fps=m["meta"]["fps"] * k, L=L, xlo=xlo, bpath=bpath)


LOOKS = {
    "charte": dict(rust="#cc4a12", grass=("#efe8d8", "#e9e0cc"), surround="#e4dccd", line="#c9c0b1", goal="#57534e", net="#8a8580", att="#8a8580", dfd="#0d8f84", ring="#0d8f84", ball="#fdfcf8", shadow=0.20),
    "realiste": dict(rust="#f2682a", grass=("#5c8a53", "#548049"), surround="#476d40", line="#f3f0e6", goal="#f3f0e6", net="#e6e2d6", att="#e6e2d8", dfd="#2f6fdc", ring="#9fd0ff", ball="#ffffff", shadow=0.30),
}
_LIGHT = np.array([-0.45, -0.35, 0.82])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)
SCALE = 1.25                                               # gabarit agrandi de 25 % pour rester lisible a cette distance
HEIGHT = dict(legs=0.88 * SCALE, torso=1.52 * SCALE, head=1.69 * SCALE)   # gabarit unique : le tracking ne donne que (x, y)


def _shade(rgb, normal, alpha=1.0, amb=0.60, gain=0.55):
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    f = amb + gain * max(0.0, float(n @ _LIGHT))
    return tuple(min(1.0, c * f) for c in rgb) + (alpha,)


def _mix(rgb, other, t):
    return tuple(a * (1 - t) + b * t for a, b in zip(rgb, other))


def _frustum(x, y, z0, z1, r0, r1, rgb, alpha, n=8):
    t = np.linspace(0, 2 * np.pi, n + 1)
    polys, cols = [], []
    for k in range(n):
        a0, a1 = t[k], t[k + 1]
        polys.append([[x + r0 * np.cos(a0), y + r0 * np.sin(a0), z0], [x + r0 * np.cos(a1), y + r0 * np.sin(a1), z0],
                      [x + r1 * np.cos(a1), y + r1 * np.sin(a1), z1], [x + r1 * np.cos(a0), y + r1 * np.sin(a0), z1]])
        am = (a0 + a1) / 2
        cols.append(_shade(rgb, [np.cos(am), np.sin(am), 0.15], alpha))
    polys.append([[x + r1 * np.cos(a), y + r1 * np.sin(a), z1] for a in t[:-1]])
    cols.append(_shade(rgb, [0, 0, 1.0], alpha))
    return polys, cols


def _sphere(x, y, z, r, rgb, alpha, nlat=4, nlon=8):
    lat, lon = np.linspace(-np.pi / 2, np.pi / 2, nlat + 1), np.linspace(0, 2 * np.pi, nlon + 1)
    polys, cols = [], []
    for i in range(nlat):
        for j in range(nlon):
            P = np.array([[x + r * np.cos(a) * np.cos(b), y + r * np.cos(a) * np.sin(b), z + r * np.sin(a)]
                          for a, b in ((lat[i], lon[j]), (lat[i], lon[j + 1]), (lat[i + 1], lon[j + 1]), (lat[i + 1], lon[j]))])
            polys.append(P.tolist())
            cols.append(_shade(rgb, P.mean(axis=0) - np.array([x, y, z]), alpha))
    return polys, cols


def _body(x, y, rgb, alpha):
    """Joueur = jambes + buste (troncs de cone) + tete : un seul objet pour garder l'ordre de dessin."""
    legs, cl = _frustum(x, y, 0.0, HEIGHT["legs"], 0.17 * SCALE, 0.21 * SCALE, _mix(rgb, (0, 0, 0), 0.25), alpha)
    torso, ct = _frustum(x, y, HEIGHT["legs"], HEIGHT["torso"], 0.27 * SCALE, 0.24 * SCALE, rgb, alpha)
    head, ch = _sphere(x, y, HEIGHT["head"], 0.14 * SCALE, _mix(rgb, (1, 1, 1), 0.45), alpha)
    return legs + torso + head, cl + ct + ch


BALL_R = 0.11 * 2.5                                           # rayon reglementaire (0,11 m) x 2,5 pour rester lisible
_PHI = (1 + 5 ** 0.5) / 2
_ICO = np.array([(0, s1, s2 * _PHI) for s1 in (-1, 1) for s2 in (-1, 1)] + [(s1, s2 * _PHI, 0) for s1 in (-1, 1) for s2 in (-1, 1)]
                + [(s2 * _PHI, 0, s1) for s1 in (-1, 1) for s2 in (-1, 1)], float)
_ICO /= np.linalg.norm(_ICO, axis=1)[:, None]                  # 12 directions : les 12 pentagones noirs d'un ballon


def _rot(axis, ang):
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n < 1e-9:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _ball(cx, cy, r, R, white, dark, nlat=8, nlon=14):
    """Ballon : sphere eclairee, pentagones sombres ; R = rotation (le ballon roule : le motif tourne avec le chemin parcouru)."""
    lat, lon = np.linspace(-np.pi / 2, np.pi / 2, nlat + 1), np.linspace(0, 2 * np.pi, nlon + 1)
    polys, cols = [], []
    for i in range(nlat):
        for j in range(nlon):
            P = np.array([[cx + r * np.cos(a) * np.cos(b), cy + r * np.cos(a) * np.sin(b), r + r * np.sin(a)]
                          for a, b in ((lat[i], lon[j]), (lat[i], lon[j + 1]), (lat[i + 1], lon[j + 1]), (lat[i + 1], lon[j]))])
            nrm = P.mean(axis=0) - np.array([cx, cy, r])
            nrm = nrm / np.linalg.norm(nrm)
            patch = float((_ICO @ (R.T @ nrm)).max()) > 0.90
            polys.append(P.tolist())
            cols.append(_shade(to_rgb(dark if patch else white), nrm, 1.0, amb=0.72, gain=0.40))
    return polys, cols


def _ellipse(cx, cy, rx, ry, z=0.012, n=24):
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t), np.full(n, z)])


def _pitch(ax, xmin, xmax, L=105.0, W=68.0, look="charte"):
    cfg_ = LOOKS[look]
    hw = W / 2
    ax.add_collection3d(Poly3DCollection([[[xmin, -hw - 3, -0.02], [xmax, -hw - 3, -0.02], [xmax, hw + 3, -0.02], [xmin, hw + 3, -0.02]]], facecolor=cfg_["surround"], edgecolor="none", zorder=0))
    nstr, step = 18, L / 18                                  # bandes de tonte
    stripes, cols = [], []
    for k in range(nstr):
        a, b = L - (k + 1) * step, L - k * step
        if b <= xmin:
            continue
        a = max(a, xmin)
        stripes.append([[a, -hw, 0], [b, -hw, 0], [b, hw, 0], [a, hw, 0]])
        cols.append(cfg_["grass"][k % 2])
    ax.add_collection3d(Poly3DCollection(stripes, facecolors=cols, edgecolors=cols, linewidths=0.3, zorder=0.5))
    kw = dict(color=cfg_["line"], lw=1.5, zorder=1)
    ax.plot([xmin, L], [-hw, -hw], [0, 0], **kw)
    ax.plot([xmin, L], [hw, hw], [0, 0], **kw)
    ax.plot([L, L], [-hw, hw], [0, 0], **kw)
    ax.plot([L / 2, L / 2], [-hw, hw], [0, 0], **kw)
    t = np.linspace(0, 2 * np.pi, 80)
    ax.plot(L / 2 + 9.15 * np.cos(t), 9.15 * np.sin(t), np.zeros(80), **kw)
    for d, w in ((16.5, 20.16), (5.5, 9.16)):
        ax.plot([L - d, L - d, L], [-w, w, w], [0, 0, 0], **kw)
        ax.plot([L, L - d], [-w, -w], [0, 0], **kw)
    ax.plot([L - 11], [0], [0], marker="o", ms=3, color=cfg_["line"], zorder=1)          # point de penalty
    a = np.arccos(5.5 / 9.15)
    ta = np.linspace(np.pi - a, np.pi + a, 30)                                              # arc de la surface
    ax.plot(L - 11 + 9.15 * np.cos(ta), 9.15 * np.sin(ta), np.zeros(30), **kw)
    # cage reglementaire : 7,32 m x 2,44 m, filet de 2 m de profondeur
    g, h, dp = 3.66, 2.44, 2.0
    posts = [[(L, -g, 0), (L, -g, h)], [(L, g, 0), (L, g, h)], [(L, -g, h), (L, g, h)]]
    ax.add_collection3d(Line3DCollection(posts, colors=cfg_["goal"], linewidths=2.6, zorder=1.6))
    net = []
    for yy in np.linspace(-g, g, 15):
        net.append([(L + dp, yy, 0), (L + dp, yy, h)])
    for zz in np.linspace(0, h, 7):
        net.append([(L + dp, -g, zz), (L + dp, g, zz)])
    for sgn in (-1, 1):
        for zz in np.linspace(0, h, 5):
            net.append([(L, sgn * g, zz), (L + dp, sgn * g, zz)])
    for yy in np.linspace(-g, g, 9):
        net.append([(L, yy, h), (L + dp, yy, h)])
    net += [[(L, -g, h), (L + dp, -g, h)], [(L, g, h), (L + dp, g, h)]]
    ax.add_collection3d(Line3DCollection(net, colors=cfg_["net"], linewidths=0.5, alpha=0.55, zorder=1.4))


def draw(ax, fig, S, i, T, readout, xlim=(None, None), azim=(165.0, 195.0), elev=28.0, zoom=1.3, look="charte"):
    cfg_ = LOOKS[look]
    xlim = (xlim[0] if xlim[0] is not None else S["xlo"], xlim[1] if xlim[1] is not None else S["L"] + 3.5)
    ax.cla()
    ax.computed_zorder = False                      # l'ordre de dessin est explicite (sol, lignes, disques, mur, joueurs, balle)
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    ax.set_xlim(*xlim)
    ax.set_ylim(-38, 38)
    ax.set_zlim(0, 12)
    ax.set_box_aspect((xlim[1] - xlim[0], 76, 14), zoom=zoom)
    ax.view_init(elev=elev, azim=azim[0] + (azim[1] - azim[0]) * i / max(1, S["n"] - 1))
    _pitch(ax, xlim[0], xlim[1], L=S["L"], look=look)
    # incertitude de la ligne puis mur de la ligne
    L, sd = S["line"][i], S["sd"][i]
    if np.isfinite(L):
        s = sd if np.isfinite(sd) else 0.0
        ax.add_collection3d(Poly3DCollection([[[L - s, -34, 0.03], [L + s, -34, 0.03], [L + s, 34, 0.03], [L - s, 34, 0.03]]], facecolor=cfg_["rust"], edgecolor="none", alpha=0.34, zorder=2))
        ax.add_collection3d(Poly3DCollection([[[L, -34, 0], [L, 34, 0], [L, 34, 3.2], [L, -34, 3.2]]], facecolor=C["ink"], edgecolor=C["ink"], alpha=0.10, linewidth=1.0, zorder=4))
    # joueurs (du fond vers l'avant)
    for p in sorted(S["players"], key=lambda q: -q["x"][i] if np.isfinite(q["x"][i]) else 0):
        x, y = p["x"][i], p["y"][i]
        if not (np.isfinite(x) and np.isfinite(y)) or x < xlim[0]:
            continue
        rgb = to_rgb(cfg_["att"] if p["side"] == "a" else cfg_["dfd"])
        seen = bool(p["det"][i])
        deep = bool(p["deep4"][i])
        if (deep or p["gk"]) and not seen and np.isfinite(p["r"][i]):          # le gardien non detecte a aussi son disque d'incertitude (plus leger)
            ax.add_collection3d(Poly3DCollection([_disc(x, y, max(p["r"][i], 0.4))], facecolor=cfg_["rust"], edgecolor=cfg_["rust"], alpha=0.30 if deep else 0.16, linewidth=1.4, linestyle="--", zorder=3))
        if deep and seen:
            ring = _disc(x, y, 1.5, z=0.04)
            ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=cfg_["ring"], lw=2.4, alpha=0.95, zorder=3)
        al = 1.0 if seen else 0.42                                   # non detecte : translucide (position estimee)
        ax.add_collection3d(Poly3DCollection([_ellipse(x + 0.45, y + 0.30, 0.62, 0.36)], facecolor="black", edgecolor="none", alpha=cfg_["shadow"] * (1.0 if seen else 0.5), zorder=3.5))
        polys, cols = _body(x, y, rgb, al)
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.25, zorder=5))
    if np.isfinite(S["bx"][i]) and S["bx"][i] >= xlim[0]:
        j0 = max(0, i - 8)
        ax.plot(np.maximum(S["bx"][j0:i + 1], xlim[0]), S["by"][j0:i + 1], np.full(i + 1 - j0, 0.05), color=C["ink"], lw=1.2, alpha=0.40, zorder=6)
        ax.add_collection3d(Poly3DCollection([_ellipse(S["bx"][i] + 0.10, S["by"][i] + 0.07, BALL_R * 1.15, BALL_R * 0.8)], facecolor="black", edgecolor="none", alpha=cfg_["shadow"] * 1.3, zorder=6.2))
        jb = max(0, i - 3)
        dx, dy = S["bx"][i] - S["bx"][jb], S["by"][i] - S["by"][jb]
        R = _rot((-dy, dx, 0.0), S["bpath"][i] / BALL_R) if np.hypot(dx, dy) > 1e-6 else np.eye(3)
        polys, cols = _ball(S["bx"][i], S["by"][i], BALL_R, R, cfg_["ball"], "#1c1917")
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors=cols, linewidths=0.2, zorder=7))
    obs4 = _observed_of_deepest(S, i)
    readout[0].set_text(T["obs"].format(k=obs4) if obs4 is not None else "")
    readout[1].set_text(T["unc"].format(s=(f"{sd:.1f}".replace(".", ",") if T is TXT["fr"] else f"{sd:.1f}")) if np.isfinite(sd) else "")


def _observed_of_deepest(S, i):
    d = [bool(p["det"][i]) for p in S["players"] if p["deep4"][i]]
    return sum(d) if len(d) == 4 else None


def render(m, out_base, lang="fr", style="linkedin", k=3, fact_all10=None, gif=True, mp4=True, size=1200, still_at=0.5, look="charte", fmt="square"):
    """Ecrit out_base.png (image fixe), .gif et .mp4 (si ffmpeg est disponible). Retourne la liste des fichiers."""
    S = prepare(m, k)
    T = TXT[lang]
    linkedin = style == "linkedin"
    F, warn = _fonts() if linkedin else ({}, None)
    if warn:
        print("ATTENTION :", warn)
    dpi = 120
    video = (fmt == "video") and not linkedin                  # 16:9 (1920 x 1080) pour la video du concours
    fig = plt.figure(figsize=((size * 1.6 if video else size) / dpi, (size * 0.9 if video else size) / dpi), dpi=dpi)
    fig.patch.set_facecolor(C["bg"])
    # matplotlib impose un cadre 3D CARRE et coupe ce qui depasse : on donne donc au cadre un carre aussi large que l'image
    # (il deborde de la figure en hauteur), sinon le terrain est tronque a gauche et a droite.
    side = 1.6 / 0.9 if video else 1.0                         # cote du carre, en hauteurs de figure
    cy = 0.48 if linkedin else (0.60 if video else 0.56)        # centre vertical du terrain dans l'image
    ax = fig.add_axes([0.0, cy - side / 2, 1.0, side], projection="3d")
    zoom = 1.15 if linkedin else (1.0 if video else 1.25)
    ax.set_proj_type("persp", focal_length=0.45)
    fp = lambda key, **kw: dict(fontproperties=F[key], **kw) if F else dict(**kw)
    if linkedin:
        fig.text(0.07, 0.93, T["title"], va="top", color=C["ink"], linespacing=1.1, **(dict(fontproperties=F["title"], fontsize=31) if F else dict(fontsize=28, family="serif")))
        fact = T["fact"].format(a=f"{100 * fact_all10:.0f}") if fact_all10 is not None else ""
        sub = T["sub"].format(fact=fact)
        import textwrap
        fig.text(0.07, 0.795, "\n".join(textwrap.wrap(sub, 84)), va="top", color=C["ink2"], linespacing=1.4, **(dict(fontproperties=F["body"], fontsize=15) if F else dict(fontsize=14)))
        fig.text(0.07, 0.035, T["src"], color=C["ink2"], linespacing=1.4, **(dict(fontproperties=F["body"], fontsize=11) if F else dict(fontsize=10)))
        fig.patches.append(FancyBboxPatch((0.855, 0.035), 0.06, 0.05, boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure, fc="none", ec=C["teal"], lw=2.2, figure=fig))
        fig.text(0.885, 0.059, "JL", ha="center", va="center", color=C["teal"], **(dict(fontproperties=F["bold"], fontsize=18) if F else dict(fontsize=16, weight="bold")))
        fig.text(0.845, 0.068, "Julie Landrevie", ha="right", va="center", color=C["ink"], **(dict(fontproperties=F["bold"], fontsize=13) if F else dict(fontsize=12, weight="bold")))
        fig.text(0.845, 0.045, "Data & performance football", ha="right", va="center", color=C["ink2"], **(dict(fontproperties=F["body"], fontsize=11) if F else dict(fontsize=10)))
    ry = 0.135 if linkedin else 0.93
    r1 = fig.text(0.07, ry, "", color=C["ink"], **(dict(fontproperties=F["black"], fontsize=15) if F else dict(fontsize=14, weight="bold")))
    r2 = fig.text(0.07, ry - 0.032 if linkedin else ry - 0.04, "", color=C["rust"], **(dict(fontproperties=F["bold"], fontsize=15) if F else dict(fontsize=14, weight="bold")))
    out = []
    ist = int(round((S["n"] - 1) * still_at))
    draw(ax, fig, S, ist, T, (r1, r2), zoom=zoom, look=look)
    fig.savefig(f"{out_base}.png", facecolor=C["bg"])
    out.append(f"{out_base}.png")
    can_mp4 = mp4 and animation.writers.is_available("ffmpeg")
    if mp4 and not can_mp4:
        print("ffmpeg absent : MP4 non produit (GIF seulement). Installer ffmpeg pour le MP4.")
    if can_mp4 or gif:
        anim = animation.FuncAnimation(fig, lambda i: draw(ax, fig, S, i, T, (r1, r2), zoom=zoom, look=look), frames=S["n"], interval=1000 / S["fps"])
        if can_mp4:
            anim.save(f"{out_base}.mp4", writer=animation.FFMpegWriter(fps=S["fps"], codec="libx264", extra_args=["-pix_fmt", "yuv420p"]), dpi=dpi, savefig_kwargs=dict(facecolor=C["bg"]))
            out.append(f"{out_base}.mp4")
        if gif:
            anim.save(f"{out_base}.gif", writer=animation.PillowWriter(fps=S["fps"]), dpi=60, savefig_kwargs=dict(facecolor=C["bg"]))
            out.append(f"{out_base}.gif")
    plt.close(fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out/moment3d", help="prefixe des fichiers produits")
    ap.add_argument("--lang", choices=["fr", "en"], default="fr")
    ap.add_argument("--style", choices=["linkedin", "plain"], default="linkedin")
    ap.add_argument("--look", choices=["charte", "realiste"], default="charte", help="aspect de la scene : pelouse creme (charte) ou verte (realiste)")
    ap.add_argument("--format", choices=["square", "video"], default="square", help="video = 1920 x 1080 (style plain seulement)")
    ap.add_argument("--moment-json", help="moment deja extrait (out/moment.json) ; sinon il est choisi dans les donnees")
    ap.add_argument("--match", type=int)
    ap.add_argument("--team", type=int)
    ap.add_argument("--frame", type=int)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--fact-all10", type=float, help="part des moments avec les 10 joueurs observes (0 a 1), ecrite dans le sous-titre")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--no-mp4", action="store_true")
    a = ap.parse_args(argv)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.moment_json:
        m = json.loads(Path(a.moment_json).read_text(encoding="utf-8"))
    else:
        from . import config as cfg, data as dat, frames as fr, moment
        dat.set_dir(a.data_dir)
        if a.match and a.team and a.frame:
            sel = (a.match, a.team, a.frame)
        else:
            df = fr.build(cfg.ALL_MATCHES if not os.environ.get("HIGHLINE_FAST") else cfg.FAST_MATCHES, cache=Path(a.cache_dir) / f"frames_{dat.goal_reference()}.csv")
            sel = moment.select(df)
            if sel is None:
                raise SystemExit("aucun moment ne repond aux criteres ; precisez --match --team --frame")
        m = moment.extract(*sel)
    for f in render(m, a.out, a.lang, a.style, fact_all10=a.fact_all10, gif=not a.no_gif, mp4=not a.no_mp4, look=a.look, fmt=a.format):
        print("ecrit :", f)


if __name__ == "__main__":
    main()
