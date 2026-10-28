"""Rendu 3D d'un moment reel (image fixe, GIF, MP4) : ce que la camera voit, ce qu'elle ne voit pas, et la ligne qui en decoule.

    python -m highline.render3d --out out/moment3d              # choisit un moment (moment.select) et ecrit .png, .gif et .mp4 si ffmpeg est present
    python -m highline.render3d --match 1996435 --team 869 --frame 10440 --lang en --style plain

Lecture : defenseurs en vert d'eau, attaquants en gris, balle en noir. Un defenseur que la camera ne voit pas a cet instant est un DISQUE ROUILLE : sa position est estimee,
le disque est le rayon a 90 % de l'erreur probable (couche de fiabilite). Le mur noir est la hauteur de ligne (4 defenseurs de champ les plus bas), la bande rouille son incertitude.
Les positions sont celles du fournisseur (SkillCorner, A-League 2024/25) ; aucune n'est inventee ; entre deux images (5 Hz) elles sont interpolees pour la fluidite.

Styles : `linkedin` = carre 1200 x 1200, fond creme, titre, signature (polices DM Serif Display et Lato telechargees si possible, licence OFL ; sinon polices par defaut, avec un avertissement) ;
`plain` = meme scene sans titre ni signature (pour la video du concours)."""
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
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

C = dict(bg="#f6f1e7", ink="#1c1917", ink2="#57534e", muted="#8a8580", grid="#e4dccd", teal="#0d8f84", rust="#cc4a12", compare="#c9c0b1", card="#fbf8f2")
FONT_DIR = os.path.expanduser("~/.fonts_jl")
_FILES = {"title": "dmserifdisplay/DMSerifDisplay-Regular.ttf", "body": "lato/Lato-Regular.ttf", "bold": "lato/Lato-Bold.ttf", "black": "lato/Lato-Black.ttf"}

TXT = {
    "fr": dict(title="La ligne haute que l’on mesure\nrepose sur des joueurs que l’on ne voit pas",
               sub=("Chaque disque rouille est un défenseur que la caméra ne voit pas à cet instant : sa position est estimée, le disque est l’erreur probable à 90 %.{fact}"),
               fact=" Les 10 défenseurs de champ ne sont tous observés que dans {a} % des moments (20 matchs d’A-League).",
               obs="4 plus bas observés : {k} sur 4", unc="incertitude de la ligne : ± {s} m", line="ligne",
               src="Source : SkillCorner Open Data, A-League 2024/25\n(tracking de diffusion). Code : github.com/Julie-Landrevie/highline"),
    "en": dict(title="The high line we measure rests on\nplayers the camera cannot see",
               sub=("Each rust disc is a defender the camera does not see at that instant: his position is estimated, the disc is the probable error at 90%.{fact}"),
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
                urllib.request.urlretrieve("https://raw.githubusercontent.com/google/fonts/main/ofl/" + rel, path)
            out[k] = fm.FontProperties(fname=path)
        except Exception:
            out[k] = fm.FontProperties(family="serif" if k == "title" else "sans-serif", weight="bold" if k in ("bold", "black") else "normal")
            warn = "polices DM Serif Display / Lato indisponibles : polices par defaut utilisees"
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


def prepare(m, k=3):
    """Series interpolees (k fois plus d'images) pour chaque joueur, la balle, la ligne et son incertitude."""
    P = []
    for p in m["players"]:
        P.append(dict(side=p["side"], gk=p["gk"], x=_interp(p["x"], k), y=_interp(p["y"], k), det=_interp(p["det"], k) >= 0.999, r=_interp(p["r"], k)))
    return dict(players=P, bx=_interp(m["ball"]["x"], k), by=_interp(m["ball"]["y"], k), line=_interp(m["line"], k), sd=_interp(m["sd"], k), n=(m["meta"]["n"] - 1) * k + 1, fps=m["meta"]["fps"] * k)


def _pitch(ax, xmin, xmax, L=105.0, W=68.0):
    hw = W / 2
    ground = [[xmin, -hw, 0], [L, -hw, 0], [L, hw, 0], [xmin, hw, 0]]       # le sol est le terrain : bords nets, pas de panneau tronque
    ax.add_collection3d(Poly3DCollection([ground], facecolor="#efe8d8", edgecolor="none", alpha=1.0, zorder=0))
    kw = dict(color=C["compare"], lw=1.2, zorder=1)
    ax.plot([xmin, L], [-hw, -hw], [0, 0], **kw)
    ax.plot([xmin, L], [hw, hw], [0, 0], **kw)
    ax.plot([L, L], [-hw, hw], [0, 0], **kw)
    ax.plot([L / 2, L / 2], [-hw, hw], [0, 0], **kw)
    t = np.linspace(0, 2 * np.pi, 60)
    ax.plot(L / 2 + 9.15 * np.cos(t), 9.15 * np.sin(t), np.zeros(60), **kw)
    for d, w in ((16.5, 20.16), (5.5, 9.16)):
        ax.plot([L - d, L - d, L], [-w, w, w], [0, 0, 0], **kw)
        ax.plot([L, L - d], [-w, -w], [0, 0], **kw)


def draw(ax, fig, S, i, T, readout, xlim=(34.0, 106.0), azim=(165.0, 195.0), elev=40.0, zoom=1.3):
    ax.cla()
    ax.computed_zorder = False                      # l'ordre de dessin est explicite (sol, lignes, disques, mur, joueurs, balle)
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    ax.set_xlim(*xlim)
    ax.set_ylim(-36, 36)
    ax.set_zlim(0, 12)
    ax.set_box_aspect((xlim[1] - xlim[0], 72, 14), zoom=zoom)
    ax.view_init(elev=elev, azim=azim[0] + (azim[1] - azim[0]) * i / max(1, S["n"] - 1))
    _pitch(ax, xlim[0], xlim[1])
    # incertitude de la ligne puis mur de la ligne
    L, sd = S["line"][i], S["sd"][i]
    if np.isfinite(L):
        s = sd if np.isfinite(sd) else 0.0
        ax.add_collection3d(Poly3DCollection([[[L - s, -34, 0.03], [L + s, -34, 0.03], [L + s, 34, 0.03], [L - s, 34, 0.03]]], facecolor=C["rust"], edgecolor="none", alpha=0.22, zorder=2))
        ax.add_collection3d(Poly3DCollection([[[L, -34, 0], [L, 34, 0], [L, 34, 3.2], [L, -34, 3.2]]], facecolor=C["ink"], edgecolor=C["ink"], alpha=0.10, linewidth=1.0, zorder=4))
    # joueurs
    for p in sorted(S["players"], key=lambda q: -q["x"][i] if np.isfinite(q["x"][i]) else 0):
        x, y = p["x"][i], p["y"][i]
        if not (np.isfinite(x) and np.isfinite(y)) or x < xlim[0]:
            continue
        if p["side"] == "a":
            col, al = C["muted"], 0.95
        else:
            col, al = C["teal"], 0.95
        seen = bool(p["det"][i])
        if p["side"] == "d" and not seen and not p["gk"] and np.isfinite(p["r"][i]):
            ax.add_collection3d(Poly3DCollection([_disc(x, y, max(p["r"][i], 0.4))], facecolor=C["rust"], edgecolor=C["rust"], alpha=0.28, linewidth=1.2, linestyle="--", zorder=3))
            al = 0.55
        ax.plot([x, x], [y, y], [0, 1.65], color=col, lw=6, alpha=al, solid_capstyle="round", zorder=5)
        ax.scatter([x], [y], [1.95], s=70, color=col, alpha=al, edgecolors=C["bg"], linewidths=0.8, zorder=5)
    if np.isfinite(S["bx"][i]):
        j0 = max(0, i - 8)
        ax.plot(S["bx"][j0:i + 1], S["by"][j0:i + 1], np.full(i + 1 - j0, 0.15), color=C["ink"], lw=1.2, alpha=0.5, zorder=6)
        ax.scatter([S["bx"][i]], [S["by"][i]], [0.3], s=90, color=C["ink"], edgecolors=C["bg"], linewidths=1.2, zorder=7)
    obs4 = _observed_of_deepest(S, i)
    readout[0].set_text(T["obs"].format(k=obs4) if obs4 is not None else "")
    readout[1].set_text(T["unc"].format(s=(f"{sd:.1f}".replace(".", ",") if T is TXT["fr"] else f"{sd:.1f}")) if np.isfinite(sd) else "")


def _observed_of_deepest(S, i):
    d = [(p["x"][i], bool(p["det"][i])) for p in S["players"] if p["side"] == "d" and not p["gk"] and np.isfinite(p["x"][i])]
    if len(d) < 4:
        return None
    d.sort()
    return sum(1 for _, o in d[:4] if o)


def render(m, out_base, lang="fr", style="linkedin", k=3, fact_all10=None, gif=True, mp4=True, size=1200, still_at=0.5):
    """Ecrit out_base.png (image fixe), .gif et .mp4 (si ffmpeg est disponible). Retourne la liste des fichiers."""
    S = prepare(m, k)
    T = TXT[lang]
    linkedin = style == "linkedin"
    F, warn = _fonts() if linkedin else ({}, None)
    if warn:
        print("ATTENTION :", warn)
    dpi = 120
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    fig.patch.set_facecolor(C["bg"])
    ax = fig.add_axes([0.0, 0.19, 1.0, 0.58] if linkedin else [0.0, 0.0, 1.0, 1.0], projection="3d")
    zoom = 1.3 if linkedin else 1.0
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
    draw(ax, fig, S, ist, T, (r1, r2), zoom=zoom)
    fig.savefig(f"{out_base}.png", facecolor=C["bg"])
    out.append(f"{out_base}.png")
    can_mp4 = mp4 and animation.writers.is_available("ffmpeg")
    if mp4 and not can_mp4:
        print("ffmpeg absent : MP4 non produit (GIF seulement). Installer ffmpeg pour le MP4.")
    if can_mp4 or gif:
        anim = animation.FuncAnimation(fig, lambda i: draw(ax, fig, S, i, T, (r1, r2), zoom=zoom), frames=S["n"], interval=1000 / S["fps"])
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
    for f in render(m, a.out, a.lang, a.style, fact_all10=a.fact_all10, gif=not a.no_gif, mp4=not a.no_mp4):
        print("ecrit :", f)


if __name__ == "__main__":
    main()
