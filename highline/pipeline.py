"""Chaine complete : donnees ouvertes -> table de moments -> table de decision, leviers, cout physique, pose -> rapport, figure, tableau, outil HTML."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import config as cfg
from . import data as dat
from . import decision as dec
from . import checks, figure, i18n, levers, moment as mom, physical, pose, tool
from . import frames as fr
from .report import Report


def reliability_summary(df):
    return dict(all10=float((df.n_det == 10).mean()), all4=float((df.n_det_back4 == 4).mean()),
                sd_med=float(df.line_sd.median()), sd_p90=float(df.line_sd.quantile(0.9)), n=int(len(df)))


def evidence_table(cell_list, diffs, inter, model_v, lev, phys, rel, pose_res, lang="fr", chk=None):
    """Tableau 1 du README : une ligne par resultat, avec son statut."""
    idx = {c["key"]: c for c in cell_list}

    def one(dim, lab):
        return idx.get(tuple(lab if k == dim else "tous" for k in dec.DIM_ORDER))

    def cell_txt(dim, lab):
        c = one(dim, lab)
        if c is None:
            return "n/d"
        s = dec.summarize(c, 1.0)
        return f"{s['net']:+.2f} [{s['net_ci'][0]:+.2f} ; {s['net_ci'][1]:+.2f}]"
    est = {x["dim"] for x in diffs if x.get("ok")}
    T = i18n.TAB[lang]
    st = lambda ok: T["etab"] if ok else T["sugg"]
    ic = inter[1]
    f3 = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f} ; {v[2]:+.2f}]"
    rows = [
        (T["rel"], T["rel_v"].format(a=100 * rel["all10"], b=100 * rel["all4"], m=rel["sd_med"]), T["meas"]),
        (T["libre"], cell_txt("porteur", "libre"), st("porteur" in est)),
        (T["presse"], cell_txt("porteur", "presse"), st("porteur" in est)),
        (T["adv"], cell_txt("zone", "adversaire"), st("zone" in est)),
        (T["inter"], T["inter_v"].format(s=f3(ic["x_score_c"]), m=f3(ic["x_f_min"])), T["etab"] if inter[0] == "GO" else T["nc"]),
    ]
    if lev:
        rows.append((T["lev"], f"{lev['swap'][0]:.2f} [{lev['swap'][1]:.2f} ; {lev['swap'][2]:.2f}]", T["lev_s"]))
    if phys:
        rows.append((T["phys"], f"{phys['full'][0]:+.1f} [{phys['full'][1]:+.1f} ; {phys['full'][2]:+.1f}]", st(phys["verdict"] == "GO")))
    rows.append((T["model"], T["model_v"].format(v=model_v), T["nc"] if model_v == "NO-GO" else T["sugg"]))
    if chk:
        al, pb, c6 = chk.get("C1", {}), chk.get("C4", {}), chk.get("C6", {})
        fp = pb.get("fp")
        rows.append((T["chk"], T["chk_v"].format(a=100 * al["within05"] if "within05" in al else float("nan"), p=100 * max(fp) if fp else float("nan"),
                                                  c=c6.get("verdict", "n/d")), T["meas"] if all(chk.get(k, {}).get("verdict") in ("GO", "STABLE") for k in ("C1", "C2", "C3", "C4", "C6")) else T["verif"]))
    if pose_res:
        mde = [f"{r['mde']:.1f}" for r in pose_res["rows"] if r["y"] == "opp15"]
        rows.append((T["pose"], T["pose_v"].format(n=pose_res["shots"], m=" / ".join(mde)), T["expl"]))
    else:
        rows.append((T["pose"], T["pose_no"], T["expl"]))
    out = ["| " + " | ".join(T["head"]) + " |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(out)


def run(mode="fast", data_dir="./data", pose_dir="./hp_data", out_dir="./out", cache_dir="./cache", matches=None, nb=None, goal_reference=None, skip_checks=False, render=False):
    t0 = time.time()
    fast = mode == "fast"
    ids = list(matches or (cfg.FAST_MATCHES if fast else cfg.ALL_MATCHES))
    nb = nb or (60 if fast else 300)
    dat.set_dir(data_dir)
    if goal_reference:
        dat.set_goal_reference(goal_reference)
    ref = dat.goal_reference()
    pose.set_dir(pose_dir)
    out, cache = Path(out_dir), Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    rep = Report()
    rep.say(f"highline : mode {mode}, {len(ids)} matchs, {nb} tirages bootstrap par match, reference de but : {'ligne de but' if ref == 'line' else 'position mediane du gardien (premiere version)'}\n" + "=" * 78)
    if fast:
        rep.say("ATTENTION : le mode rapide (4 matchs) verifie que la chaine fonctionne ; ses resultats ne sont PAS representatifs (intervalles instables, differences artificielles). "
                "Les resultats du README viennent de : python reproduce.py --mode full\n" + "=" * 78)

    df = fr.build(ids, cache=cache / f"frames_{ref}.csv")
    rel = reliability_summary(df)
    rep.say(f"{rel['n']} moments defensifs ; 10 joueurs de champ observes : {100 * rel['all10']:.0f} % ; 4 plus bas observes : {100 * rel['all4']:.0f} % ; "
            f"incertitude de la ligne : mediane {rel['sd_med']:.1f} m, p90 {rel['sd_p90']:.1f} m")

    rep.say("\n" + "=" * 78 + "\nTABLE DE DECISION DE LA LIGNE HAUTE (bloc installe, balle >= 35 m de son but)")
    d = dec.prepare(df, strict=False)
    cell_list = dec.cells(d, nb=nb, rng=np.random.default_rng(204))
    diffs = dec.diff_tests(cell_list)
    rep.say(f"{len(d)} moments ; {d.match_id.nunique()} matchs ; 1 ecart-type de hauteur residuelle = {d.res.std():.1f} m ; tir adverse 15 s : {d.opp15.mean():.1%} ; tir de l'equipe 15 s : {d.own15.mean():.1%}\n")
    dec.report_table(cell_list, diffs, rep)

    rep.say("\n" + "=" * 78)
    ds = dec.prepare(df, strict=True)
    inter = dec.interaction_test(ds, rep)
    rep.say("\n" + "=" * 78)
    model_v = dec.model_validation(ds, rep, nb=nb)

    chk = None
    if not skip_checks:
        rep.say("\n" + "=" * 78)
        chk = checks.run(ids, d, ds, cell_list, diffs, rep, nb=nb, nperm=40 if fast else 200, nperm_model=8 if fast else 30, nperm_boot=15 if fast else 100, nb_boot=40 if fast else 200)

    rep.say("\n" + "=" * 78)
    rows = levers.build_rows(ids, cache=cache / f"losses_{ref}.csv")
    lev = levers.analyse(rows, rep, nb=nb)
    rep.say("\n" + "=" * 78)
    phys = physical.analyse(rows, rep)
    rep.say("\n" + "=" * 78)
    pose_res = pose.analyse(ds, rep, pose_dir) if all(m in ids for m in cfg.POSE_MATCHES) else (rep.say("BODY POSE : matchs a pose absents de cette selection.") or None)

    table = None
    for lang, suf in (("fr", ""), ("en", "_en")):
        figure.make(cell_list, lev, out / f"figure_1{suf}.png", lang)
        t = evidence_table(cell_list, diffs, inter, model_v, lev, phys, rel, pose_res, lang, chk)
        if fast:
            t = i18n.TAB[lang]["fast"] + "\n\n" + t
        (out / f"table_1{suf}.md").write_text(t, encoding="utf-8")
        table = table or t
    dec.export_cells(cell_list, d.res.std(), out / "decision_cells.json", goal_reference=ref)
    m = None
    try:
        sel = mom.select(df)
        if sel:
            m = mom.extract(*sel)
            (out / "moment.json").write_text(json.dumps(m, separators=(",", ":")), encoding="utf-8")
            rep.say(f"Moment illustre : match {sel[0]}, equipe {sel[1]}, image {sel[2]} (choisi par moment.select ; autre moment : python -m highline.render3d --match ... --team ... --frame ...)")
    except Exception as ex:                                              # la vue d'un moment est facultative
        rep.say(f"vue d'un moment non produite ({type(ex).__name__}: {ex})")
    tool.build(cell_list, diffs, lev, phys, rel, pose_res, out / "outil_decision.html", sd_line=d.res.std(), fast=fast, moment=m)
    if render and m is not None:
        from . import render3d
        for lang in ("fr", "en"):
            for style in ("linkedin", "plain"):
                for f in render3d.render(m, str(out / f"moment3d_{style}_{lang}"), lang=lang, style=style, fact_all10=rel["all10"]):
                    rep.say(f"rendu 3D : {f}")
    rep.say("\n" + "=" * 78 + f"\nFichiers ecrits dans {out}/ : rapport.txt, figure_1(.en).png, table_1(_en).md, outil_decision.html, decision_cells.json, moment.json  ({time.time() - t0:.0f} s)")
    rep.save(out / "rapport.txt")
    return dict(cells=cell_list, diffs=diffs, lev=lev, phys=phys, rel=rel, inter=inter, model=model_v, pose=pose_res, checks=chk, table=table)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reproduit l'analyse 'ligne haute' (donnees ouvertes SkillCorner A-League 2024/25).")
    ap.add_argument("--mode", choices=["fast", "full"], default="fast", help="fast : 4 matchs, ~5 min ; full : 20 matchs, resultats du README")
    ap.add_argument("--data-dir", default="./data", help="dossier de telechargement des donnees (non versionne)")
    ap.add_argument("--pose-dir", default="./hp_data", help="cache Body Pose (optionnel)")
    ap.add_argument("--out-dir", default="./out")
    ap.add_argument("--cache-dir", default="./cache")
    ap.add_argument("--matches", nargs="*", type=int)
    ap.add_argument("--bootstrap", type=int, help="nombre de tirages bootstrap par match")
    ap.add_argument("--goal-reference", choices=["line", "gk_median"], default="line",
                    help="origine des distances 'a son but' : ligne de but (defaut) ou position mediane du gardien (premiere version, pour comparer)")
    ap.add_argument("--skip-checks", action="store_true", help="ne pas lancer les controles de credibilite (placebo, alignement, etc.)")
    ap.add_argument("--render3d", action="store_true", help="ecrit aussi le rendu 3D du moment (PNG, GIF, MP4 si ffmpeg ; FR et EN ; ~4 min)")
    a = ap.parse_args(argv)
    run(a.mode, a.data_dir, a.pose_dir, a.out_dir, a.cache_dir, a.matches, a.bootstrap, a.goal_reference, a.skip_checks, a.render3d)


if __name__ == "__main__":
    main()
