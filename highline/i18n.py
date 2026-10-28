"""Textes bilingues (francais / English) de la figure et du tableau. L'outil HTML a son propre dictionnaire (tool.py)."""

DIMN = {
    "fr": {"porteur": "porteur", "zone": "zone", "score": "score", "minute": "minute",
           "libre": "libre", "presse": "presse", "milieu": "milieu", "adversaire": "chez l'adversaire",
           "derriere": "menee", "egalite": "egalite", "devant": "elle mene", "tous": "tous", "ensemble": "ensemble", "0-30": "0-30", "30-60": "30-60", "60+": "60+", "zone de recuperation": "zone de recuperation", "score de A": "score de l'equipe", "moment": "moment", "hauteur de ligne de A": "hauteur de ligne", "profonde": "profonde", "mediane": "mediane", "haute": "haute", "ligne basse": "ligne basse", "ligne haute": "ligne haute"},
    "en": {"porteur": "carrier", "zone": "zone", "score": "score", "minute": "minute",
           "libre": "free", "presse": "pressed", "milieu": "midfield", "adversaire": "opponent half",
           "derriere": "trailing", "egalite": "level", "devant": "leading", "tous": "all", "ensemble": "overall", "0-30": "0-30", "30-60": "30-60", "60+": "60+", "zone de recuperation": "recovery zone", "score de A": "team score", "moment": "time", "hauteur de ligne de A": "line height", "profonde": "deep", "mediane": "middle", "haute": "high", "ligne basse": "low line", "ligne haute": "high line"},
}

FIG = {
    "fr": dict(w1="tir concede = 1 tir cree", w2="tir concede = 2 tirs crees",
               x0="effet net d'un bloc plus haut de 1 ecart-type (points de probabilite) [IC95 %]\n> 0 : monter paie ; < 0 : monter coute",
               t0="Bloc plus haut : gain moins cout, par situation",
               x1="D = ln OR(contre-pressing) - ln OR(repli), entree dans la surface [IC95 %]\n< 0 : contre-pressing mieux ; > 0 : repli mieux ; gris : indistinct",
               t1="Contre-presser ou se replier ? (par ecart-type)"),
    "en": dict(w1="conceded shot = 1 created shot", w2="conceded shot = 2 created shots",
               x0="net effect of a line 1 SD higher (probability points) [95% CI]\n> 0: going up pays; < 0: going up costs",
               t0="Higher line: gain minus cost, by situation",
               x1="D = ln OR(counter-press) - ln OR(recovery run), box entry [95% CI]\n< 0: counter-press better; > 0: recovery better; grey: unclear",
               t1="Counter-press or recover? (per SD)"),
}

TAB = {
    "fr": dict(head=["Resultat", "Valeur [IC95 %]", "Statut"],
               rel="Fiabilite des positions", rel_v="10 joueurs observes dans {a:.0f} % des moments ; 4 plus bas observes dans {b:.0f} % ; incertitude mediane de la ligne {m:.1f} m",
               libre="Ligne haute, porteur libre (bilan, points par ecart-type)", presse="Ligne haute, porteur presse", adv="Ligne haute, balle chez l'adversaire",
               inter="Score et minute, a situation egale", inter_v="interactions : {s} (score), {m} (minute)",
               lev="Contre-pressing contre repli, un joueur deplace (OR surface)", lev_s="non concluant (deux fonctions differentes)",
               phys="Cout physique : course rapide de repli apres perte (points de % par ecart-type de ligne)",
               model="Modele predictif hors echantillon", model_v="verdict {v}",
               chk="Controles (alignement, placebo, contexte fournisseur)", chk_v="alignement : {a:.0f} % des ecarts < 0,5 m ; placebo : au plus {p:.0f} % d'intervalles excluant 0 (5 % attendu) ; valeur de possession fournisseur : {c}", verif="a verifier",
               pose="Body Pose (2 matchs)", pose_v="{n} tirs adverses ; ecart minimal detectable {m} points", pose_no="non evalue dans cette execution",
               etab="etabli", sugg="suggestif", nc="non concluant", expl="exploratoire", meas="etabli (mesure)",
               fast="> Mode rapide (4 matchs) : test de bon fonctionnement, resultats non representatifs."),
    "en": dict(head=["Result", "Value [95% CI]", "Status"],
               rel="Position reliability", rel_v="all 10 players observed in {a:.0f}% of moments; 4 deepest observed in {b:.0f}%; median line uncertainty {m:.1f} m",
               libre="High line, free carrier (net, points per SD)", presse="High line, pressed carrier", adv="High line, ball in opponent half",
               inter="Score and minute, same situation", inter_v="interactions: {s} (score), {m} (minute)",
               lev="Counter-press vs recovery run, one player swapped (box-entry OR)", lev_s="inconclusive (two different functions)",
               phys="Physical cost: sprint during recovery after a loss (% points per SD of line)",
               model="Out-of-sample predictive model", model_v="verdict {v}",
               chk="Checks (alignment, placebo, provider context)", chk_v="alignment: {a:.0f}% of gaps < 0.5 m; placebo: at most {p:.0f}% of intervals excluding 0 (5% expected); provider possession value: {c}", verif="to verify",
               pose="Body Pose (2 matches)", pose_v="{n} opponent shots; minimum detectable effect {m} points", pose_no="not evaluated in this run",
               etab="established", sugg="suggestive", nc="inconclusive", expl="exploratory", meas="established (measured)",
               fast="> Fast mode (4 matches): smoke test only, results not representative."),
}


def label(key, dim_order, lang):
    n = DIMN[lang]
    parts = [f"{n[k]} : {n[v]}" for k, v in zip(dim_order, key) if v != "tous"]
    return " / ".join(parts) if parts else n["ensemble"]
