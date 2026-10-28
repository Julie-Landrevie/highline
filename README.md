# highline — La ligne haute est un pari sur le porteur

🇫🇷 Français (ce fichier) · 🇬🇧 [English version](README_en.md) — contenu équivalent.

**Faut-il monter sa ligne quand le porteur est libre ?**

> **La ligne haute est un pari sur le porteur : elle rapporte s'il est pressé, elle coûte s'il est libre.**

Analytics Cup 2.0 (SkillCorner × PySport). Données : SkillCorner A-League 2024/25 uniquement (20 matchs de tracking, événements dynamiques, Body Pose sur 2 matchs). Aucune donnée externe.

## Question
Monter sa ligne donne des occasions mais expose le dos de la défense. Quand cela vaut-il la peine ? Le staff ne voit que des moyennes ; nous donnons, **situation par situation, le gain, le coût et le bilan**, avec l'incertitude.

## Méthode
1. **Fiabilité des positions.** Un joueur non détecté a une position extrapolée dont l'erreur croît avec la durée du trou (0,1 m à 0 s, 1,8 m à 5 s, 6 m à 30 s) ; 100 jeux de positions plausibles donnent l'incertitude de la ligne. Les 10 défenseurs de champ ne sont observés que dans 14 % des moments (36 % pour les 4 plus bas) : la ligne est la moyenne des **4 plus bas**.
2. **Ligne résiduelle.** Tous les 2 s, par équipe qui défend : ligne moins sa valeur habituelle pour la même position du ballon (ajustement quadratique par équipe et par match). Retenus : bloc installé (≥ 5 s) et balle à plus de 35 m de la **ligne de but**.
3. **Gain et coût.** Gain = tirs de l'équipe dans les 15 s ; coût = tirs adverses ; bilan(w) = gain − w·coût. Points de probabilité par écart-type de ligne, **bootstrap par match**, réplication par moitiés de matchs.
4. **Décision.** Porteur (libre ou pressé, `overall_pressure_start`) × zone × score × minute, cellules d'au moins 400 moments. Le poids w* = gain/coût indique à partir de quelle aversion au risque la ligne haute cesse de payer.
5. **Leviers.** Contre-pressing (adversaires à moins de 8 m) contre repli ; coût physique (course > 5,5 m/s dans les 8 s après une perte).
6. **Contrôles et verdicts pré-enregistrés (GO / À VOIR / NO-GO).** Tracking contre événements, tirs contre `lead_to_shot`, ligne contre celle du fournisseur, **placebo** (ligne décalée au hasard : 5 % d'intervalles excluant 0 attendus), faux positifs attendus, contrôle de la valeur de possession du fournisseur (`docs/rapport_full.txt`).

## Résultats
![Bilan par situation et leviers](docs/figure_1.png)
| Résultat (points par écart-type de ligne, IC 95 %) | Valeur | Statut |
|---|---|---|
| Porteur libre : bilan | −0,90 [−1,81 ; −0,02] | établi |
| Porteur pressé : bilan | +1,43 [+0,52 ; +2,62] | établi |
| Balle chez l'adversaire : bilan | +2,79 [+1,92 ; +3,72] | établi |
| Score, minute à situation égale | −0,18 [−0,83 ; +0,47] ; +0,65 [−0,12 ; +1,42] | non concluant |
| Course de repli rapide (points de %) | +7,6 [+5,5 ; +9,8] | établi |
| Modèle hors échantillon | NO-GO | non concluant |
| Contrôles | alignement 96 % < 0,5 m ; placebo analytique 5 % ; bootstrap 8–11 % | à lire avec prudence |
| Body Pose (2 matchs) | 86 tirs ; écart min. détectable ≈ 2 | exploratoire |

**Lecture.** La ligne haute rapporte en moyenne, mais cache deux situations opposées : contre un porteur **pressé** elle paie, contre un porteur **libre** elle coûte. Ce qui tranche est la pression sur le porteur, la zone du ballon et, plus faiblement, la minute ; le score n'est pas établi.

## Perspective exploratoire : Body Pose
Sur 2 matchs, aucun intervalle n'exclut 0 et l'écart minimal détectable dépasse les effets : aucune conclusion, quel que soit le seuil (10, 20 ou 30 cm, test post hoc). Pose valide si l'erreur p90 des épaules ≤ 20 cm ; l'orientation du buste n'est pas un regard ; sans pose valide, un joueur est exclu, jamais imputé. **Hypothèses pré-enregistrées** (`docs/HYPOTHESES_BODYPOSE.md`) : plus de défenseurs proches tournés vers leur but, ou d'attaquants face au jeu, annoncent plus de tirs adverses.

## Outil
`docs/outil_decision.html` : page autonome **bilingue FR/EN**, sans serveur. On choisit porteur, zone, score, minute et poids w ; elle affiche gain, coût, bilan, intervalle, et ce qui n'est pas établi. Une vue 3D d'un moment réel montre les défenseurs que la caméra ne voit pas (`--render3d` : image, GIF, MP4).

## Reproduire
```
pip install -e .
python reproduce.py --mode full
python reproduce.py --mode fast
pytest
```
Données téléchargées depuis le dépôt ouvert SkillCorner (non versionnées ; Body Pose ~600 Mo par match). `highline.api` (`docs/API.md`) lit la table de décision sans les données.

## Limites
Association, pas causalité (la ligne est peut-être montée parce que la situation s'y prête) ; **l'effet vient surtout des moments que le fournisseur ne juge pas en défense organisée** (restreint aux 77 % organisés, le bilan n'exclut plus 0) ; intervalles bootstrap un peu optimistes (8 à 11 % d'intervalles placebo excluant 0 pour 5 % attendus) ; positions extrapolées (l'incertitude est mesurée, pas supprimée) ; une saison, une compétition.

## Travaux connexes
Shah (édition 1, `databallpy`), Bekkers & Sahasrabudhe, Choi 2026, Visual Exploratory Behaviour, VPEP, exPress. Mêmes données : « The shape of a collective press » (Karayaz, forme du pressing sur le porteur) et « He raised his hand » (de Amorim, option de passe offensive). Nous traitons la ligne défensive et la fiabilité des positions, et nous distinguons de la phase « hors possession » de SkillCorner (18 % de nos pertes « directes » seulement tombent dans une transition du fournisseur).

## Bibliothèques et licences
numpy, pandas, scipy, matplotlib, requests (BSD / MIT) ; huggingface_hub (Apache 2.0) ; pytest (MIT). Code MIT (`LICENSE.md`).

## Déclaration d'usage d'outils d'IA
Un assistant d’IA (modèle de langage) a aidé à écrire et relire le code et les textes (français et anglais). La question, les hypothèses et les choix d’analyse ont été arrêtés par l’autrice, qui a exécuté et vérifié l’ensemble sur le jeu de données fourni (tests, `reproduce.py --mode full`) et en assume la responsabilité. Aucune donnée externe n’a été utilisée.
