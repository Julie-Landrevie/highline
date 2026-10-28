# highline — The high line is a bet on the ball carrier

🇬🇧 English (this file) · 🇫🇷 [Version française](README.md) — equivalent content.

**Should you push your line up when the ball carrier is free?**

> **The high defensive line is a bet on the ball carrier: it pays if he is pressed, it costs if he is free.**

Analytics Cup 2.0 (SkillCorner × PySport). Data: SkillCorner A-League 2024/25 only (20 tracking matches, dynamic events, Body Pose on 2 matches). No external data.

## Question
Pushing the line up creates chances but exposes the space behind. When is it worth it? Staffs see averages; we give, **situation by situation, the gain, the cost and the net balance**, with uncertainty.

## Method
1. **Position reliability.** An undetected player has an extrapolated position whose error grows with gap length (0.1 m at 0 s, 1.8 m at 5 s, 6 m at 30 s); 100 plausible position sets give the line uncertainty. All 10 outfield defenders are observed in only 14% of moments (36% for the 4 deepest): the line is the mean of the **4 deepest**.
2. **Residual line.** Every 2 s, per defending team: line minus its usual value for the same ball position (quadratic fit per team and match). Kept: settled block (≥ 5 s) and ball more than 35 m from the **goal line**.
3. **Gain and cost.** Gain = own shots within 15 s; cost = opponent shots; net(w) = gain − w·cost. Probability points per SD of line, **match-level bootstrap**, split-half replication.
4. **Decision.** Carrier (free or pressed, `overall_pressure_start`) × zone × score × minute, cells of at least 400 moments. The weight w* = gain/cost tells from which risk aversion the high line stops paying.
5. **Levers.** Counter-press (opponents within 8 m) versus recovery run; physical cost (sprinting > 5.5 m/s within 8 s after a loss).
6. **Pre-registered verdicts** (GO / MAYBE / NO-GO) in the script headers.
7. **Checks.** Tracking versus events, shots versus `lead_to_shot`, our line versus the provider's, **placebo** (randomly shifted line: 5% of intervals excluding 0 expected), expected false positives, control for the provider's possession value (`docs/rapport_full.txt`).

## Results
![Net balance by situation and levers](docs/figure_1_en.png)
*(French version: `docs/figure_1.png`)*

| Result (points per SD of line, 95% CI) | Value | Status |
|---|---|---|
| Free carrier: net | −0.90 [−1.81; −0.02] | established |
| Pressed carrier: net | +1.43 [+0.52; +2.62] | established |
| Ball in opponent half: net | +2.79 [+1.92; +3.72] | established |
| Score, minute at equal situation | −0.18 [−0.83; +0.47]; +0.65 [−0.12; +1.42] | inconclusive |
| Fast recovery run (% points) | +7.6 [+5.5; +9.8] | established |
| Out-of-sample model | NO-GO | inconclusive |
| Checks | alignment 96% < 0.5 m; analytic placebo 5%; bootstrap 8–11% | read with caution |
| Body Pose (2 matches) | 86 shots; min. detectable ≈ 2 | exploratory |

**Reading.** The high line pays on average but hides two opposite situations: against a **pressed** carrier it pays, against a **free** carrier it costs. What decides is pressure on the carrier, ball zone and, more weakly, match minute; score is not established.

## Exploratory perspective: Body Pose
With 2 matches no interval excludes 0 and the minimum detectable effect exceeds the effects: no conclusion, whatever the threshold (10, 20 or 30 cm, post hoc). Valid pose if shoulder p90 error ≤ 20 cm; trunk orientation is not a gaze; a player without valid pose is excluded, never imputed. **Pre-registered hypotheses** (`docs/HYPOTHESES_BODYPOSE.md`): more near-ball defenders turned toward their own goal, or attackers facing forward, precede more opponent shots.

## Tool
`docs/outil_decision.html`: standalone page, **bilingual FR/EN**, no server. Pick carrier, zone, score, minute and weight w; it shows gain, cost, net, interval, and what is not established. A 3D view of a real moment shows the defenders the camera cannot see (`--render3d`: image, GIF, MP4).

## Reproduce
```
pip install -e .
python reproduce.py --mode full     # 20 matches, results above
python reproduce.py --mode fast     # 4 matches, smoke test
pytest                              # synthetic scenes, no data needed
```
Data are downloaded from SkillCorner's open repository (not versioned); Body Pose included by default (~600 MB per match, cached). `highline.api` (`docs/API.md`) reads the decision table without the data.

## Limits
Association, not causation (the line may go up because the situation allows it); **the effect comes mostly from moments the provider does not label organised defence** (restricted to the 77% organised, the net no longer excludes 0); bootstrap intervals slightly optimistic (8–11% of placebo intervals exclude 0 vs 5% expected); extrapolated positions (uncertainty is measured, not removed); one season, one competition; counter-press vs recovery: only the one-player swap is comparable.

## Related work
Shah (edition 1, `databallpy`), Bekkers & Sahasrabudhe, Choi 2026, Visual Exploratory Behaviour, VPEP, exPress. Same data: “The shape of a collective press” (Karayaz, press shape around the carrier) and “He raised his hand” (de Amorim, an attacking pass option). We address the defensive line and position reliability, and differ from SkillCorner's out-of-possession phases (only 18% of our “direct” losses fall in a provider transition phase).

## Libraries and licences
numpy, pandas, scipy, matplotlib, requests (BSD / MIT); huggingface_hub (Apache 2.0); pytest (MIT). Code under MIT licence (`LICENSE.md`).

## AI-tool use declaration
An AI assistant (language model) helped write and review the code and the texts (French and English). The question, hypotheses and analysis choices were decided by the author, who ran and checked everything on the provided dataset (tests, `reproduce.py --mode full`) and takes responsibility for it. No external data was used.
