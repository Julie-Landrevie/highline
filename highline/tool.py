"""Outil de decision : UNE page HTML autonome (aucun serveur), generee a l'execution a partir des resultats calcules. Aucune donnee brute n'est incluse."""
import json

import numpy as np

from . import decision as dec

TEMPLATE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>highline</title>
<style>
:root{--bg:#f7f7f4;--fg:#1d2330;--mut:#5d6677;--card:#fff;--line:#d9dce3;--ok:#1b7f5c;--ko:#b3402f;--mid:#8a6d1c;--acc:#2a5bd7}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#12151c;--fg:#e8eaf0;--mut:#9aa3b5;--card:#1b202b;--line:#2e3647;--ok:#4cc29a;--ko:#ef7f6d;--mid:#d9b653;--acc:#7aa2ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:1.6rem;margin:0 0 4px}h2{font-size:1.1rem;margin:28px 0 8px}
.sub{color:var(--mut);margin:0 0 20px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}label{display:block;font-size:.85rem;color:var(--mut)}
select,input[type=range]{width:100%;font:inherit;margin-top:4px}select{padding:6px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:12px}.kpi div{border:1px solid var(--line);border-radius:10px;padding:10px}
.kpi b{display:block;font-size:1.3rem}.kpi span{color:var(--mut);font-size:.8rem}
.tag{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.8rem;border:1px solid var(--line)}
.paie{color:var(--ok)}.coute{color:var(--ko)}.nt{color:var(--mid)}
table{border-collapse:collapse;width:100%;font-size:.88rem}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
#lang{position:absolute;top:12px;right:16px;display:flex;gap:4px}#lang button{font:inherit;font-size:.8rem;padding:3px 10px;border-radius:99px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}#lang button[aria-pressed=true]{background:var(--acc);color:#fff;border-color:var(--acc)}
small,.note{color:var(--mut);font-size:.82rem}svg{width:100%;height:64px}
</style></head><body><main>
<div id="lang"><button id="b_fr" aria-pressed="true">FR</button><button id="b_en" aria-pressed="false">EN</button></div>
<div id="fast" class="card" style="display:none;border-color:var(--ko)"></div>
<h1 id="h1"></h1>
<p class="sub" id="sub"></p>
<div class="card"><h2 style="margin-top:0" id="t1"></h2>
<div class="grid">
<div><label><span id="l_porteur"></span><select id="s_porteur"></select></label></div>
<div><label><span id="l_zone"></span><select id="s_zone"></select></label></div>
<div><label><span id="l_score"></span><select id="s_score"></select></label></div>
<div><label><span id="l_minute"></span><select id="s_minute"></select></label></div>
<div><label><span id="l_w"></span> : <b id="wv">1</b><input id="w" type="range" min="0.5" max="5" step="0.1" value="1"></label></div>
</div>
<div id="res"></div></div>
<div class="card"><h2 style="margin-top:0" id="t2"></h2><div id="lev"></div></div>
<div class="card"><h2 style="margin-top:0" id="t3"></h2><div id="phys"></div></div>
<div class="card"><h2 style="margin-top:0" id="t4"></h2><div id="rel"></div></div>
<div class="card"><h2 style="margin-top:0" id="t5"></h2><div id="pose"></div></div>
<p class="note" id="limits"></p>
</main>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);
let LANG = (()=>{try{const s=localStorage.getItem("hl_lang");if(s)return s}catch(e){}return (navigator.language||"fr").toLowerCase().startsWith("fr")?"fr":"en"})();
const I = {
fr:{h1:"La ligne haute est un pari sur le porteur",sub:"Elle rapporte s'il est pressé, elle coûte s'il est libre. Association observée sur 20 matchs de l'A-League 2024/25 (tracking SkillCorner) ; chaque chiffre porte son intervalle et son niveau de preuve.",
 fast:"<b>Mode rapide (4 matchs)</b> : test de bon fonctionnement, résultats non représentatifs. Lancer <code>python reproduce.py --mode full</code>.",
 t1:"1. Décrire la situation",t2:"2. Après une perte : contre-presser ou se replier ?",t3:"3. Le coût en course de repli",t4:"4. Peut-on croire les positions ?",t5:"5. Body Pose (exploratoire)",
 l_porteur:"Porteur",l_zone:"Zone de balle",l_score:"Score de l'équipe qui défend",l_minute:"Moment du match (min)",l_w:"Poids d'un tir concédé (en tirs créés)",
 limits:"Limites : association et non causalité ; 20 matchs (3 à 4 par équipe) ; hauteur de ligne = moyenne des 4 défenseurs de champ les plus bas ; positions extrapolées modélisées ; un tir concédé peut valoir plus ou moins qu'un tir créé (d'où le curseur). Prototype non validé auprès d'un staff.",
 names:{porteur:{tous:"tous",libre:"libre",presse:"pressé"},zone:{tous:"toutes",milieu:"milieu (35-70 m)",adversaire:"chez l'adversaire"},score:{tous:"tous",derriere:"menée",egalite:"égalité",devant:"elle mène"},minute:{tous:"tous","0-30":"0-30","30-60":"30-60","60+":"60+"}},
 dimn:{porteur:"porteur",zone:"zone",score:"score",minute:"minute",ensemble:"ensemble","zone de recuperation":"zone de récupération","score de A":"score de l'équipe",moment:"moment","hauteur de ligne de A":"hauteur de ligne"},
 few:m=>"Cette combinaison compte moins de "+m+" moments : choisissez moins de filtres.",
 up:"monter paie",down:"monter coûte",open:"pas tranchable",
 pe:"établi (la différence entre situations est confirmée)",ps:"suggestif (à confirmer)",pn:"non tranché",proof:"preuve",
 k_n:"moments observés",k_cost:"coût : tirs concédés",k_gain:"gain : tirs créés",k_net:"bilan à poids",k_w:"poids à partir duquel rester bas est préférable",
 unit:(sd)=>"Points de probabilité de tir en 15 s par écart-type de hauteur de ligne (≈ "+sd+" m). Bilan = gain − poids × coût.",
 nc:"Non calculé.",
 lev1:(a,b,c,same)=>"<p>Le contre-pressing sert à <b>reprendre</b> la balle ; le repli <b>limite les dégâts</b> si on ne la reprend pas. Par joueur déplacé de « derrière la balle » à « près de la balle », l'entrée adverse dans la surface change de ×"+a+" ["+b+" ; "+c+"] : "+(same?"les deux usages ne se distinguent pas":"un usage domine")+".</p>",
 lev_h:["Situation","D = ln OR(contre-pressing) − ln OR(repli)","Lecture"],lev_p:{pressing:"contre-pressing mieux",repli:"repli mieux",none:"indistinct"},
 lev_n:"D < 0 : le contre-pressing réduit plus l'entrée dans la surface. Calculé par écart-type ; la lecture par joueur ci-dessus est la plus parlante.",
 phys:(sd,e,lo,hi,m)=>"<p>Après une perte, une ligne plus haute d'un écart-type (≈ "+sd+" m) ajoute <b>"+e+" points</b> de temps passé en course rapide (&gt; 5,5 m/s) aux 4 défenseurs les plus bas, dans les 8 s ["+lo+" ; "+hi+"] (moyenne "+m+" %). Coût d'effort local, pas une charge de match.</p>",
 rel:(a,b,m,p)=>"<p>Dans seulement <b>"+a+" %</b> des moments, les 10 joueurs de champ sont observés ; les 4 défenseurs les plus bas le sont tous dans <b>"+b+" %</b>. L'incertitude médiane de la hauteur de ligne est de <b>"+m+" m</b> (p90 : "+p+" m).</p><p class='note'>Les positions extrapolées sont simulées (100 tirages) à partir de l'erreur typique selon la durée du trou ; le tableau ci-dessus s'appuie sur les positions du fournisseur.</p>",
 pose0:"<p>Hypothèses enregistrées avant toute donnée supplémentaire : <b>H1</b> plus de joueurs proches de la balle tournés vers leur but, plus de tirs adverses ; <b>H2</b> plus d'adversaires proches face à l'avant, plus de tirs adverses.</p>",
 pose_h:["Variable","Effet sur le tir adverse (points)","Écart minimal détectable"],pose_v:{turn:"proches tournés vers leur but",fwd:"adversaires face à l'avant"},
 pose_n:(n,s)=>"<p class='note'>"+n+" moments, "+s+" tirs adverses, 2 matchs : aucune conclusion possible, signal exploratoire.</p>",pose_no:"<p class='note'>Non évalué dans cette exécution (installer l'extra pose).</p>"},
en:{h1:"The high line is a bet on the ball carrier",sub:"It pays if he is pressed, it costs if he is free. Association observed over 20 A-League 2024/25 matches (SkillCorner tracking); every figure carries its interval and its level of evidence.",
 fast:"<b>Fast mode (4 matches)</b>: smoke test only, results not representative. Run <code>python reproduce.py --mode full</code>.",
 t1:"1. Describe the situation",t2:"2. After a loss: counter-press or recover?",t3:"3. The running cost of recovering",t4:"4. Can we trust the positions?",t5:"5. Body Pose (exploratory)",
 l_porteur:"Ball carrier",l_zone:"Ball zone",l_score:"Defending team's score",l_minute:"Match time (min)",l_w:"Weight of a conceded shot (in created shots)",
 limits:"Limits: association, not causation; 20 matches (3 to 4 per team); line height = mean of the 4 deepest outfield defenders; extrapolated positions are modelled; a conceded shot may be worth more or less than a created one (hence the slider). Prototype not validated with a coaching staff.",
 names:{porteur:{tous:"all",libre:"free",presse:"pressed"},zone:{tous:"all",milieu:"midfield (35-70 m)",adversaire:"opponent half"},score:{tous:"all",derriere:"trailing",egalite:"level",devant:"leading"},minute:{tous:"all","0-30":"0-30","30-60":"30-60","60+":"60+"}},
 dimn:{porteur:"carrier",zone:"zone",score:"score",minute:"minute",ensemble:"overall","zone de recuperation":"recovery zone","score de A":"team score",moment:"time","hauteur de ligne de A":"line height"},
 few:m=>"This combination has fewer than "+m+" moments: use fewer filters.",
 up:"going up pays",down:"going up costs",open:"cannot tell",
 pe:"established (the difference between situations is confirmed)",ps:"suggestive (to be confirmed)",pn:"undecided",proof:"evidence",
 k_n:"observed moments",k_cost:"cost: shots conceded",k_gain:"gain: shots created",k_net:"net at weight",k_w:"weight above which staying deep is preferable",
 unit:(sd)=>"Shot-probability points within 15 s per SD of line height (≈ "+sd+" m). Net = gain − weight × cost.",
 nc:"Not computed.",
 lev1:(a,b,c,same)=>"<p>Counter-pressing is for <b>winning the ball back</b>; recovering <b>limits the damage</b> if you don't. For each player moved from “behind the ball” to “near the ball”, the opponent's box-entry odds change by ×"+a+" ["+b+" ; "+c+"]: "+(same?"the two uses cannot be told apart":"one use dominates")+".</p>",
 lev_h:["Situation","D = ln OR(counter-press) − ln OR(recovery)","Reading"],lev_p:{pressing:"counter-press better",repli:"recovery better",none:"unclear"},
 lev_n:"D < 0: counter-pressing reduces box entry more. Computed per SD; the per-player reading above is the most telling.",
 phys:(sd,e,lo,hi,m)=>"<p>After a loss, a line one SD higher (≈ "+sd+" m) adds <b>"+e+" points</b> of time spent sprinting (&gt; 5.5 m/s) for the 4 deepest defenders within 8 s ["+lo+" ; "+hi+"] (mean "+m+"%). A local effort cost, not a match load.</p>",
 rel:(a,b,m,p)=>"<p>In only <b>"+a+"%</b> of moments are all 10 outfield players observed; the 4 deepest defenders are all observed in <b>"+b+"%</b>. Median line-height uncertainty is <b>"+m+" m</b> (p90: "+p+" m).</p><p class='note'>Extrapolated positions are simulated (100 draws) from the typical error given gap length; the table above uses the provider's positions.</p>",
 pose0:"<p>Hypotheses registered before any additional data: <b>H1</b> more near-ball players turned toward their own goal, more opponent shots; <b>H2</b> more near-ball attackers facing forward, more opponent shots.</p>",
 pose_h:["Variable","Effect on opponent shot (points)","Minimum detectable effect"],pose_v:{turn:"near-ball players turned toward own goal",fwd:"attackers facing forward"},
 pose_n:(n,s)=>"<p class='note'>"+n+" moments, "+s+" opponent shots, 2 matches: no conclusion possible, exploratory signal.</p>",pose_no:"<p class='note'>Not evaluated in this run (install the pose extra).</p>"}};
const L = () => I[LANG];
const fmt = (x,d=2) => { const t=(x>=0?"+":"")+x.toFixed(d); return LANG==="fr"?t.replace(".",","):t };
const fmtn = (x,d=1) => { const t=x.toFixed(d); return LANG==="fr"?t.replace(".",","):t };
const DIMS = ["porteur","zone","score","minute"];
function fillSelects(){for(const d of DIMS){const s=$("s_"+d);const cur=s.value||"tous";s.innerHTML="";for(const v of ["tous"].concat(DATA.dims[d])){const o=document.createElement("option");o.value=v;o.textContent=L().names[d][v]||v;s.appendChild(o)};s.value=cur;s.onchange=render}}
$("w").oninput=render;
function pct(a,p){const s=a.slice().sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.max(0,Math.floor(p*s.length)))]}
function find(key){return DATA.cells.find(c=>c.key.every((k,i)=>k===key[i]))}
function render(){
  const l=L(),w=parseFloat($("w").value);$("wv").textContent=fmtn(w);
  const key=DIMS.map(d=>$("s_"+d).value);const c=find(key);const out=$("res");
  if(!c){out.innerHTML="<p class='note'>"+l.few(DATA.meta.min_cell)+"</p>";return}
  const nets=c.draws.map(d=>d[1]-w*d[0]);const net=c.gain-w*c.cost;const lo=pct(nets,.025),hi=pct(nets,.975);
  const cl=pct(c.draws.map(d=>d[0]),.025),ch=pct(c.draws.map(d=>d[0]),.975),gl=pct(c.draws.map(d=>d[1]),.025),gh=pct(c.draws.map(d=>d[1]),.975);
  const tag=lo>0?[l.up,"paie"]:(hi<0?[l.down,"coute"]:[l.open,"nt"]);
  const sel=key.map((k,i)=>[DIMS[i],k]).filter(x=>x[1]!=="tous");
  const est=sel.length>0&&DATA.diffs.some(d=>d.ok&&sel.some(s=>s[0]===d.dim));
  const proof=lo>0||hi<0?(est?l.pe:l.ps):l.pn;
  const wstar=(c.cost>0&&c.gain>0)?fmtn(c.gain/c.cost):"-";
  const lim=Math.max(2,Math.abs(lo),Math.abs(hi),Math.abs(net))*1.1,X=v=>50+(v/lim)*45;
  out.innerHTML="<div class='kpi'><div><b>"+c.n+"</b><span>"+l.k_n+"</span></div><div><b>"+fmt(c.cost)+"</b><span>"+l.k_cost+" ["+fmt(cl)+" ; "+fmt(ch)+"]</span></div><div><b>"+fmt(c.gain)+"</b><span>"+l.k_gain+" ["+fmt(gl)+" ; "+fmt(gh)+"]</span></div><div><b class='"+tag[1]+"'>"+fmt(net)+"</b><span>"+l.k_net+" "+fmtn(w)+" ["+fmt(lo)+" ; "+fmt(hi)+"]</span></div><div><b>"+wstar+"</b><span>"+l.k_w+"</span></div></div>"+
  "<svg viewBox='0 0 100 20' preserveAspectRatio='none'><line x1='5' x2='95' y1='12' y2='12' stroke='currentColor' opacity='.25'/><line x1='50' x2='50' y1='4' y2='16' stroke='currentColor' opacity='.5'/><line x1='"+X(lo)+"' x2='"+X(hi)+"' y1='12' y2='12' stroke='currentColor' stroke-width='2'/><circle cx='"+X(net)+"' cy='12' r='2'/></svg>"+
  "<p><span class='tag "+tag[1]+"'>"+tag[0]+"</span> <span class='tag'>"+l.proof+" : "+proof+"</span></p><p class='note'>"+l.unit(fmtn(DATA.meta.sd_line))+"</p>";
}
const XL={fr:{profonde:"profonde",mediane:"médiane",haute:"haute","ligne basse":"ligne basse","ligne haute":"ligne haute",derriere:"menée",egalite:"égalité",devant:"elle mène"},en:{profonde:"deep",mediane:"middle",haute:"high","ligne basse":"low line","ligne haute":"high line",derriere:"trailing",egalite:"level",devant:"leading"}};
function slabel(k){return k.length===0?L().dimn.ensemble:(L().dimn[k[0]]||k[0])+" : "+(XL[LANG][k[1]]||k[1])}
function levers(){const l=L(),V=DATA.levers;if(!V){$("lev").innerHTML="<p class='note'>"+l.nc+"</p>";return}
  let h=l.lev1(fmtn(V.swap[0],2),fmtn(V.swap[1],2),fmtn(V.swap[2],2),V.swap[1]<=1&&V.swap[2]>=1)+"<table><tr><th>"+l.lev_h.join("</th><th>")+"</th></tr>";
  for(const s of V.strata){h+="<tr><td>"+slabel(s.key)+"</td><td>"+fmt(s.D)+" ["+fmt(s.lo)+" ; "+fmt(s.hi)+"]</td><td>"+(l.lev_p[s.pref]||l.lev_p.none)+"</td></tr>"}
  $("lev").innerHTML=h+"</table><p class='note'>"+l.lev_n+"</p>"}
function phys(){const P=DATA.physical,l=L();$("phys").innerHTML=P?l.phys(fmtn(P.sd_line),fmt(P.full[0],1),fmt(P.full[1],1),fmt(P.full[2],1),fmtn(P.mean)):"<p class='note'>"+l.nc+"</p>"}
function rel(){const R=DATA.reliability;$("rel").innerHTML=L().rel(fmtn(R.all10*100,0),fmtn(R.all4*100,0),fmtn(R.sd_med,1),fmtn(R.sd_p90,1))}
function pose(){const P=DATA.pose,l=L();let h=l.pose0;
  if(P){h+="<table><tr><th>"+l.pose_h.join("</th><th>")+"</th></tr>"+P.rows.filter(r=>r.y==="opp15").map(r=>"<tr><td>"+l.pose_v[r.var]+"</td><td>"+fmt(r.eff)+" ["+fmt(r.lo)+" ; "+fmt(r.hi)+"]</td><td>"+fmtn(r.mde)+"</td></tr>").join("")+"</table>"+l.pose_n(P.n,P.shots)}else{h+=l.pose_no}
  $("pose").innerHTML=h}
function paint(){const l=L();document.documentElement.lang=LANG;
  for(const id of ["h1","t1","t2","t3","t4","t5","limits"])$(id).textContent=l[id==="h1"?"h1":id];
  $("sub").textContent=l.sub;for(const d of DIMS)$("l_"+d).textContent=l["l_"+d];$("l_w").textContent=l.l_w;
  $("fast").innerHTML=l.fast;$("fast").style.display=DATA.meta.fast?"block":"none";
  $("b_fr").setAttribute("aria-pressed",LANG==="fr");$("b_en").setAttribute("aria-pressed",LANG==="en");
  fillSelects();levers();phys();rel();pose();render()}
function setLang(x){LANG=x;try{localStorage.setItem("hl_lang",x)}catch(e){};paint()}
$("b_fr").onclick=()=>setLang("fr");$("b_en").onclick=()=>setLang("en");
paint();
</script></body></html>"""


def build(cell_list, diffs, lev, phys, rel, pose, path, sd_line, thin=100, min_cell=400, fast=False):
    """Ecrit la page. Les tirages bootstrap sont amincis pour limiter la taille (permet de recalculer l'IC pour tout poids)."""
    rng = np.random.default_rng(1)
    cells_js = []
    for c in cell_list:
        dr = c["draws"][~np.isnan(c["draws"]).any(axis=1)]
        if len(dr) > thin:
            dr = dr[rng.choice(len(dr), thin, replace=False)]
        cells_js.append(dict(key=list(c["key"]), n=c["n"], cost=round(float(c["point"][0]), 3), gain=round(float(c["point"][1]), 3),
                             draws=[[round(float(a), 3), round(float(b), 3)] for a, b in dr]))
    lev_js = None
    if lev:
        from .stats import pct
        st = []
        for k in lev["strata"]:
            p = lev["point"].get(k)
            if p is None:
                continue
            lo, hi = pct(lev["boots"][k])
            st.append(dict(key=[] if k == "ensemble" else list(k), D=round(p["D"], 3), lo=round(lo, 3), hi=round(hi, 3), pref=lev["pref"][k]))
        lev_js = dict(strata=st, swap=[round(x, 3) for x in lev["swap"]])
    diffs_js = [dict(dim=x["dim"], ok=bool(x.get("ok"))) for x in diffs]
    data = dict(dims=dec.LEVELS, cells=cells_js, diffs=diffs_js, levers=lev_js,
                physical=None if not phys else dict(full=[round(float(v), 3) for v in phys["full"]], mean=round(phys["mean"], 3), sd_line=round(phys["sd_line"], 2)),
                reliability=rel, pose=pose, meta=dict(sd_line=round(float(sd_line), 2), min_cell=min_cell, fast=bool(fast)))
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=float))
    open(path, "w", encoding="utf-8").write(html)
    return path
