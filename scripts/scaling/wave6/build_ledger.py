"""Build the 'Motion Token Ledger' dashboard (wave 6, 2026-09-02).

Reads: viper_mirror/crosseval/*.json (wave-6 arms + controls), local eval JSONs,
videos from experiments/local_w6/media (fallback experiments/local_3t/media_final).
Writes: experiments/local_w6/token_ledger.html
"""
import base64
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[3]
CE = R / "viper_mirror/crosseval"
MEDIA_NEW = R / "experiments/local_w6/media"
MEDIA_OLD = R / "experiments/local_3t/media_final"
OUT = R / "experiments/local_w6/token_ledger.html"
BUDGET = 13.5e6  # bytes of embedded video

# ------------------------------------------------------------------ data
def agg(patterns):
    """{(arm, cond, robot): dict(rmse, legs, arms, air, n, seeds)}"""
    out = defaultdict(list)
    for pat, cond_rx in patterns:
        for f in glob.glob(str(CE / pat)):
            b = os.path.basename(f)[:-5]
            m = re.match(r"(.+?)_s(\d)_" + cond_rx + r"$", b)
            if not m:
                continue
            d = json.load(open(f))
            cond = m.group(3) if m.lastindex >= 3 else "h1"
            for rb, r in d["robots"].items():
                pj = r["per_joint_rmse_rad"]
                legs = [v for j, v in pj.items() if re.search("hip|knee|ankle", j, re.I)]
                arms = [v for j, v in pj.items() if re.search("arm|elbow|shoulder|wrist", j, re.I)]
                out[(m.group(1), cond, rb[-2:])].append(
                    (r["raw_rmse_rad"], st.mean(legs), st.mean(arms), r["foot_metrics"]["foot_airborne"], int(m.group(2))))
    res = {}
    for k, v in out.items():
        res[k] = dict(rmse=st.mean(x[0] for x in v), legs=st.mean(x[1] for x in v), arms=st.mean(x[2] for x in v),
                      air=st.mean(x[3] for x in v), n=len(v), seeds=len(set(x[4] for x in v)))
    return res

DATA = agg([
    ("n5v2_tok_s*_s?.json", r"(s\d)"),          # cond captured as s0.. -> treat as h1
    ("n5sw05_ref_s*_s?.json", r"(s\d)"),
    ("n6*_s?_h1_s?.json", r"(h1)_s\d"),
    ("n6*_s?_h20_s?.json", r"(h20)_s\d"),
    ("n6*_s?_zsh1_*.json", r"(zsh1)_s?\d"),
])
# normalise the control conds
D = {}
for (arm, cond, rb), v in DATA.items():
    if cond.startswith("s") and cond != "s":
        cond = "h1"
    key = (arm, cond, rb)
    if key in D:  # merge
        n = D[key]["n"] + v["n"]
        D[key] = {kk: (D[key][kk] * D[key]["n"] + v[kk] * v["n"]) / n if kk in ("rmse", "legs", "arms", "air") else D[key][kk] for kk in D[key]}
        D[key]["n"] = n; D[key]["seeds"] = max(D[key]["seeds"], v["seeds"])
    else:
        D[key] = dict(v)

def cell(arm, cond, rb, key="rmse"):
    v = D.get((arm, cond, rb))
    return v[key] if v else None

def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"

def pct(a, b):
    if a is None or b is None:
        return "—"
    return f"{100 * (a - b) / b:+.0f} %"

def seeds(arm, cond="h1"):
    v = D.get((arm, cond, "g1"))
    return f"{v['seeds']}×{v['n'] // max(v['seeds'], 1)}" if v else "—"

REF = "n5sw05_ref"; TOK = "n5v2_tok"

HOLD1_ROWS = [  # (label, arm, note)
    ("Reference only (control)", REF, "fresh reference every step, no token"),
    ("Precomputed token, shared routing (every prior token arm)", TOK, "design B, divisor 10"),
    ("Precomputed token, separate projection (first real split)", "n6split_tok", "JLAT_CH = −1"),
    ("Split + auxiliary next-token head", "n6aux_tok", "coeff 0.5, horizon 5"),
    ("Co-trained encoder (SONIC-style), tokenizer init", "n6cot_tok", "encoder + FSQ inside the policy, PPO gradients reach it; 4 seeds"),
    ("Co-trained encoder, from scratch", "n6cotsc_tok", "no tokenizer init"),
    ("Co-trained encoder, frozen", "n6cotfr_tok", "online window through a frozen encoder"),
    ("Co-trained + aux head", "n6cotaux_tok", ""),
    ("Reference only, leg-weighted kernel ×5", "n6legw5_ref", "leg joints ×5 in the tracking error"),
    ("Token, leg-weighted kernel ×5", "n6legw5_tok", ""),
]
REPLACE_ROWS = [
    ("Reference only, 2× budget", "n6ref2x"), ("Token only (reference removed), 2× budget", "n6rep2x_tok"),
    ("Reference only, 4× budget (1 seed)", "n6ref4x"), ("Token only, 4× budget", "n6rep4x_tok"),
]
MORPH_ROWS = [
    ("Nominal body only", "n6m0_ref", "n6m0_tok"),
    ("Default schedule (0.2 → 0.44)", REF, TOK),
    ("Ramp to 0.7 by 15M steps", "n6m7_ref", "n6m7_tok"),
]
COT6C = [("Co-trained, all four seeds", "n6cot_tok"), ("Co-trained, recon weight 0.1", "n6cot01_tok"),
         ("Co-trained, reference removed", "n6cotrep_tok"), ("Co-trained + leg kernel ×5", "n6cotlegw5_tok"),
         ("Co-trained, nominal body", "n6cotm0_tok"), ("Co-trained, ramp to 0.7", "n6cotm7_tok"),
         ("Co-trained on 5 dances (2×)", "n6cotsup_tok")]
H20_3WAY = [("Reference only", "n6h20_ref"), ("Precomputed token", "n6h20_tok"), ("Co-trained encoder", "n6coth20_tok")]

# ------------------------------------------------------------------ videos
VIDEOS = [  # (file, title, caption)
    ("cot_dance4_g1.mp4", "G1 · co-trained encoder · dance2_subject4 · fresh reference",
     "The co-trained encoder arm, seed 2 (0.145 rad; four-seed mean 0.155 vs 0.163 reference-only). 24 s rollout. The encoder that produces the motion code runs inside the policy and was fine-tuned by PPO."),
    ("ref_dance4_g1.mp4", "G1 · reference only · dance2_subject4 · fresh reference",
     "The control the co-trained arm is compared against: same recipe, same clip, no token. Watch the legs: both policies keep the feet low and slide."),
    ("cot_dance4_h1.mp4", "H1 · co-trained encoder · dance2_subject4 · fresh reference",
     "Same policy on the other body (one network drives both). This seed 0.173 rad, four-seed mean 0.184, vs 0.195 reference-only."),
    ("cot_walk1_zeroshot_g1.mp4", "G1 · co-trained encoder · walk1, never seen · fresh reference",
     "Zero-shot: neither the policy nor the tokenizer saw this clip. Four-seed mean 0.149 rad vs 0.154 reference-only at the same rate."),
    ("cot_walk1_zeroshot_h1.mp4", "H1 · co-trained encoder · walk1, never seen",
     "The unseen walk on H1. Heading is observed and rewarded, so the robot turns with the clip."),
    ("super5_h20_tok_g1.mp4", "G1 · precomputed token · five dances in one clip · reference held 20 steps",
     "The multi-motion arm trained at a stale reference: 0.215 rad against 0.337 for reference-only. 36 s covering more than one dance."),
    ("super5_h20_ref_g1.mp4", "G1 · reference only · five dances · reference held 20 steps",
     "The reference-only control at the same staleness. This is what a held reference does to tracking without a token."),
    ("cot_dance4_h20_h1.mp4", "H1 · co-trained encoder · dance2_subject4 · reference held 20 steps",
     "The hold-1-trained co-trained arm evaluated stale (0.359 rad). Trained at hold 20 it reaches 0.267 on H1, still behind the precomputed token's 0.216: for staleness the frozen sidecar is the better tool."),
    ("3robot_tok_t1.mp4", "BoosterT1 · three-topology single policy · token",
     "One network drives H1, G1 and T1. T1's arms track (r ≈ 0.96) while its legs sit near the zero-action floor, the pattern every policy shares."),
    ("transfer_g1_from_h1_stream.mp4", "G1 driven by H1's token stream · reference held 20 steps",
     "Cross-robot transfer: G1's only fresh motion signal is the per-joint code emitted from the H1 retarget, re-mapped by joint semantics. 83 % of own-token value recovered."),
]
FALLBACK = {  # old media if new ones are not rendered yet
    "cot_dance4_g1.mp4": None, "ref_dance4_g1.mp4": "superclip_ref_g1.mp4", "cot_dance4_h1.mp4": None,
    "cot_walk1_zeroshot_g1.mp4": None, "cot_walk1_zeroshot_h1.mp4": None,
    "super5_h20_tok_g1.mp4": "superclip_tok_g1.mp4", "super5_h20_ref_g1.mp4": "superclip_ref_g1.mp4",
    "cot_dance4_h20_h1.mp4": None, "3robot_tok_t1.mp4": "ln5_3robot_tok_t1.mp4",
    "transfer_g1_from_h1_stream.mp4": "FLAGSHIP_g1_from_h1_stream.mp4",
}

def video_cards():
    cards, total, seen = [], 0, set()
    for f, title, cap in VIDEOS:
        p = MEDIA_NEW / f
        note = ""
        if not p.exists():
            fb = FALLBACK.get(f)
            if not fb or not (MEDIA_OLD / fb).exists():
                cards.append(f'<figure class="vid pending"><div class="ph">rendering…</div><figcaption><b>{title}</b><p>{cap}</p><p class="mono">render pending — the local GPU is finishing a training arm first</p></figcaption></figure>')
                continue
            p = MEDIA_OLD / fb
            note = f'<p class="mono">interim: earlier {fb} until the longer render lands</p>'
        if p.name in seen:
            continue
        seen.add(p.name)
        data = p.read_bytes()
        if total + len(data) > BUDGET:
            cards.append(f'<figure class="vid pending"><div class="ph">size budget</div><figcaption><b>{title}</b><p>{cap}</p></figcaption></figure>')
            continue
        total += len(data)
        b64 = base64.b64encode(data).decode()
        cards.append(f'<figure class="vid"><video controls muted loop playsinline preload="metadata" src="data:video/mp4;base64,{b64}"></video><figcaption><b>{title}</b><p>{cap}</p>{note}</figcaption></figure>')
    return "\n".join(cards), total

# ------------------------------------------------------------------ html
def row_hold1(label, arm, note):
    g, h = cell(arm, "h1", "g1"), cell(arm, "h1", "h1")
    rg, rh = cell(REF, "h1", "g1"), cell(REF, "h1", "h1")
    lg, lh = cell(arm, "h1", "g1", "legs"), cell(arm, "h1", "h1", "legs")
    cls = ' class="best"' if arm == "n6cot_tok" else ""
    return (f"<tr{cls}><td>{label}<span class='note'>{note}</span></td><td class='num'>{fmt(g)}</td><td class='num'>{fmt(h)}</td>"
            f"<td class='num'>{pct(g, rg)} / {pct(h, rh)}</td><td class='num'>{fmt(lg)} / {fmt(lh)}</td><td class='num mono'>{seeds(arm)}</td></tr>")

def row2(label, arm, cond="h1"):
    g, h = cell(arm, cond, "g1"), cell(arm, cond, "h1")
    return f"<tr><td>{label}</td><td class='num'>{fmt(g)}</td><td class='num'>{fmt(h)}</td><td class='num mono'>{seeds(arm, cond)}</td></tr>"

hold1_rows = "\n".join(row_hold1(*r) for r in HOLD1_ROWS)
replace_rows = "\n".join(row2(l, a) for l, a in REPLACE_ROWS)
morph_rows = "\n".join(
    f"<tr><td>{l}</td><td class='num'>{fmt(cell(r,'h1','g1'))} / {fmt(cell(r,'h1','h1'))}</td><td class='num'>{fmt(cell(t,'h1','g1'))} / {fmt(cell(t,'h1','h1'))}</td>"
    f"<td class='num'>{pct(cell(t,'h1','g1'),cell(r,'h1','g1'))} / {pct(cell(t,'h1','h1'),cell(r,'h1','h1'))}</td>"
    f"<td class='num'>{pct(cell(t,'h20','g1'),cell(r,'h20','g1'))} / {pct(cell(t,'h20','h1'),cell(r,'h20','h1'))}</td></tr>"
    for l, r, t in MORPH_ROWS)
cot6c_rows = "\n".join(row2(l, a) for l, a in COT6C)
h20_rows = "\n".join(
    f"<tr><td>{l}</td><td class='num'>{fmt(cell(a,'h20','g1'))} / {fmt(cell(a,'h20','h1'))}</td><td class='num'>{fmt(cell(a,'h1','g1'))} / {fmt(cell(a,'h1','h1'))}</td><td class='num mono'>{seeds(a,'h20')}</td></tr>"
    for l, a in H20_3WAY)
sup_h20_tok, sup_h20_ref = cell("n6suph20_tok", "h20", "g1"), cell("n6suph20_ref", "h20", "g1")
sup_h20_tok_h, sup_h20_ref_h = cell("n6suph20_tok", "h20", "h1"), cell("n6suph20_ref", "h20", "h1")
sup2x_tok, sup2x_ref = cell("n6sup2x_tok", "h1", "g1"), cell("n6sup2x_ref", "h1", "g1")
sup2x_tok_h, sup2x_ref_h = cell("n6sup2x_tok", "h1", "h1"), cell("n6sup2x_ref", "h1", "h1")
zs_cot_g, zs_cot_h = cell("n6cot_tok", "zsh1", "g1"), cell("n6cot_tok", "zsh1", "h1")

videos_html, vid_bytes = video_cards()

HTML = f"""<title>Motion Token Ledger</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root {{
  color-scheme: light;
  --paper:#f4f3ee; --sheet:#fbfaf7; --ink:#191a1d; --ink2:#4c4e54; --mute:#83858c; --rule:#dcdad2; --rule2:#c9c7be;
  --tok:#2457c5; --tokbg:#e6ecfa; --ref:#c4561f; --refbg:#f9e8dd; --good:#1f7a45; --goodbg:#e3f1e8; --open:#b07a0a; --openbg:#faf0d6; --block:#a83232; --blockbg:#f7e1e1;
  --best:#eef4ff;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  color-scheme: dark;
  --paper:#141518; --sheet:#1c1d21; --ink:#ecebe6; --ink2:#bdbcb5; --mute:#8b8a84; --rule:#2e2f34; --rule2:#3c3d43;
  --tok:#6f95ee; --tokbg:#1e2a45; --ref:#e58a58; --refbg:#3d251a; --good:#5fc48a; --goodbg:#193225; --open:#e0b04a; --openbg:#3a2f14; --block:#e26d6d; --blockbg:#3e1f1f;
  --best:#1b2437;
}} }}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --paper:#141518; --sheet:#1c1d21; --ink:#ecebe6; --ink2:#bdbcb5; --mute:#8b8a84; --rule:#2e2f34; --rule2:#3c3d43;
  --tok:#6f95ee; --tokbg:#1e2a45; --ref:#e58a58; --refbg:#3d251a; --good:#5fc48a; --goodbg:#193225; --open:#e0b04a; --openbg:#3a2f14; --block:#e26d6d; --blockbg:#3e1f1f;
  --best:#1b2437;
}}
body {{ margin:0; background:var(--paper); color:var(--ink); font:15.5px/1.55 "Source Sans 3", "Segoe UI", system-ui, sans-serif; }}
.wrap {{ max-width:1120px; margin:0 auto; padding:36px 24px 90px; }}
h1,h2,h3 {{ font-family:"Fraunces", Georgia, serif; text-wrap:balance; margin:0; }}
h1 {{ font-size:clamp(30px,4.2vw,44px); font-weight:700; letter-spacing:-.015em; line-height:1.08; }}
h2 {{ font-size:24px; font-weight:600; margin:0 0 6px; }}
h3 {{ font-size:17px; font-weight:600; margin:0 0 6px; }}
p {{ margin:0 0 10px; max-width:70ch; }}
.eyebrow {{ font:500 11.5px/1 "JetBrains Mono", monospace; letter-spacing:.12em; text-transform:uppercase; color:var(--mute); margin-bottom:10px; }}
.mono {{ font-family:"JetBrains Mono", monospace; font-size:12.5px; color:var(--mute); }}
header {{ display:grid; grid-template-columns:1.4fr 1fr; gap:28px; align-items:end; padding-bottom:26px; border-bottom:2px solid var(--ink); margin-bottom:34px; }}
.tally {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }}
.tally div {{ padding:12px 14px; border-radius:4px; }}
.tally b {{ display:block; font-family:"Fraunces", serif; font-size:30px; line-height:1; font-weight:700; }}
.tally span {{ font:500 11px/1 "JetBrains Mono", monospace; letter-spacing:.1em; text-transform:uppercase; }}
.t-good {{ background:var(--goodbg); color:var(--good); }} .t-open {{ background:var(--openbg); color:var(--open); }} .t-block {{ background:var(--blockbg); color:var(--block); }}
section {{ margin:0 0 46px; }}
.verdict {{ font-family:"Fraunces", serif; font-size:21px; line-height:1.4; max-width:62ch; font-weight:500; }}
.verdict em {{ color:var(--tok); font-style:normal; }}
.ledger {{ display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }}
.ledger > div {{ background:var(--sheet); border-top:3px solid; padding:14px 16px 8px; border-radius:0 0 4px 4px; }}
.ledger .good {{ border-color:var(--good); }} .ledger .open {{ border-color:var(--open); }} .ledger .block {{ border-color:var(--block); }}
.ledger h3 {{ font-size:15px; letter-spacing:.02em; }}
.ledger ul {{ margin:8px 0 0; padding:0; list-style:none; }}
.ledger li {{ padding:7px 0; border-top:1px solid var(--rule); font-size:14.2px; line-height:1.4; }}
.ledger li b {{ font-weight:600; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th {{ text-align:left; font:500 11.5px/1.3 "JetBrains Mono", monospace; letter-spacing:.06em; text-transform:uppercase; color:var(--mute); padding:8px 10px; border-bottom:1px solid var(--rule2); }}
td {{ padding:8px 10px; border-bottom:1px solid var(--rule); vertical-align:top; }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
tr.best td {{ background:var(--best); font-weight:600; }}
.note {{ display:block; font-size:12.5px; color:var(--mute); font-weight:400; }}
.tablewrap {{ overflow-x:auto; background:var(--sheet); border:1px solid var(--rule); border-radius:4px; padding:4px 6px 2px; margin:10px 0 14px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:13px; color:var(--ink2); margin:6px 0 12px; }}
.legend i {{ display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-1px; margin-right:6px; }}
.vids {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:18px; }}
figure.vid {{ margin:0; background:var(--sheet); border:1px solid var(--rule); border-radius:4px; overflow:hidden; }}
figure.vid video {{ width:100%; display:block; background:#000; aspect-ratio:3/2; }}
figure.vid .ph {{ aspect-ratio:3/2; display:grid; place-items:center; color:var(--mute); font-family:"JetBrains Mono", monospace; font-size:13px; background:var(--paper); }}
figcaption {{ padding:10px 12px 12px; }}
figcaption b {{ display:block; font-size:13.5px; margin-bottom:4px; }}
figcaption p {{ font-size:13.2px; color:var(--ink2); margin:0 0 4px; }}
.spec {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px 22px; font-size:14px; }}
.spec div {{ padding:8px 0; border-top:1px solid var(--rule); }}
.spec b {{ display:block; font:500 11px/1.4 "JetBrains Mono", monospace; letter-spacing:.08em; text-transform:uppercase; color:var(--mute); }}
svg text {{ font-family:"Source Sans 3", system-ui, sans-serif; fill:var(--ink); }}
svg .lab {{ font-family:"JetBrains Mono", monospace; font-size:10.5px; fill:var(--mute); }}
svg .box {{ fill:var(--sheet); stroke:var(--rule2); }}
svg .tokb {{ fill:var(--tokbg); stroke:var(--tok); }}
svg .refb {{ fill:var(--refbg); stroke:var(--ref); }}
svg .arrow {{ stroke:var(--ink2); fill:none; }}
.waves li {{ margin:0 0 6px; }}
.waves code, .mono code {{ font-family:"JetBrains Mono", monospace; font-size:12.5px; background:var(--sheet); padding:1px 5px; border-radius:3px; }}
.foot {{ color:var(--mute); font-size:13px; border-top:1px solid var(--rule2); padding-top:14px; }}
@media (max-width:820px) {{ header, .grid2, .ledger {{ grid-template-columns:1fr; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation:none !important; transition:none !important; }} }}
</style>
<div class="wrap">
<header>
  <div>
    <div class="eyebrow">FSQ motion tokens · URMA2 one-policy-many-bodies · state as of 2026-09-02 15:00</div>
    <h1>Motion Token Ledger</h1>
    <p style="margin-top:12px;color:var(--ink2)">What the token has solved, what it has not, and what is blocked. Tracking error is joint-space RMSE in radians against the retargeted clip, lower is better; every number is a mean over train seeds × rollout seeds (shown as s×r), all rollouts alive unless stated.</p>
  </div>
  <div class="tally">
    <div class="t-good"><b>8</b><span>solved</span></div>
    <div class="t-open"><b>7</b><span>open</span></div>
    <div class="t-block"><b>5</b><span>blocked</span></div>
  </div>
</header>

<section>
  <div class="eyebrow">the question this project asks</div>
  <p class="verdict">Does a motion token let one policy across bodies perform complex motion better than the raw reference does, at full reference rate or with the reference removed? <em>As a precomputed sidecar with its own projection: −5 to −6 % on both robots (four seeds), and legs move for the first time (−4 to −5 %). Training the encoder inside the policy adds nothing more on a single clip, but keeps a gain where the sidecar loses it: heavily randomized bodies (−6 / −5 %) and five dances (−7 %, one seed).</em> With the reference removed the sidecar token is information-limited (the gap widens with budget) and the co-trained encoder collapses outright. Where the token wins big is only when the reference goes stale or comes from another robot, and there the frozen sidecar beats the co-trained encoder.</p>
</section>

<section>
  <div class="eyebrow">ledger</div>
  <div class="ledger">
    <div class="good"><h3>Solved</h3><ul>
      <li><b>A properly routed token beats the reference at hold 1.</b> Separate projection G1 {fmt(cell('n6split_tok','h1','g1'))} / H1 {fmt(cell('n6split_tok','h1','h1'))}, co-trained encoder {fmt(cell('n6cot_tok','h1','g1'))} / {fmt(cell('n6cot_tok','h1','h1'))} (four seeds) vs reference {fmt(cell(REF,'h1','g1'))} / {fmt(cell(REF,'h1','h1'))}; legs −4 to −5 %. The co-trained variant keeps its gain under heavy body randomization and on five dances, where the sidecar loses it.</li>
      <li><b>Staleness.</b> Token −20 to −40 % at reference hold 5 to 20 on two robots; −36 to −38 % on the 5-dance clip when trained at hold 20.</li>
      <li><b>Cross-robot transfer.</b> G1 driven by H1's code stream keeps 83 % of own-token value, H1 from G1 90 %; codec decode error predicts it.</li>
      <li><b>Unseen motion.</b> walk1 (never in tokenizer or policy training) tracks through the token at hold 20; token arms at hold 1 sit at or below reference (co-trained {fmt(zs_cot_g)} / {fmt(zs_cot_h)} vs 0.154 / 0.170).</li>
      <li><b>Three topologies, one policy.</b> H1, G1 and BoosterT1 at 1.8 to 2.0× the zero-action floor after the T1 reference fix.</li>
      <li><b>Tokenizer quality.</b> v2 per-joint FSQ with foot channels: held-out walk1 reconstruction 0.16 to 0.20 rad, 2× better than v1.</li>
      <li><b>Token routing bugs.</b> The 559× loudness bug (divisor 10) and, today, the never-existing "separate projection" (wrapper attribute always 0) are fixed with old checkpoints still loading.</li>
      <li><b>Heading and reference feasibility.</b> Heading observed + rewarded (80° → 8°); infeasible retargets fixed with stiffened limits on 4 families.</li>
    </ul></div>
    <div class="open"><h3>Open</h3><ul>
      <li><b>Legs skate everywhere.</b> Arms track at r ≈ 0.95, legs at 0.33 to 0.40, ankles ≈ 0; leg RMSE sits 10 to 15 % above the zero-action floor on every policy, token or not. The leg-weighted kernel buys 4 to 7 % on legs at the cost of arms.</li>
      <li><b>Reference-free tracking.</b> Sidecar-token-only is +17 / +14 % worse than reference at 2× budget and +24 / +15 % at 4×: information-limited. The co-trained encoder without the reference channel collapses to the zero-action floor (0.40 / 0.49).</li>
      <li><b>Hold-1 gains are modest</b> (−5 to −6 %) and the co-training increment over the separate projection is within seed noise on a single clip (seed spread 0.145 to 0.163 on G1).</li>
      <li><b>Multi-motion at hold 1</b> is a tie (5 dances, 1× and 2× budget); only stale-reference training separates the arms.</li>
      <li><b>Three-robot staleness</b> gain is weak (−2 / −5 / +2 %) when trained at hold 1; the hold-20-trained 3-robot pair is queued locally.</li>
      <li><b>Heavy morphology randomization</b> (ramp to 0.7) erases the token's hold-1 gain; the hold-20 gain survives.</li>
      <li><b>T1 transfer</b> recovers only 5 to 23 %: T1's feasibility-clamped retarget lives in a different code region (and its arm retarget is off by 14 to 23 cm).</li>
    </ul></div>
    <div class="block"><h3>Blocked</h3><ul>
      <li><b>More than two topologies on Viper.</b> Any 3-robot graph faults on ROCm at every env count; 3-robot training is local-only at 192 envs (6.5M steps per robot).</li>
      <li><b>Morphology-perturbed evaluation on Viper</b> faults on ROCm; runs locally on CUDA only.</li>
      <li><b>Local GPU is one process wide.</b> Three concurrent compiles restarted the WSL VM and killed the ssh masters; queues are strictly sequential (each arm ≈ 55 min + 30 min eval).</li>
      <li><b>Disk.</b> C: had 3.5 GB free during the mirror; older checkpoints, snapshots and the 31 GB gaitproof tree need an external disk.</li>
      <li><b>Aux next-token head is closed negative.</b> The detach probe shows the trunk already predicts the next token; the head adds nothing.</li>
    </ul></div>
  </div>
</section>

<section>
  <div class="eyebrow">how the token is made</div>
  <h2>The FSQ tokenizer, and the three ways the policy meets it</h2>
  <div class="grid2">
   <div>
    <p>A per-joint autoencoder (khaendler, loco-mujoco <span class="mono">autoencoder</span> branch) trained on the same retargeted clips the RL environment consumes. For every frame the input is a window of 11 rows (frame t, then goal frames t … t+9, clamped at the clip end) with 4 channels per joint: joint angle, joint velocity, left and right foot height above ground (the foot channels are the v2 addition, broadcast to every joint so each code must carry contact context). Each joint is encoded independently, so the code is a per-joint stream, not a whole-body one.</p>
    <p>The encoder is two Dense(64) layers, two temporal Conv(3) layers with LayerNorm, a mean over the 11 rows, and Dense(32). Finite scalar quantization rounds each of the 32 latent dimensions to one of 8 levels with a straight-through gradient, so the code is one of 8<sup>32</sup> discrete cells, and a 100 Hz stream of 32 small integers per joint is the whole interface. The decoder is a URMA-style network conditioned on the 47-dimensional joint descriptor, which is what lets one tokenizer serve H1, G1 and BoosterT1 with different joint sets. Loss is plain MSE on the reconstructed window. Warning that cost us a week: that loss is 97 % velocity, so tokenizers are ranked by held-out joint-angle RMSE, never by training loss.</p>
   </div>
   <div class="spec">
    <div><b>training data</b>3 robots × 9 clips (5 dances, 4 walk cycles); walk1_subject1 held out entirely; 8113 train / 902 test windows per dance clip</div>
    <div><b>optimisation</b>AdamW 1.5e-3, cosine decay, batch 512, 250 epochs, 35 min on the local GPU</div>
    <div><b>codebook</b>32 dims × 8 levels per joint, rounded with straight-through gradients; 147k distinct codes used on dance2_subject4 H1</div>
    <div><b>held-out quality</b>walk1 angle RMSE H1 0.162 / G1 0.204 / T1 0.158 rad (v1: 0.345)</div>
    <div><b>emission</b><span class="mono">&lt;clip&gt;_zq.npz</span> sidecar: (frames, joints, 32) at 100 Hz, aligned to the clip; the env maps joints by name and holds the code for K frames to simulate a low-rate stream</div>
    <div><b>window sidecar (co-training)</b><span class="mono">&lt;clip&gt;_win.npz</span>: the raw 11×4 window per joint (44 channels), so the encoder can run online</div>
   </div>
  </div>
  <svg viewBox="0 0 1080 300" width="100%" style="margin-top:14px" role="img" aria-label="Tokenizer and the three policy interfaces">
    <rect class="box" x="10" y="20" width="200" height="72" rx="4"/><text x="20" y="44" font-size="14" font-weight="600">reference window</text><text x="20" y="64" class="lab">11 rows × 4 ch per joint</text><text x="20" y="80" class="lab">angle · velocity · foot L · foot R</text>
    <path class="arrow" d="M210 56 H262" marker-end="url(#a)"/>
    <rect class="box" x="264" y="20" width="176" height="72" rx="4"/><text x="274" y="44" font-size="14" font-weight="600">encoder</text><text x="274" y="64" class="lab">Dense·Dense·Conv·Conv·mean</text><text x="274" y="80" class="lab">per joint, shared weights</text>
    <path class="arrow" d="M440 56 H492" marker-end="url(#a)"/>
    <rect class="tokb" x="494" y="20" width="150" height="72" rx="4"/><text x="504" y="44" font-size="14" font-weight="600" fill="var(--tok)">FSQ code</text><text x="504" y="64" class="lab">32 dims × 8 levels</text><text x="504" y="80" class="lab">straight-through</text>
    <path class="arrow" d="M644 56 H696" marker-end="url(#a)"/>
    <rect class="box" x="698" y="20" width="176" height="72" rx="4"/><text x="708" y="44" font-size="14" font-weight="600">URMA decoder</text><text x="708" y="64" class="lab">conditioned on 47-d joint descriptor</text><text x="708" y="80" class="lab">→ reconstructed window</text>
    <text x="890" y="44" font-size="13" fill="var(--ink2)">MSE on the window</text><text x="890" y="64" class="lab">gate: held-out angle RMSE</text>
    <line x1="10" y1="118" x2="1070" y2="118" stroke="var(--rule2)" stroke-dasharray="4 4"/>
    <text x="10" y="142" class="lab">HOW THE POLICY SEES IT</text>
    <rect class="refb" x="10" y="156" width="330" height="60" rx="4"/><text x="20" y="178" font-size="13.5" font-weight="600" fill="var(--ref)">design A (retired)</text><text x="20" y="198" font-size="12.5">decode the code, hand the policy a reconstructed reference</text><text x="20" y="211" class="lab">measured reference quality, not the token</text>
    <rect class="tokb" x="370" y="156" width="330" height="60" rx="4"/><text x="380" y="178" font-size="13.5" font-weight="600" fill="var(--tok)">design B (all campaign results)</text><text x="380" y="198" font-size="12.5">precomputed code appended to each joint's observation</text><text x="380" y="211" class="lab">5 + 32 channels per joint · divisor 10 · shared Dense(8)</text>
    <rect class="tokb" x="730" y="156" width="340" height="60" rx="4"/><text x="740" y="178" font-size="13.5" font-weight="600" fill="var(--tok)">co-training (wave 6, best hold-1 result)</text><text x="740" y="198" font-size="12.5">encoder + FSQ live inside the policy; PPO gradients reach them</text><text x="740" y="211" class="lab">init from the tokenizer · recon head keeps it a codec</text>
    <text x="10" y="252" font-size="12.5" fill="var(--ink2)">Design B's code is fixed once the clip is tokenized; co-training lets the control objective reshape which distinctions the code keeps. Both hand the policy the same 32 channels per joint.</text>
    <text x="10" y="272" font-size="12.5" fill="var(--ink2)">"Token only" removes the explicit reference channel and leaves the policy with proprioception + code; "hold K" freezes the observed reference for K control steps while the reward stays fresh.</text>
    <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--ink2)"/></marker></defs>
  </svg>
</section>

<section>
  <div class="eyebrow">results at full reference rate (hold 1) — the project's question</div>
  <h2>Tracking with a fresh reference every step, dance2_subject4, one policy over H1 + G1</h2>
  <p>Same recipe for every row (bundle contact terms, swing term 0.5, heading observed, Viper's default morphology schedule 0.2 → 0.44). Delta is against the reference-only control. Legs = mean RMSE over hip, knee and ankle joints.</p>
  <div class="tablewrap"><table>
    <tr><th>arm</th><th class="num">G1</th><th class="num">H1</th><th class="num">Δ vs reference</th><th class="num">legs G1 / H1</th><th class="num">s×r</th></tr>
    {hold1_rows}
  </table></div>
  <div class="grid2">
   <div>
    <h3>Reference removed: token only</h3>
    <p>The user's second question: can the code replace the reference? Not as a precomputed sidecar. The gap does not close with compute.</p>
    <div class="tablewrap"><table><tr><th>arm</th><th class="num">G1</th><th class="num">H1</th><th class="num">s×r</th></tr>{replace_rows}</table></div>
   </div>
   <div>
    <h3>Body randomization during training</h3>
    <p>Token minus reference at hold 1 and at hold 20, by how hard the bodies are randomized in training. Heavier randomization erases the hold-1 gain and leaves the hold-20 gain alone.</p>
    <div class="tablewrap"><table><tr><th>training bodies</th><th class="num">ref G1 / H1</th><th class="num">token G1 / H1</th><th class="num">Δ hold 1</th><th class="num">Δ hold 20</th></tr>{morph_rows}</table></div>
   </div>
  </div>
  <div class="grid2">
   <div>
    <h3>Multi-motion (five dances in one clip)</h3>
    <div class="tablewrap"><table><tr><th>condition</th><th class="num">ref G1 / H1</th><th class="num">token G1 / H1</th></tr>
      <tr><td>hold 1, 2× budget</td><td class="num">{fmt(sup2x_ref)} / {fmt(sup2x_ref_h)}</td><td class="num">{fmt(sup2x_tok)} / {fmt(sup2x_tok_h)}</td></tr>
      <tr><td>trained and evaluated at hold 20</td><td class="num">{fmt(sup_h20_ref)} / {fmt(sup_h20_ref_h)}</td><td class="num">{fmt(sup_h20_tok)} / {fmt(sup_h20_tok_h)}</td></tr>
    </table></div>
    <p>Tie at full rate for the sidecar token, −36 / −38 % once the reference is stale. The co-trained encoder on the same five dances: {fmt(cell('n6cotsup_tok','h1','g1'))} / {fmt(cell('n6cotsup_tok','h1','h1'))} at hold 1 (one seed, second running).</p>
   </div>
   <div>
    <h3>Three topologies, one policy (local, H1 + G1 + T1, hold 1)</h3>
    <div class="tablewrap"><table><tr><th>arm</th><th class="num">H1</th><th class="num">G1</th><th class="num">T1</th><th class="num">zero-action floor</th></tr>
      <tr><td>reference only</td><td class="num">0.226</td><td class="num">0.195</td><td class="num">0.211</td><td class="num">0.433 / 0.388 / 0.415</td></tr>
      <tr><td>precomputed token</td><td class="num">0.240</td><td class="num">0.187</td><td class="num">0.218</td><td class="num"></td></tr>
      <tr><td>reference only, evaluated at hold 20</td><td class="num">0.387</td><td class="num">0.340</td><td class="num">0.352</td><td class="num"></td></tr>
      <tr><td>token, evaluated at hold 20</td><td class="num">0.380</td><td class="num">0.324</td><td class="num">0.359</td><td class="num"></td></tr>
    </table></div>
    <p>Neutral at hold 1 (4 rollout seeds); the token does fix heading there (40° → 8–17°). The hold-20-trained 3-robot pair is queued locally.</p>
   </div>
  </div>
</section>

<section>
  <div class="eyebrow">where the token clearly pays</div>
  <h2>Stale, low-rate, or foreign reference</h2>
  <div class="grid2">
   <div>
    <h3>Reference rate curve (two robots, dance2_subject4)</h3>
    <div class="tablewrap"><table><tr><th>reference hold</th><th class="num">Δ token H1</th><th class="num">Δ token G1</th></tr>
      <tr><td>1 (every step)</td><td class="num">+2 %</td><td class="num">+4 %</td></tr><tr><td>2</td><td class="num">−7 %</td><td class="num">−12 %</td></tr><tr><td>5</td><td class="num">−21 %</td><td class="num">−22 %</td></tr><tr><td>20</td><td class="num">−40 %</td><td class="num">−42 %</td></tr>
      <tr><td>20, bodies perturbed 0.3 / 0.6 at eval</td><td class="num">−38 / −34 %</td><td class="num">−40 / −33 %</td></tr>
    </table></div>
    <p>The token arm is nearly flat in staleness; reference-only collapses. On perturbed bodies the split is −55 to −62 % on arms and −13 to −15 % on legs.</p>
   </div>
   <div>
    <h3>Cross-robot transfer (hold 20, recovery of own-token value)</h3>
    <div class="tablewrap"><table><tr><th>stream from</th><th class="num">→ H1</th><th class="num">→ G1</th><th class="num">→ T1</th></tr>
      <tr><td>H1</td><td class="num">own</td><td class="num">83 %</td><td class="num">22 %</td></tr><tr><td>G1</td><td class="num">90 %</td><td class="num">own</td><td class="num">23 %</td></tr><tr><td>T1</td><td class="num">5 %</td><td class="num">12 %</td><td class="num">own</td></tr>
    </table></div>
    <p>Offline decode error of the crossed stream orders every cell (G1←H1 0.22 rad → 83 %; T1←H1 0.59 → 22 %). Zero-shot walk1 × transfer: ~90 %.</p>
   </div>
  </div>
  <h3 style="margin-top:6px">Wave 6c, running now on Viper (rows fill in as cross-evals land)</h3>
  <div class="grid2">
   <div class="tablewrap"><table><tr><th>hold-20-trained three-way</th><th class="num">at hold 20 G1 / H1</th><th class="num">at hold 1 G1 / H1</th><th class="num">s×r</th></tr>{h20_rows}</table></div>
   <div class="tablewrap"><table><tr><th>co-training follow-ups (hold 1)</th><th class="num">G1</th><th class="num">H1</th><th class="num">s×r</th></tr>{cot6c_rows}</table></div>
  </div>
</section>

<section>
  <div class="eyebrow">the visible gap</div>
  <h2>Every policy skates</h2>
  <div class="grid2">
   <div>
    <div class="tablewrap"><table><tr><th>per-joint correlation with the clip</th><th class="num">arms</th><th class="num">legs</th><th class="num">ankles</th></tr>
      <tr><td>3-robot policy, H1 / G1 / T1 (hold 1)</td><td class="num">0.95 / 0.95 / 0.96</td><td class="num">0.39 / 0.33 / 0.33</td><td class="num">0.05 / 0.06 / 0.06</td></tr>
      <tr><td>token arm, perturbed body 0.3, H1 / G1 (hold 20)</td><td class="num">0.96 / 0.95</td><td class="num">0.35 / 0.42</td><td class="num">0.12 / 0.13</td></tr>
      <tr><td>co-trained encoder, G1 / H1 (hold 1)</td><td class="num">—</td><td class="num">0.59 / 0.54</td><td class="num">—</td></tr>
    </table></div>
   </div>
   <div>
    <p>Hip pitch and knee reach r ≈ 0.5–0.7; hip roll/yaw and ankles do not track. Foot airborne fraction is 0.02–0.06 against a reference 0.12–0.28: the policies match arm angles and shuffle. Swing-match dose buys airborne up to reference level (dose 50, +14 % RMSE), the leg-weighted kernel buys 4–7 % leg RMSE, and the properly routed token variants are the first whose gain lands on the legs (−4 to −5 %; with the leg kernel the co-trained arm reaches 0.170 on G1 legs vs 0.190 reference). Fixing legs is the prerequisite for presentable dance on any body.</p>
   </div>
  </div>
</section>

<section>
  <div class="eyebrow">videos · {vid_bytes/1e6:.1f} MB embedded</div>
  <h2>Rollouts</h2>
  <p>Rendered from cross-eval dumps with the environment's own model. Green marker = the clip's root target. Longer rollouts (24–36 s) are rendered on the local GPU between training arms; a card shows the earlier short clip until its longer version lands.</p>
  <div class="vids">
  {videos_html}
  </div>
</section>

<section>
  <div class="eyebrow">experiment index</div>
  <h2>What exists, by wave</h2>
  <ul class="waves">
    <li><b>Waves 1–5 (Aug 22–29)</b> — baseline fixes (contact dose, feet, heading, T1 reference), tokenizer quality (Kevin v2 decoder, mc emission), design A vs B, width/routing ablations; reports in <code>experiments/fsq_khaendler/REPORT_*.md</code>.</li>
    <li><b>V1 / W1–W3 (Aug 29–30)</b> — token routing fix (divisor 10), rate curve at hold 1/2/5/10/20, token-only, heading; <code>docs/notes/FINAL_CONFIG_2026-08-30.md</code>.</li>
    <li><b>Night 3–5 (Aug 30–Sep 1)</b> — gait dose grid, super-clip, morphology eval, v2 tokenizer, zero-shot walk1, cross-robot transfer matrix; <code>experiments/fsq_khaendler/REPORT_NIGHT5.md</code>.</li>
    <li><b>Wave 6a (Sep 2)</b> — budget 2×/4×, multi-motion at 2× and hold-20-trained, morphology controls, swing 25, zero-shot at hold 1. 26 trains, 170 cross-evals, done.</li>
    <li><b>Wave 6b (Sep 2)</b> — real split projection, aux next-token head, leg-weighted kernel, SONIC co-training (init / scratch / frozen / + aux). 22 trains, done.</li>
    <li><b>Wave 6c (Sep 2, running)</b> — co-training seeds 3–4, hold-20-trained three-way, recon 0.1, reference-free co-training, co-training × morphology, × leg kernel, × 5 dances. 22 trains.</li>
    <li><b>Local queue (Sep 2)</b> — the same eight arms on nominal bodies for a clean local replicate, then the 3-robot hold-20 pair; <code>experiments/local_w6/eval/</code>.</li>
    <li>All Viper cross-evals, logs, curves and checkpoints mirrored to <code>viper_mirror/</code>; code changes are anchored patch scripts in <code>scripts/scaling/wave6/</code>; narrative in <code>docs/notes/WAVE6_2026-09-02.md</code>.</li>
  </ul>
</section>

<p class="foot">Controls for the hold-1 table: reference-only n5sw05_ref (3 train seeds × 4 rollouts), precomputed token n5v2_tok (3 × 4). Every earlier "tk4" arm ran the shared routing; the separate projection first exists in wave 6b. All Viper arms train under the sbatch default morphology schedule; the local 3-robot arms train on nominal bodies. Rollout seed alone moves a cross-eval by up to 5 %, so single-seed rows are labelled and not claimed.</p>
</div>
"""
OUT.write_bytes(HTML.encode("utf-8"))
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB, videos {vid_bytes/1e6:.1f} MB)")
for k in ["n6cot_tok", "n6split_tok", REF, TOK]:
    print(k, {c: D.get((k, c, "g1"), {}).get("rmse") for c in ("h1", "h20", "zsh1")})
