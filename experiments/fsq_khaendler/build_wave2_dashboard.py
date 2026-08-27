"""Build the WAVE 2 dashboard: verdicts, every figure, every video, one file.

Figures come from plot_wave2.py (plain white matplotlib). Both figures and videos
are embedded as data URIs so the page works from anywhere.
"""
from __future__ import annotations

import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
VID = HERE / "videos_wave2"
FIG = HERE / "plots_wave2"
OUT = HERE / "dashboard_fsq_wave2.html"


def uri(p: Path, mime: str) -> str | None:
    if not p.exists():
        print(f"  MISSING {p.name}")
        return None
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


# (file, heading, body) — body may contain inline HTML
FIGURES = {
    "heading": ("f1_heading_unlearned.png", "Heading is given up in the first 10M steps",
        "This curve is logged by every run this project has ever done, and nobody had plotted it. "
        "Every arm starts near the reference's own motion and reaches ~84&deg; before training is a "
        "tenth done. The two arms with the heading reward switched on climb the same way."),
    "blind": ("f2_training_is_blind.png", "Why every number here comes from a crosseval",
        "Reference, token and both are indistinguishable on episode length, return and training joint "
        "error &mdash; they converge together and saturate. The effect under study is roughly 4 %, and "
        "none of these three can see it. This is why an arm is never judged by its training curve."),
    "canon": ("f10_canonical_fails_visibly.png", "The one exception",
        "The shared body-independent stream is the single failure large enough for a training curve to "
        "show. When a curve does separate in this project, it means something has gone very wrong."),
    "degrade": ("f3_reference_degradation.png", "Take the reference away and watch who copes",
        "The load-bearing experiment. The observed reference is frozen every K clip frames while the "
        "reward target stays fresh &mdash; before this run they shared one array, so the reference could "
        "not be degraded for the policy without also degrading what it was scored against."),
    "rate": ("f4_rate_curve.png", "How often the token has to arrive",
        "Nine dances, token replacing the reference. The interface survives a 5&times; rate cut almost "
        "intact and breaks below 8 tokens/s."),
    "msweep": ("f5_motion_sweep.png", "The token helps at every number of motions",
        "Negative at M = 1, 4, 5 and 9, significant at M = 1 and 4 &mdash; and not monotone. Whatever "
        "sets the size of the effect, it is not how much motion there is to compress."),
    "twobytwo": ("f6_two_by_two.png", "What the token survives, and what it does not",
        "The 2&times;2 that took four training arms. Randomizing the body is free to the token; a second "
        "robot erases it, on both robots, with and without randomization."),
    "decoder": ("f7_decoder_width.png", "The body-independent token, and its gate",
        "Decoder width is the only one of three suspects that moves the number, and it does so "
        "monotonically with no sign of saturating &mdash; it is simply not enough. The gate declined to "
        "spend a 3-hour RL arm on it, automatically, without a human in the loop."),
    "selloff": ("f11_selloff.png", "The recipe sells heading and foot-lift off",
        "The most useful measurement of the campaign, available in every log since the project began. "
        "At one million steps the policy lifts its feet MORE than the clip asks and faces nearly the "
        "right way. Heading goes between 6M and 10M steps, the feet between 10M and 20M, and joint "
        "accuracy improves the whole way &mdash; it is what they are traded for. Turning the foot terms "
        "on lifts the feet back (ratio 0.30 &rarr; 0.54) at a cost of 44 % joint accuracy. The four "
        "heading curves lie on top of each other."),
    "headbar": ("f8_heading_across_arms.png", "Sixteen arms, one control, nothing in between",
        "Amber is a policy that emits no torque at all. Everything trained &mdash; every motion count, "
        "both topology counts, fixed and randomized bodies, real token and scrambled &mdash; lands "
        "between 70&deg; and 87&deg;."),
    "ladder": ("f9_ladder_rungs.png", "An earlier campaign: what each step of generality costs",
        "The LADDER rungs, on three motions. Randomized bodies (L1) cost real training accuracy; adding "
        "the second topology (L2) does not, but read that with care &mdash; L2's curve averages over both "
        "robots and G1 tracks better than H1, which pulls the mean down."),
}

VIDEOS = [
    ("M9_both_h1.mp4", "M9_both", "H1",
     "The best single-body arm, 0.1263 rad. Policy travels 1.13 m against the clip's 1.25 m &mdash; the "
     "closest match of the three."),
    ("M9_ref_h1.mp4", "M9_ref", "H1",
     "The same recipe with the token removed, 0.1315 rad. The 4 % that separates it from the pane above "
     "is not visible to the eye &mdash; which is the whole reason this project measures rather than "
     "watches. Note the policy covers 2.92 m where the clip covers 1.25 m."),
    ("M9_z_h1.mp4", "M9_z", "H1",
     "Token <em>replacing</em> the reference rather than supplementing it, 0.1341 rad. Slightly worse "
     "than reference-alone, and the only configuration where FSQ loses. 2.74 m of travel against 1.25 m."),
    ("M5_2t_both_morph_unitree_h1.mp4", "M5_2t_both_morph", "H1",
     "One policy: two topologies, five motions, morphology randomization ramping to 0.3. The most "
     "general arm the data supports."),
    ("M5_2t_both_morph_unitree_g1.mp4", "M5_2t_both_morph", "G1",
     "The same weights, the other robot, 0.1289 rad. One network produced both this and the pane above "
     "&mdash; but the two panes are at <b>different points in the clip</b> (see the note above)."),
    ("M5_2t_ref_morph_unitree_h1.mp4", "M5_2t_ref_morph", "H1",
     "Its control, without the token. At two topologies the two are indistinguishable &mdash; this pair "
     "is what the right-hand column of the 2&times;2 looks like."),
    ("M5_2t_ref_morph_unitree_g1.mp4", "M5_2t_ref_morph", "G1",
     "0.1245 rad, the best G1 number of the campaign, from the arm <em>without</em> the token."),
    ("M5_2t_canon_unitree_h1.mp4", "M5_2t_canon", "H1",
     "The body-independent token: one shared code stream driving both robots. It tracks worse than "
     "sending no torque at all &mdash; and note it is not frozen. It moves, just not with the clip."),
    ("M5_2t_canon_unitree_g1.mp4", "M5_2t_canon", "G1",
     "The same stream on G1, 0.22 m of travel against the clip's 0.28 m &mdash; it stays roughly put "
     "while the joints do the wrong thing."),
]


def build() -> str:
    figs = {}
    for key, (fn, title, blurb) in FIGURES.items():
        u = uri(FIG / fn, "image/png")
        figs[key] = "" if not u else (
            f'<figure class="card fig"><h3>{title}</h3>'
            f'<div class="imgwrap"><img src="{u}" alt="{title}" loading="lazy"></div>'
            f'<figcaption>{blurb}</figcaption></figure>')

    vids = []
    for fn, arm, robot, blurb in VIDEOS:
        u = uri(VID / fn, "video/mp4")
        if not u:
            continue
        vids.append(
            f'      <div class="vid">\n'
            f'        <video src="{u}" controls loop muted playsinline preload="none"></video>\n'
            f'        <div class="meta"><span class="eyebrow mono">{arm} &middot; {robot}</span>'
            f'<p>{blurb}</p></div>\n      </div>')
    return (TEMPLATE.replace("%%CSS%%", CSS)
            .replace("%%VIDEOS%%", "\n".join(vids))
            .replace("%%F_" + "%%", "")
            .replace("%%FIG_HEADING%%", figs["heading"])
            .replace("%%FIG_BLIND%%", figs["blind"])
            .replace("%%FIG_CANON%%", figs["canon"])
            .replace("%%FIG_DEGRADE%%", figs["degrade"])
            .replace("%%FIG_RATE%%", figs["rate"])
            .replace("%%FIG_MSWEEP%%", figs["msweep"])
            .replace("%%FIG_2X2%%", figs["twobytwo"])
            .replace("%%FIG_DECODER%%", figs["decoder"])
            .replace("%%FIG_HEADBAR%%", figs["headbar"])
            .replace("%%FIG_SELLOFF%%", figs["selloff"])
            .replace("%%FIG_LADDER%%", figs["ladder"]))


CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --ground:#E9EFF1; --surface:#F7F9FA; --surface-2:#FFFFFF;
  --ink:#101A1F; --ink-2:#3D4B53; --ink-3:#6A7A82;
  --line:#CBD8DD; --line-soft:#DEE7EA;
  --accent:#1B62A5; --accent-soft:#DCE8F3;
  --s1:#1B62A5; --s2:#C67B10; --s3:#8E3C86;
  --pass:#1F6F49; --pass-bg:#DCEDE3;
  --warn:#8A5A00; --warn-bg:#F5E9D2;
  --fail:#9C3227; --fail-bg:#F6E0DC;
  --figure-bg:#FFFFFF;
  --shadow:0 1px 2px rgba(16,26,31,.06),0 8px 24px -16px rgba(16,26,31,.28);
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0B1215; --surface:#131C20; --surface-2:#18242A;
    --ink:#E7EEF1; --ink-2:#AABBC3; --ink-3:#7D9099;
    --line:#25353B; --line-soft:#1D2A2F;
    --accent:#5AA3E4; --accent-soft:#172A3B;
    --s1:#3F87CF; --s2:#BC8329; --s3:#A65EA0;
    --pass:#5FBF8B; --pass-bg:#14291F;
    --warn:#D8A445; --warn-bg:#2A2113;
    --fail:#E0796B; --fail-bg:#2C1815;
    --figure-bg:#FFFFFF;
    --shadow:0 1px 2px rgba(0,0,0,.5),0 10px 30px -18px rgba(0,0,0,.9);
  }
}
:root[data-theme="dark"]{
  --ground:#0B1215; --surface:#131C20; --surface-2:#18242A;
  --ink:#E7EEF1; --ink-2:#AABBC3; --ink-3:#7D9099;
  --line:#25353B; --line-soft:#1D2A2F;
  --accent:#5AA3E4; --accent-soft:#172A3B;
  --s1:#3F87CF; --s2:#BC8329; --s3:#A65EA0;
  --pass:#5FBF8B; --pass-bg:#14291F;
  --warn:#D8A445; --warn-bg:#2A2113;
  --fail:#E0796B; --fail-bg:#2C1815;
  --figure-bg:#FFFFFF;
  --shadow:0 1px 2px rgba(0,0,0,.5),0 10px 30px -18px rgba(0,0,0,.9);
}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;font-size:16px;line-height:1.6;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 96px}
h1,h2,h3{font-family:"IBM Plex Serif",Georgia,serif;text-wrap:balance;margin:0}
h1{font-size:clamp(2rem,4.4vw,3.1rem);line-height:1.08;font-weight:600;letter-spacing:-.015em}
h2{font-size:clamp(1.35rem,2.4vw,1.75rem);font-weight:600;letter-spacing:-.01em}
h3{font-size:1.02rem;font-weight:600}
p{margin:0}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.7rem;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink-3)}
.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
header.top{padding:72px 0 40px;display:flex;flex-direction:column;gap:22px}
.thesis{font-family:"IBM Plex Serif",Georgia,serif;font-size:clamp(1.1rem,2vw,1.4rem);line-height:1.45;
  color:var(--ink-2);max-width:64ch;border-left:3px solid var(--accent);padding-left:20px}
.thesis strong{color:var(--ink);font-weight:600}
.runbar{display:flex;flex-wrap:wrap;gap:8px 20px;align-items:center;border-top:1px solid var(--line);
  border-bottom:1px solid var(--line);padding:12px 0;font-size:.8rem;color:var(--ink-3)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(196px,1fr));gap:14px;margin:34px 0 8px}
.stat{background:var(--surface);border:1px solid var(--line-soft);border-radius:10px;padding:18px 18px 16px;
  display:flex;flex-direction:column;gap:6px;box-shadow:var(--shadow)}
.stat .n{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:1.9rem;font-variant-numeric:tabular-nums;
  letter-spacing:-.02em;line-height:1}
.stat .l{font-size:.82rem;color:var(--ink-3);line-height:1.35}
section{margin-top:60px;display:flex;flex-direction:column;gap:18px}
.lede{max-width:68ch;color:var(--ink-2)}
.card{background:var(--surface);border:1px solid var(--line-soft);border-radius:12px;padding:24px;
  box-shadow:var(--shadow)}
figure.fig{margin:0;display:flex;flex-direction:column;gap:12px}
.imgwrap{background:var(--figure-bg);border-radius:8px;padding:10px;overflow-x:auto;
  border:1px solid var(--line-soft)}
.imgwrap img{display:block;width:100%;height:auto;min-width:520px}
figcaption{font-size:.85rem;color:var(--ink-2);max-width:74ch}
.verdicts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.verdict{background:var(--surface);border:1px solid var(--line-soft);border-radius:10px;padding:18px;
  display:flex;flex-direction:column;gap:9px;box-shadow:var(--shadow)}
.chip{display:inline-flex;align-items:center;gap:6px;align-self:flex-start;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.66rem;letter-spacing:.1em;
  text-transform:uppercase;padding:3px 9px;border-radius:999px;font-weight:500}
.chip.pass{color:var(--pass);background:var(--pass-bg)}
.chip.warn{color:var(--warn);background:var(--warn-bg)}
.chip.fail{color:var(--fail);background:var(--fail-bg)}
.verdict p{font-size:.9rem;color:var(--ink-2)}
.tablewrap{overflow-x:auto;border:1px solid var(--line-soft);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.87rem;min-width:460px}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--line-soft);vertical-align:top}
thead th{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.68rem;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink-3);font-weight:500;white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
td.num,th.num{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;
  font-variant-numeric:tabular-nums;white-space:nowrap}
td.k{font-family:"IBM Plex Mono",ui-monospace,monospace;white-space:nowrap;color:var(--ink)}
tr.hl td{background:var(--accent-soft)}
.vids{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}
.vid{background:var(--surface);border:1px solid var(--line-soft);border-radius:11px;overflow:hidden;
  box-shadow:var(--shadow);display:flex;flex-direction:column}
.vid video{width:100%;display:block;background:#000}
.vid .meta{padding:12px 15px 15px;display:flex;flex-direction:column;gap:6px}
.vid .meta p{font-size:.83rem;color:var(--ink-2)}
ul.clean{margin:0;padding-left:1.15rem;display:flex;flex-direction:column;gap:9px;color:var(--ink-2);
  font-size:.92rem;max-width:72ch}
ul.clean strong{color:var(--ink)}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.86em;background:var(--accent-soft);
  padding:1px 5px;border-radius:4px;color:var(--ink)}
pre{margin:0;overflow-x:auto;background:var(--surface-2);border:1px solid var(--line-soft);border-radius:9px;
  padding:14px 16px;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.79rem;line-height:1.55;
  color:var(--ink-2)}
footer{margin-top:72px;padding-top:22px;border-top:1px solid var(--line);font-size:.82rem;color:var(--ink-3);
  display:flex;flex-direction:column;gap:6px}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""

TEMPLATE = """<title>The Token's Ledger</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@600&display=swap">
<style>%%CSS%%</style>
<div class="wrap">

<header class="top">
  <span class="eyebrow">FSQ Wave 2 &middot; overnight 26&ndash;27 August 2026 &middot; MPCDF Viper</span>
  <h1>Where the motion token actually pays</h1>
  <p class="thesis">The token <strong>carries information the reference does not</strong> &mdash; more of it
    as the reference degrades, and none of it lost when the body is randomized. It does not survive a
    second robot. And every arm ever measured here spends its first ten million steps
    <strong>learning to face the wrong way</strong>.</p>
  <div class="runbar mono">
    <span>n&nbsp;=&nbsp;4 rollout seeds minimum</span>
    <span>executed-vs-clip shape RMSE</span>
    <span>H1 unless stated</span>
    <span>scored against a zero-action floor</span>
  </div>
</header>

<div class="stats">
  <div class="stat"><span class="n" style="color:var(--s1)">&minus;4.0%</span>
    <span class="l">token on top of the reference, single body. The effect under test</span></div>
  <div class="stat"><span class="n" style="color:var(--pass)">t&thinsp;&asymp;&thinsp;6.6</span>
    <span class="l">real token vs the same token scrambled. The control that made it a result</span></div>
  <div class="stat"><span class="n" style="color:var(--fail)">0.177</span>
    <span class="l">best body-independent reconstruction, rad. The gate was 0.10 and it refused to spend the slot</span></div>
  <div class="stat"><span class="n" style="color:var(--warn)">15.3&times;</span>
    <span class="l">how much worse than <em>doing nothing</em> every trained arm is at heading</span></div>
</div>

<section>
  <span class="eyebrow">The question you asked</span>
  <h2>Are we close to understanding when FSQ works?</h2>
  <p class="lede">Partly. <strong>We can predict the sign reliably and the size not at all.</strong> Four
    conditions are settled, each with its own control, and together they give a rule you can design
    against.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Condition</th><th>Effect on the token's value</th><th>Where</th></tr></thead>
    <tbody>
      <tr class="hl"><td><b>On top of the reference, not instead of it</b></td>
        <td>the only configuration that ever wins. Token-<em>replacing</em> is 1&ndash;3&nbsp;% worse than
        reference alone, at every M</td><td class="k">motion sweep</td></tr>
      <tr class="hl"><td><b>The reference channel is degraded</b></td>
        <td>value <b>rises</b> &mdash; 4.0&nbsp;% &rarr; 14.0&nbsp;% &rarr; 11.6&nbsp;% as the reference
        drops 40 &rarr; 8 &rarr; 4&nbsp;Hz</td><td class="k">degradation</td></tr>
      <tr><td><b>The body is randomized</b></td>
        <td><b>no effect at all</b> &mdash; &minus;4.0&nbsp;% with and without</td><td class="k">2&times;2</td></tr>
      <tr><td><b>A second topology is present</b></td>
        <td>value <b>collapses to zero</b>, on both robots, with and without randomization</td>
        <td class="k">2&times;2</td></tr>
    </tbody></table></div>
  <p class="lede"><b>The rule:</b> the token is worth having when it <em>supplements</em> a reference that is
    slow, stale or unreliable, on a body family it was fitted to. That is a real engineering claim and it
    is testable downstream.</p>
  <div class="card">
    <h3>What is still not understood &mdash; and why that matters</h3>
    <ul class="clean">
      <li><strong>Why it works at all.</strong> Three explanations are excluded: smoothing and
      extra-observation-width (both by the scramble), and lookahead-as-a-later-reference (the lead arms
      are 16&ndash;79&nbsp;% <em>worse</em>). What survives &mdash; "a better summary of a window than an
      instant sample" &mdash; is a description, not a mechanism.</li>
      <li><strong>Why the size varies.</strong> &minus;5.7&nbsp;%, &minus;8.7&nbsp;%, &minus;2.0&nbsp;%,
      &minus;4.0&nbsp;% at M&nbsp;=&nbsp;1,&nbsp;4,&nbsp;5,&nbsp;9. Not monotone, not explained by how much
      motion there is, and the spread is wider than most of the effects inside it.</li>
      <li><strong>Why a second robot erases it.</strong> Leading candidate: the per-joint code is fitted on
      H1's clip, so G1 reads an address never built for it. Untested &mdash; and it makes a sharp
      prediction, that a token fitted <em>jointly</em> on both bodies restores the gain. That is one
      training arm, and it is the cheapest next step in the whole programme.</li>
    </ul>
    <p class="lede" style="margin-top:14px">So the honest reading is that FSQ is today a
      <strong>conditional engineering win with an unexplained mechanism</strong>, not an understood
      component. The third question above is also the one that decides whether the direction scales past a
      single body family &mdash; which is what the project actually needs to know.</p>
  </div>
</section>

<section>
  <span class="eyebrow">Verdicts</span>
  <h2>What the night settled</h2>
  <div class="verdicts">
    <div class="verdict"><span class="chip pass">Confirmed</span>
      <h3>The token carries information</h3>
      <p>A token stream rolled 7919 frames &mdash; same statistics, invalid phase &mdash; loses the gain and
      then some. Scrambled is <em>worse than no token at all</em>, which is what noise does and the opposite
      of what a regulariser does.</p></div>
    <div class="verdict"><span class="chip pass">Confirmed</span>
      <h3>It pays more as the channel worsens</h3>
      <p>At matched sample rates the token beats the reference delta by 4.0&nbsp;% at 40&nbsp;Hz,
      14.0&nbsp;% at 8&nbsp;Hz, 11.6&nbsp;% at 4&nbsp;Hz. That was the stated gate for this wave.</p></div>
    <div class="verdict"><span class="chip pass">Confirmed</span>
      <h3>Randomized bodies are nearly free</h3>
      <p>Two topologies, five motions, morphology ramping to 0.3: H1 1.8&nbsp;% worse than fixed bodies,
      G1 5.0&nbsp;% <em>better</em>. The token's own effect is unchanged.</p></div>
    <div class="verdict"><span class="chip fail">Closed</span>
      <h3>The body-independent token</h3>
      <p>Decoder width was the right suspect &mdash; 0.2486&nbsp;&rarr;&nbsp;0.2178&nbsp;&rarr;&nbsp;0.1774
      across two doublings, unsaturated, against 6&nbsp;% for a 32&times; codebook and 0&nbsp;% for encoder
      sites. Still 1.77&times; the gate.</p></div>
    <div class="verdict"><span class="chip fail">Closed</span>
      <h3>Three topologies, on this stack</h3>
      <p>Every one- and two-topology combination trains. Every three-topology one aborts with
      <code>ROCM_ERROR_ILLEGAL_ADDRESS</code> &mdash; two env counts, two algorithms, either robot order, and
      a <em>different kernel each time</em>. That is an out-of-bounds write, not a broken kernel.</p></div>
    <div class="verdict"><span class="chip fail">Closed</span>
      <h3>Turning the heading term on</h3>
      <p>Four times the weight buys 10&deg; of heading and costs 18&nbsp;% of joint accuracy, and still sits
      13&times; above the zero-action floor. The term being off was a true fact; the fix that seemed to
      follow from it does not work.</p></div>
  </div>
</section>

<section>
  <span class="eyebrow">Reading the arm names</span>
  <h2>What M1, L0, B2 and the rest mean</h2>
  <p class="lede">Arm names are campaign-local: each overnight run coined its own prefix, and the same letter
    means different things in different campaigns. This is the decoder ring for the names on this page.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Name</th><th>Reads as</th><th>Campaign</th></tr></thead>
    <tbody>
      <tr class="hl"><td class="k">M&lt;K&gt;_&lt;channel&gt;</td>
        <td><b>M = how many motions are in the training clip.</b> <code>M1</code> one dance,
        <code>M4</code> four, <code>M5</code> five, <code>M9</code> nine. The channel suffix is
        <code>ref</code> (explicit reference only), <code>z</code> (FSQ token <em>replacing</em> the
        reference), <code>both</code> (reference <em>and</em> token), or <code>canon</code> (one shared
        body-independent stream for all robots).</td><td>FSQ-SCALE, Wave&nbsp;2</td></tr>
      <tr><td class="k">_2t</td><td>two topologies (H1 + G1) sharing one policy, instead of one</td>
        <td>FSQ-SCALE</td></tr>
      <tr><td class="k">_h&lt;N&gt;</td><td>the token is <b>held</b> for N clip frames &mdash; the rate knob.
        The clip is 40&nbsp;fps, so <code>_h5</code> is 8 tokens/s</td><td>FSQ-SCALE</td></tr>
      <tr><td class="k">_scram</td><td>the scramble <b>control</b>: same token stream rolled 7919 frames, so
        its contents no longer describe the frames they accompany</td><td>Wave&nbsp;2</td></tr>
      <tr><td class="k">_morph</td><td>morphology randomization on (limb lengths, masses, gains), ramping to
        0.3</td><td>Wave&nbsp;2</td></tr>
      <tr><td class="k">_lead&lt;N&gt;</td><td>the <em>observed</em> reference is shifted N frames into the
        future while the reward target stays at t &mdash; the raw-lookahead control</td><td>Wave&nbsp;2</td></tr>
      <tr><td class="k">_head&lt;W&gt;</td><td>heading reward weight ratio W (05 = 0.5, 20 = 2.0)</td>
        <td>Wave&nbsp;2</td></tr>
      <tr class="hl"><td class="k">L0 / L1 / L2</td>
        <td><b>rungs of generality</b>, each adding one demand: L0 a nominal body, L1 randomized bodies,
        L2 a second topology on top</td><td>LADDER</td></tr>
      <tr><td class="k">B0 / B1 / B2 / B100</td>
        <td>baseline <b>recipes</b> under test, not difficulty levels &mdash; different combinations of
        reference bias, deviation tolerance and termination. <code>B100</code> is the recipe the
        22-08 dance baseline was built on</td><td>baseline-fix</td></tr>
      <tr><td class="k">G0 / G1</td><td><b>gates</b> &mdash; go/no-go checks run before spending compute
        (G0 checked the reference itself, G1 the policy's complaint)</td><td>baseline-fix</td></tr>
      <tr><td class="k">C1 / C2 / C3</td><td>the three <b>questions</b> of the FSQ-SCALE night: does the
        token match the reference (C1), can one shared stream drive several bodies (C2), how slow can the
        token get (C3)</td><td>FSQ-SCALE</td></tr>
      <tr><td class="k">X_&lt;knob&gt;</td><td>single-knob probes &mdash; one setting changed from the
        baseline and nothing else, e.g. <code>X_temp005</code> is tracking temperature 0.05</td>
        <td>baseline-fix</td></tr>
      <tr><td class="k">scale_&lt;N&gt;t</td><td>compute-scaling arms: how many topologies share the
        policy</td><td>Wave&nbsp;2</td></tr>
    </tbody></table></div>
  <p class="lede">Two conventions hold across every campaign: an arm named <code>*_ref</code> is always the
    control the token arms are measured against, and <b>zero action</b> &mdash; a policy that emits no
    torque &mdash; is the floor under everything. A tracking number without that denominator has repeatedly
    turned out to mean nothing.</p>
</section>

<section>
  <span class="eyebrow">The finding that outranks the rest</span>
  <h2>Every arm learns to face the wrong way</h2>
  <p class="lede">Until this run the crosseval scored joint angles and feet and nothing about where the robot
    pointed. The quantity was being computed inside the render branch and thrown away &mdash; and it was
    also being written to the training log of every run this project has ever done.</p>
  %%FIG_HEADING%%
  %%FIG_HEADBAR%%
  %%FIG_SELLOFF%%
  <div class="card">
    <h3>The cause is one line. The fix that follows from it does not work.</h3>
    <pre>default_config.py:457   "root_heading_tracking_weight_ratio": 0.0,
viper_train.sbatch:144  ...weight_ratio="${HEADING_RATIO:-0.0}"
default_config.py:461   "deepmimic_heading_free": True,</pre>
    <p class="lede" style="margin-top:14px">The explicit heading term was off in every arm, and
      <code>deepmimic_heading_free</code> <em>removes</em> the implicit anchor the DeepMimic pose terms used
      to carry &mdash; its own comment says "this is where heading is supposed to be scored instead". Anchor
      removed, replacement at weight zero: no reward term is a function of where the robot faces.</p>
    <p class="lede" style="margin-top:12px">But switching it on does not repair the behaviour. At four times
      the weight, heading improves from 82&deg; to 72&deg; while joint accuracy gets 18&nbsp;% worse &mdash;
      and the curve above shows why: the arms with the term on climb to 84&deg; on the same schedule as the
      arms without it. <strong>The policy is not failing to acquire heading; it acquires the task by giving
      heading up</strong>, in the first tenth of training. Finding what rotating buys it is a reward-shaping
      question, and it is now the most valuable open item in the project &mdash; it sits underneath every
      number here.</p>
  </div>
</section>

<section>
  <span class="eyebrow">Method</span>
  <h2>Why none of this is read off a training curve</h2>
  %%FIG_BLIND%%
  %%FIG_CANON%%
</section>

<section>
  <span class="eyebrow">The load-bearing experiment</span>
  <h2>Take the reference away and watch who copes</h2>
  %%FIG_DEGRADE%%
  <div class="tablewrap"><table>
    <thead><tr><th>K</th><th class="num">reference only</th><th class="num">+ token (fresh)</th>
      <th class="num">+ token (matched)</th><th class="num">token gain, matched</th></tr></thead>
    <tbody>
      <tr><td class="k">1</td><td class="num">0.1315</td><td class="num">0.1263</td>
        <td class="num">0.1263</td><td class="num">4.0%</td></tr>
      <tr class="hl"><td class="k">5</td><td class="num">0.2065</td><td class="num">0.1484</td>
        <td class="num">0.1776</td><td class="num">14.0%</td></tr>
      <tr><td class="k">10</td><td class="num">0.2847</td><td class="num">0.1740</td>
        <td class="num">0.2516</td><td class="num">11.6%</td></tr>
      <tr><td class="k">20</td><td class="num">0.3549</td><td class="num">0.2126</td>
        <td class="num">0.3376</td><td class="num">4.9%</td></tr>
    </tbody></table></div>
  <p class="lede">The fresh-token column overstates the token: it keeps a per-frame channel the reference has
    lost. The matched column is the fair comparison &mdash; and the token still wins there, by more than it
    does when both channels are fresh.</p>
</section>

<section>
  <span class="eyebrow">Scope</span>
  <h2>Where the effect holds, and where it stops</h2>
  %%FIG_MSWEEP%%
  %%FIG_2X2%%
  %%FIG_RATE%%
</section>

<section>
  <span class="eyebrow">The direction that closed</span>
  <h2>One token stream for every body</h2>
  %%FIG_DECODER%%
</section>

<section>
  <span class="eyebrow">Earlier campaigns</span>
  <h2>What each step of generality has cost</h2>
  %%FIG_LADDER%%
</section>

<section>
  <span class="eyebrow">Watch it</span>
  <h2>Policy beside the reference it was scored against</h2>
  <p class="lede">Left pane is the policy as it actually moved; right is the clip, at its own root pose.
    Rendered locally &mdash; compute nodes have no OSMesa and no <code>/dev/dri</code>, so only the rollout
    dump happens on the cluster. Press play; they are muted and loop.</p>
  <div class="card">
    <h3>Two things to know before you watch, one of which was a bug in these videos</h3>
    <ul class="clean">
      <li><strong>Fixed: the reference used to be dragged around by the policy.</strong> The first version
      of this page rendered the reference at the <em>policy's</em> root position with a yaw-only
      orientation, so it slid across the floor without walking and its torso never leaned. That was a
      drawing artefact, not the reference: the rollout dump already carried
      <code>reference_root</code>, the clip's own root pose, and the renderer simply never read it. The
      panes below use it, so the reference now travels under its own feet and can lean.</li>
      <li><strong>The H1 and G1 panes are not the same moment of the same dance.</strong> Every
      environment starts at a random clip phase, drawn independently per robot, so the two robots are
      showing <em>different segments</em> of the five-dance super-clip. Cross-correlating the two
      reference root paths puts them 281 and &minus;630 frames apart at correlations of 0.34 and 0.41
      &mdash; different parts of the motion, not a mismatch between the robots. The reference heights also
      differ because the robots do: <b>1.02&nbsp;m for H1, 0.80&nbsp;m for G1</b>. Neither is a G1 defect.</li>
    </ul>
    <p class="lede" style="margin-top:14px">One thing the corrected renders do make visible, and it is
      real: <strong>the policies travel much further than the clip asks.</strong> On
      <code>M9_ref</code> the policy covers 2.92&nbsp;m where the reference covers 1.25&nbsp;m, and on
      <code>M9_z</code> 2.74&nbsp;m. That is the same root-and-heading failure the heading figure above
      quantifies, seen from the side.</p>
  </div>
  <div class="vids">
%%VIDEOS%%
  </div>
</section>

<section>
  <span class="eyebrow">Harness</span>
  <h2>Six defects, each of which had already cost an arm</h2>
  <ul class="clean">
    <li><strong>The decoder width knob never worked.</strong> Two hardcoded <code>256</code>s in the vendored
      decoder while the joint mask scaled. At multiplier 1.0 they coincide; at anything else the broadcast
      fails. Decoder capacity had therefore never been tested at any width.</li>
    <li><strong>The trainer reported success for runs that died.</strong> The script ended on
      <code>echo "TRAINING EXITED rc=$?"</code>, so its exit status was <em>echo's</em>. Every
      <code>afterok</code> chain behind a training arm was unprotected.</li>
    <li><strong>Minibatch 8192 is not divisible by 3.</strong> The first thing between us and a third
      topology was an assertion about arithmetic, not memory.</li>
    <li><strong>The clip map had no <code>t1</code> entry</strong> &mdash; and the robot's own name is
      <code>t1</code>, not <code>booster_t1</code>. The clip was already on the cluster.</li>
    <li><strong>Heading was computed and discarded.</strong> See above.</li>
    <li><strong>Command substitution ran on the wrong machine.</strong> Driving the cluster through nested
      shells evaluated <code>$(...)</code> locally, so <code>squeue</code> worked while <code>sbatch</code>
      reported "command not found" &mdash; with the binary present and executable.</li>
  </ul>
  <div class="card">
    <h3>And a screen that pays for itself</h3>
    <p class="lede"><code>derive_clip_signs.py</code> recovers a family's joint-axis sign table from the clip
      alone, by asking which signs make the simulator's own forward kinematics reproduce the poses the clip
      recorded. It reproduces all three known-good tables exactly. Orientation is mandatory in the objective:
      scored on body <em>positions</em> only it hits a 0.0000&nbsp;cm residual and still gets 2&ndash;4 joints
      wrong per family &mdash; always the terminal ones, whose child body sits on the joint axis, where
      flipping the sign moves nothing.</p>
    <p class="lede" style="margin-top:12px">Run on the four families uploaded to extend the topology count,
      it rejected three &mdash; atlas 0.043, talos 0.83, toddlerbot 2.40 against ~1.5e&minus;4 for the good
      three &mdash; before a single GPU-hour was spent. The honest data ceiling is three families, and that
      limit is independent of the trainer's.</p>
  </div>
</section>

<footer>
  <span>Full numbers and method: <code>experiments/fsq_khaendler/REPORT_FSQ_WAVE2.md</code>. Ideas opened and
    closed: <code>docs/notes/FSQ_NEXT_WAVE.md</code>. Figures: <code>plot_wave2.py</code> over
    <code>curves_all.csv</code> (365&thinsp;007 logged updates across 300 arms).</span>
  <span>Every figure is a mean over at least four rollout seeds. One seed alone moves an arm by up to
    4.95&nbsp;% &mdash; larger than most of the effects argued about here.</span>
</footer>

</div>
"""


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
