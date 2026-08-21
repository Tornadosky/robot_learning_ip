"""C33 -- build the review dashboard: embedded videos + exact provenance.

Reads the WebP clips rendered by c32 and emits a single self-contained HTML file
with each clip inlined as a data URI, alongside a precise statement of what was
actually used: which reference, which retargeting, which randomizations, whose
code, whether URMA was involved, and which parts are stock loco-mujoco versus
written here.

The page is for judging the videos, so every clip carries the one comparison
that makes it judgeable: the robot's mean root error against how far the
reference itself travels.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]

CARDS = {
    "ref_nominal_dance": dict(
        title="Dance reference · nominal H1",
        kind="reference",
        blurb="Kinematic playback of the reference the reward scores. The robot is placed frame by frame; no policy, no physics. Spheres sit on the robot because the reference is its own target.",
        watch="Does this look like the dance it should be? The feet barely settle — this clip has a true stance foot in only 11% of frames.",
    ),
    "ref_body04_dance": dict(
        title="Dance reference · randomized body 04",
        kind="reference",
        blurb="Same motion re-grounded onto the most extreme catalog body (legs 0.85×, arms 0.90×, shoulders 0.90×, torso mass 1.01×).",
        watch="Compare limb proportions with the nominal clip. The reference is rebuilt per body; contact quality is unchanged (stance spread across bodies is 0.02).",
    ),
    "policy_nominal_dance": dict(
        title="Trained policy · nominal H1 · dance",
        kind="policy",
        blurb="Deterministic rollout of a PPO policy trained 20M steps on this body against its own IK-retargeted reference.",
        watch="The joints track well. Watch the spheres separate from the robot — it performs the dance while leaving the place the dance happens.",
    ),
    "policy_body04_dance": dict(
        title="Trained policy · randomized body 04 · dance",
        kind="policy",
        blurb="Same training recipe on the extreme body, against its FK-constructed re-grounded reference.",
        watch="Survives the full 8 s window. The gap to the spheres is the failure mode this whole audit converged on.",
    ),
    "ref_nominal_walk": dict(
        title="Walk reference · nominal H1",
        kind="reference",
        blurb="The contact-consistent counterexample: 91.6% of frames have a genuine stance foot, against the dance's 11%.",
        watch="Feet plant and stay planted. This is what a reference the robot could physically follow looks like.",
    ),
    "policy_nominal_walk": dict(
        title="Trained policy · nominal H1 · walk",
        kind="policy",
        blurb="Same recipe, same budget, on the contact-consistent walk clip.",
        watch="The reference walks ~10 m in 8 s. Watch whether the robot goes with it.",
    ),
    "multibody_retargeted": dict(
        title="ONE policy · 6 randomized bodies · REFERENCE RETARGETED PER BODY",
        kind="multibody",
        blurb="The fix. Until now the multi-body trainer scored every randomized body against the NOMINAL body's site targets, because MimicReward reads them from a trajectory whose site data was computed once on the nominal model. `MorphMimicReward` recomputes them by forward kinematics on each environment's own morphology arrays — in-graph, no dataset, works for continuously sampled bodies. A 1.18x-arm body is now told to put its hands 1.190 m apart instead of 1.064 m.",
        watch="The spheres now DIFFER per panel: hand-target span reads 0.955 m on the 0.86x-arm body up to 1.136 m on the 1.18x-arm body — 18 cm across the six. Each panel prints its own span. Site error against body-correct targets: 9.11 → 7.76 cm (−15%), seed spread 0.56 → 0.02 cm. Root drift is unchanged — separate problem.",
    ),
    "multibody_nodev": dict(
        title="ONE policy · 6 randomized bodies · dance",
        kind="multibody",
        blurb="The same setup BEFORE the retargeting fix: one MLP over continuous morphology, but every body scored against the nominal body's site targets. For the ±20% arm-scale bodies here those targets are up to 21 cm wrong.",
        watch="Every panel here shows the SAME nominal targets, whatever the body's proportions. That is what the multi-body trainer did for the whole audit — per-body retargeting existed only in the single-body arms.",
    ),
    "multibody_dev05": dict(
        title="ONE policy · 6 bodies · with deviation termination",
        kind="multibody",
        blurb="Identical training except the episode now ends when the root strays more than 0.5 m from the reference — a mechanism that was already implemented upstream but disabled by a 1e6 default.",
        watch="Compare the robot-to-sphere gap with the panel above. Mean root error moved 0.66 → 0.69 m across panels: no better. The mechanism is not the missing piece.",
    ),
    "locomotion_baseline": dict(
        title="Locomotion baseline · no imitation",
        kind="control",
        blurb="Not an imitation policy: a plain velocity-command controller (RLFactory + GoalRandomRootVelocity + LocomotionReward), same robot, same torque actuators, same PPO, 20M steps. No reference, so no spheres. Undertrained by locomotion standards — it falls after ~1.6 s — but the question was only whether translation is possible here.",
        watch="This one translates — 0.85 m of directed motion in 1.6 s before falling. Path length 0.87 m against 0.85 m net, so it is walking in a line, not flailing. It proves the stack CAN move the robot, so the imitation failure is not the simulator or the actuators.",
    ),
}

ORDER = ["multibody_retargeted", "multibody_nodev", "multibody_dev05",
         "ref_nominal_dance", "policy_nominal_dance", "ref_body04_dance",
         "policy_body04_dance", "ref_nominal_walk", "policy_nominal_walk",
         "locomotion_baseline"]


def data_uri(path: Path) -> str:
    return "data:image/webp;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def verdict(rec: dict) -> tuple[str, str]:
    """(state, text) — the judgement that makes a clip scannable."""
    if rec.get("mode") == "reference":
        return "ok", "reference · exact by construction"
    if rec.get("label") == "locomotion_baseline":
        return "warn", "moves 0.85 m, then falls"
    if rec.get("label") == "multibody_retargeted":
        return "ok", "retargeted · site 7.76 cm (was 9.11)"
    if rec.get("label") == "multibody_nodev":
        return "warn", "shared nominal targets · site 9.11 cm"
    if rec.get("label") == "multibody_dev05":
        return "warn", "deviation termination on · 0.69 m, no gain"
    err, travel = rec.get("mean_root_error_m"), rec.get("reference_travel_m")
    if err is None:
        return "warn", "no metric"
    ratio = err / travel if travel else float("inf")
    state = "ok" if ratio < 0.5 else "bad"
    return state, f"root error {err:.2f} m · reference travels {travel:.2f} m"


def build(manifest: dict, video_dir: Path) -> str:
    by_label = {r["label"]: r for r in manifest["videos"]}
    for extra in ("multibody_retargeted", "multibody_nodev", "multibody_dev05"):
        by_label.setdefault(extra, {"label": extra, "mode": "multibody",
                                    "clip": "dance2_subject4", "body": "6 sampled bodies"})
    cards = []
    for label in ORDER:
        meta = CARDS[label]
        path = video_dir / f"{label}.webp"
        if not path.exists():
            continue
        rec = by_label.get(label, {"label": label, "mode": meta["kind"]})
        rec["label"] = label
        state, vtext = verdict(rec)
        steps = rec.get("steps")
        sub = []
        if rec.get("clip"):
            sub.append(rec["clip"])
        if rec.get("body"):
            sub.append(rec["body"].replace("body00_nominal", "nominal").replace("_seed", " seed "))
        if steps:
            sub.append(f"{steps} steps")
        cards.append(f"""
      <article class="card">
        <figure>
          <img src="{data_uri(path)}" alt="{meta['title']}" loading="lazy" />
        </figure>
        <div class="card-body">
          <div class="card-head">
            <h3>{meta['title']}</h3>
            <span class="chip chip-{state}">{vtext}</span>
          </div>
          <p class="meta">{' · '.join(sub) if sub else '&nbsp;'}</p>
          <p>{meta['blurb']}</p>
          <p class="watch"><span>Look for</span> {meta['watch']}</p>
        </div>
      </article>""")

    return TEMPLATE.replace("{{CARDS}}", "\n".join(cards))


TEMPLATE = r"""<title>H1 DeepMimic — video review</title>
<style>
  :root{
    --bg:#F5F7F4; --surface:#FFFFFF; --surface-2:#EDF1EC;
    --ink:#171C19; --muted:#666E69; --line:#DCE3DD;
    --accent:#B06F12; --accent-soft:#F3E5CE;
    --ok:#2F6D50; --ok-soft:#DDEDE4;
    --bad:#A03A34; --bad-soft:#F6DFDC;
    --warn:#8A6D1F; --warn-soft:#F2E9D2;
    --mono:ui-monospace,"Cascadia Mono","SF Mono",Menlo,Consolas,monospace;
    --sans:ui-sans-serif,system-ui,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  @media (prefers-color-scheme:dark){
    :root:not([data-theme="light"]){
      --bg:#101410; --surface:#181D19; --surface-2:#1F2621;
      --ink:#E7EBE6; --muted:#97A099; --line:#2B322D;
      --accent:#DFA455; --accent-soft:#38301F;
      --ok:#77C39A; --ok-soft:#1D3129;
      --bad:#E28880; --bad-soft:#361F1D;
      --warn:#D9BC6A; --warn-soft:#332C19;
    }
  }
  :root[data-theme="dark"]{
    --bg:#101410; --surface:#181D19; --surface-2:#1F2621;
    --ink:#E7EBE6; --muted:#97A099; --line:#2B322D;
    --accent:#DFA455; --accent-soft:#38301F;
    --ok:#77C39A; --ok-soft:#1D3129;
    --bad:#E28880; --bad-soft:#361F1D;
    --warn:#D9BC6A; --warn-soft:#332C19;
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--bg); color:var(--ink);
    font-family:var(--sans); font-size:16px; line-height:1.6;
    -webkit-font-smoothing:antialiased;
  }
  .wrap{max-width:1180px; margin:0 auto; padding:clamp(24px,4vw,56px) clamp(16px,4vw,40px) 80px}
  header.top{display:flex; flex-direction:column; gap:10px; padding-bottom:22px; border-bottom:2px solid var(--ink)}
  .eyebrow{font-family:var(--mono); font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--accent)}
  h1{margin:0; font-size:clamp(28px,4.4vw,42px); line-height:1.1; letter-spacing:-.02em; text-wrap:balance; font-weight:650}
  .lede{margin:0; max-width:62ch; color:var(--muted); font-size:17px}
  h2{margin:0 0 4px; font-size:20px; letter-spacing:-.01em; font-weight:640}
  section{margin-top:44px}
  .section-label{font-family:var(--mono); font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin-bottom:14px}

  .spec{display:grid; grid-template-columns:minmax(150px,210px) 1fr; gap:0; border:1px solid var(--line); border-radius:3px; overflow:hidden; background:var(--surface)}
  .spec dt{padding:12px 16px; font-family:var(--mono); font-size:12px; letter-spacing:.04em; color:var(--muted); border-bottom:1px solid var(--line); background:var(--surface-2)}
  .spec dd{margin:0; padding:12px 16px; border-bottom:1px solid var(--line); font-size:15px}
  .spec dt:last-of-type,.spec dd:last-of-type{border-bottom:none}
  .spec code{font-family:var(--mono); font-size:13px; background:var(--surface-2); padding:1px 5px; border-radius:2px}
  .tag{display:inline-block; font-family:var(--mono); font-size:11px; letter-spacing:.06em; text-transform:uppercase; padding:2px 7px; border-radius:2px; margin-right:6px}
  .tag-yes{background:var(--ok-soft); color:var(--ok)}
  .tag-no{background:var(--bad-soft); color:var(--bad)}
  .tag-part{background:var(--warn-soft); color:var(--warn)}

  .legend{display:flex; flex-wrap:wrap; gap:8px 20px; align-items:center; padding:14px 16px; border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:3px; background:var(--surface); font-family:var(--mono); font-size:12.5px}
  .dot{display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:7px; vertical-align:-1px; border:1px solid rgba(0,0,0,.18)}

  .grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:22px}
  .card{background:var(--surface); border:1px solid var(--line); border-radius:4px; overflow:hidden; display:flex; flex-direction:column}
  .card figure{margin:0; background:var(--surface-2); border-bottom:1px solid var(--line)}
  .card img{display:block; width:100%; height:auto}
  .card-body{padding:16px 18px 18px; display:flex; flex-direction:column; gap:9px}
  .card-head{display:flex; flex-wrap:wrap; gap:8px; align-items:baseline; justify-content:space-between}
  .card h3{margin:0; font-size:16.5px; font-weight:640; letter-spacing:-.01em}
  .chip{font-family:var(--mono); font-size:11px; letter-spacing:.03em; padding:3px 8px; border-radius:2px; white-space:nowrap; font-variant-numeric:tabular-nums}
  .chip-ok{background:var(--ok-soft); color:var(--ok)}
  .chip-bad{background:var(--bad-soft); color:var(--bad)}
  .chip-warn{background:var(--warn-soft); color:var(--warn)}
  .card p{margin:0; font-size:14.5px}
  .meta{font-family:var(--mono); font-size:11.5px; color:var(--muted); letter-spacing:.03em}
  .watch{padding-top:9px; border-top:1px dashed var(--line); color:var(--muted); font-size:14px}
  .watch span{font-family:var(--mono); font-size:10.5px; letter-spacing:.13em; text-transform:uppercase; color:var(--accent); margin-right:7px}

  .callout{border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:3px; background:var(--surface); padding:18px 20px}
  .callout p{margin:0 0 10px; max-width:70ch}
  .callout p:last-child{margin-bottom:0}
  .scroll{overflow-x:auto}
  table{border-collapse:collapse; width:100%; font-size:14px; background:var(--surface)}
  th,td{text-align:left; padding:9px 14px; border-bottom:1px solid var(--line)}
  th{font-family:var(--mono); font-size:11px; letter-spacing:.09em; text-transform:uppercase; color:var(--muted); font-weight:500}
  td.num{font-family:var(--mono); font-variant-numeric:tabular-nums}
  footer{margin-top:52px; padding-top:18px; border-top:1px solid var(--line); color:var(--muted); font-size:13px; font-family:var(--mono)}
  a{color:var(--accent)}
  @media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
</style>

<div class="wrap">
  <header class="top">
    <div class="eyebrow">Unitree H1 · DeepMimic · randomized morphology</div>
    <h1>Video review: what the policies actually do</h1>
    <p class="lede">Seven clips. Every sphere is a target the reward genuinely scores, read from the same array
      <code>MimicReward</code> indexes at the same frame — nothing is reconstructed for display. Each card carries the
      one number that makes it judgeable: how far the robot is from where the reference is, against how far the
      reference itself travels.</p>
  </header>

  <section>
    <div class="section-label">Marker legend</div>
    <div class="legend">
      <span><span class="dot" style="background:#F2D933"></span>upper body</span>
      <span><span class="dot" style="background:#33BFF2"></span>left hand</span>
      <span><span class="dot" style="background:#1A73E6"></span>right hand</span>
      <span><span class="dot" style="background:#F2734D"></span>left foot</span>
      <span><span class="dot" style="background:#D93340"></span>right foot</span>
      <span style="color:var(--muted)">camera frames robot + targets together, widening as they separate</span>
    </div>
  </section>

  <section>
    <div class="section-label">The clips</div>
    <div class="grid">
{{CARDS}}
    </div>
  </section>

  <section>
    <div class="section-label">Exactly what was used</div>
    <dl class="spec">
      <dt>Motion source</dt>
      <dd>LAFAN1 mocap, <strong>already retargeted to H1 upstream</strong> by loco-mujoco and shipped on HuggingFace
        (<code>robfiras/loco-mujoco-datasets</code>). Loaded via <code>LAFAN1DatasetConf</code>; the trajectory handler
        resamples it to the env's 100 Hz. I did not retarget human→robot myself.</dd>

      <dt>Reference window</dt>
      <dd>Dance: <code>dance2_subject4</code>, frames <code>19482–20282</code> @100 Hz = 8.00 s, content-hashed.
        Walk: <code>walk1_subject1</code>, frames <code>10521–11321</code>. Windows are defined in <em>seconds</em>
        because the same clip exists at 40 Hz and 100 Hz in this repo and the cache key does not record which.</dd>

      <dt>Robot→robot retargeting</dt>
      <dd><span class="tag tag-no">not GMR</span><span class="tag tag-no">not SMPL</span>
        <strong>Multi-body runs now use <code>MorphMimicReward</code></strong>, which recomputes the site
        targets by forward kinematics on each environment's own morphology arrays every step. Before that fix
        the multi-body trainer used the nominal body's targets for every body; the per-body constructions below
        applied only to the single-body arms.
        <br />Two constructions, both mine. <strong>fk</strong>: nominal joint angles applied to the variant body, root
        re-grounded per frame from that body's own forward kinematics. <strong>ik_scaled</strong>: batched
        Gauss-Newton IK on <code>mjx.kinematics</code> — 19 joints + root xyz, 4 site targets, 8 iterations,
        targets scale-normalized by the carrying limb, then clamped to joint limits and re-grounded.
        <br />I deliberately avoided loco-mujoco's SMPL robot→robot path (1–9 s per body-clip); the GN kernel runs at
        <strong>165k body-frames/s</strong>, i.e. 1000 bodies × 3000 frames in ~18 s.</dd>

      <dt>Body randomization</dt>
      <dd>Four scalars — leg length, arm length, shoulder width, torso mass — bounds
        <code>[0.85, 1.20]</code> for the three lengths and <code>[0.70, 1.50]</code> for torso mass.
        Applied two ways that were verified equivalent to <strong>sub-micron</strong> on the reward's sites:
        an XML generator (edits <code>MjSpec</code> body pos/ipos/mass/inertia/sites <em>and meshes</em>) for
        retargeting and rendering, and dynamic MJX array patching (same arrays, <em>meshes left nominal</em>) for
        training, so body count never multiplies XLA compiles.</dd>

      <dt>Whose code</dt>
      <dd><span class="tag tag-no">not Nico's loco_mjx / URMA2</span>
        Everything here runs on <strong>loco-mujoco</strong> (robfiras) plus this repository's own
        <code>scripts/scaling</code> multi-body trainer. Nico Bohlinger's <code>loco_mjx</code> stack was not used
        in any run on this page.</dd>

      <dt>URMA</dt>
      <dd><span class="tag tag-part">tested, rejected</span>
        This repo's own URMA implementation (<code>scripts/scaling/urma_networks.py</code>), both v1 and v2, compared
        at matched 20M steps against a plain MLP on continuous morphology. MLP 382 episode length @68k steps/s;
        urma 67 @39k; urmav2 69 @32k. v2 buys nothing over v1. <strong>No policy on this page uses URMA.</strong></dd>

      <dt>DeepMimic implementation</dt>
      <dd><span class="tag tag-yes">stock kernels</span>
        <code>MimicReward</code>, <code>GoalTrajMimic</code>, <code>PPOJax</code> MLP, RSI, termination and the
        trajectory handler are upstream loco-mujoco, <strong>unmodified</strong>. Vendored code was not patched.</dd>

      <dt>What is custom</dt>
      <dd>Reward-weight presets (configuration only, defaults unchanged); a <code>GoalTrajMimicRootErr</code> subclass
        that appends the local-frame root position error; the body and reference adapters; the mocap-sphere target
        overlay — needed because <em>both</em> <code>GoalTrajMimic</code> and <code>GoalTrajMimicv2</code> crash at
        render on MuJoCo 3.9 (<code>mjv_initGeom</code> type error).</dd>

      <dt>Training recipe</dt>
      <dd>PPO, MLP [512, 256], 512 envs × 100 steps, <strong>20M steps</strong> per policy (~5 min on the local
        RTX 4060 Ti), lr 1e-4, reward weights <code>qpos 0.4 / qvel 0.2 / rpos 0.5 / rquat 0.3 / rvel 0.1</code>.
        Single seed per video; the numeric claims elsewhere use n=3.</dd>
    </dl>
  </section>

  <section>
    <div class="section-label">The one result that matters for judging these</div>
    <div class="callout">
      <p><strong>No configuration tracks the root better than a robot that does not move.</strong> Freeze the robot at
        its reset phase and it scores 0.47 m of error on the dance and 3.56 m on the walk. Every trained policy is
        worse than that — by 30–85% — across four reward variants, budgets to 100M steps, two clips, and a custom
        observation that exposes the root error directly.</p>
      <p>Joint tracking, meanwhile, is good: per-term joint reward 0.60–0.71, episode length 390–450 of 800,
        2–14× better than zero-action on survival. That combination is exactly what the videos show — a convincing
        dance performed in the wrong place.</p>
      <p>The locomotion baseline is the control that narrows this: with no imitation at all, the same robot and
        actuators translate at 0.61 m/s in a straight line. So the failure lives in the imitation configuration, not the simulator.</p>
    </div>
  </section>

  <section>
    <div class="section-label">Numbers behind the chips</div>
    <div class="scroll">
      <table>
        <thead><tr><th>Clip</th><th>Stance fraction</th><th>Reference travel</th><th>Stand-still floor</th><th>Best policy error</th></tr></thead>
        <tbody>
          <tr><td>dance2_subject4 <em>(the frozen clip)</em></td><td class="num">0.109</td><td class="num">0.66 m</td><td class="num">0.47 m</td><td class="num">0.61 m</td></tr>
          <tr><td>dance2_subject3</td><td class="num">0.615</td><td class="num">1.93 m</td><td class="num">—</td><td class="num">not run</td></tr>
          <tr><td>walk1_subject1</td><td class="num">0.916</td><td class="num">10.34 m</td><td class="num">3.56 m</td><td class="num">6.21 m</td></tr>
        </tbody>
      </table>
    </div>
    <p style="color:var(--muted); font-size:14px; margin-top:10px; max-width:70ch">Stance = a foot within 5 mm of the
      floor <em>and</em> moving under 10 cm/s, after per-frame re-grounding. It costs seconds per clip, needs no GPU,
      is insensitive to morphology (spread 0.020 across ±20% limb scaling), and predicted every training outcome here.
      The frozen clip is the least suitable of the ten available.</p>
  </section>

  <footer>experiments/h1_morphology_deepmimic_20260808 · local RTX 4060 Ti · loco-mujoco 3921fed · MuJoCo 3.9.0 · JAX 0.10.1</footer>
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=WORKSPACE / "experiments" / "h1_morphology_deepmimic_20260808" / "videos" / "web" / "manifest.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    html = build(manifest, args.manifest.parent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({mb:.2f} MB)")
    if mb > 15:
        print("WARNING: over the 16 MB artifact limit")


if __name__ == "__main__":
    main()
