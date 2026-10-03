"""Rebuild the historical report from local media; never overwrite an artifact."""
import argparse
import base64
from pathlib import Path

HISTORICAL_NOTICE = (
    '<p role="note">Historical July 2026 report text. Numbers and conclusions '
    'are preserved from that report, not recalculated or independently validated '
    'by this builder.</p>\n'
)


def main():
    repo = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=repo / "foveation_results")
    parser.add_argument("--output", type=Path, default=repo / "foveation_results/rebuilt_reports/stability_memory_report.historical.html")
    args = parser.parse_args()
    output = args.output.expanduser()
    if output.exists():
        parser.error(f"refusing to replace an existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    R = args.input_root.expanduser()
    def b64(path):
        with open(path,"rb") as f: return base64.b64encode(f.read()).decode()
    def img(path,mime="image/png"): return f"data:{mime};base64,{b64(path)}"

    A_static = img(f"{R}/tref_stability/constant/tref_stability_static.png")
    A_anim   = f"data:video/mp4;base64,{b64(f'{R}/tref_stability/constant/tref_stability_anim.mp4')}"
    F1 = img(f"{R}/savings/formation_c0.1/formation.png")
    F2 = img(f"{R}/savings/formation_c0.5/formation.png")
    TS = img(f"{R}/savings/timescale/timescale.png")

    HTML = f"""<style>
    :root{{
      --paper:#f7f4ec; --ink:#14161f; --ink-soft:#3a3d4c;
      --panel:#fffdf8; --line:#e2ddd0; --muted:#6a6656;
      --amber:#c9781f; --amber-soft:#e9c48d; --teal:#2f8f83; --plum:#5b4b8a;
      --good:#2f8f5b; --bad:#b23b3b;
      --measure:68ch;
      --serif:ui-serif,Georgia,"Iowan Old Style","Times New Roman",serif;
      --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
      --mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,monospace;
    }}
    @media (prefers-color-scheme:dark){{
      :root{{
        --paper:#101219; --ink:#eae6da; --ink-soft:#c3bfb2;
        --panel:#171a23; --line:#282c38; --muted:#8b8778;
        --amber:#e0952f; --amber-soft:#6a541f; --teal:#4bb7a8; --plum:#a493d6;
        --good:#5cc48a; --bad:#e07a7a;
      }}
    }}
    :root[data-theme="light"]{{
      --paper:#f7f4ec; --ink:#14161f; --ink-soft:#3a3d4c; --panel:#fffdf8;
      --line:#e2ddd0; --muted:#6a6656; --amber:#c9781f; --amber-soft:#e9c48d;
      --teal:#2f8f83; --plum:#5b4b8a; --good:#2f8f5b; --bad:#b23b3b;
    }}
    :root[data-theme="dark"]{{
      --paper:#101219; --ink:#eae6da; --ink-soft:#c3bfb2; --panel:#171a23;
      --line:#282c38; --muted:#8b8778; --amber:#e0952f; --amber-soft:#6a541f;
      --teal:#4bb7a8; --plum:#a493d6; --good:#5cc48a; --bad:#e07a7a;
    }}
    *{{box-sizing:border-box}}
    body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
      line-height:1.62;-webkit-font-smoothing:antialiased;}}
    .wrap{{max-width:var(--measure);margin:0 auto;padding:clamp(1.4rem,4vw,3.2rem) 1.3rem 5rem;}}
    .fig-wide{{max-width:1080px;margin-inline:auto;}}
    h1,h2,h3{{font-family:var(--serif);font-weight:600;text-wrap:balance;line-height:1.15;}}
    h1{{font-size:clamp(2rem,5vw,3rem);margin:.2em 0 .1em;letter-spacing:-.01em;}}
    h2{{font-size:clamp(1.4rem,3.4vw,1.9rem);margin:2.6em 0 .5em;padding-top:.6em;}}
    h3{{font-size:1.16rem;margin:1.8em 0 .3em;}}
    p{{margin:.7em 0;}}
    .eyebrow{{font-family:var(--mono);font-size:.72rem;letter-spacing:.18em;
      text-transform:uppercase;color:var(--amber);font-weight:600;}}
    .lede{{font-size:1.16rem;color:var(--ink-soft);font-family:var(--serif);
      font-style:italic;border-left:2px solid var(--amber);padding-left:1rem;margin:1.4em 0;}}
    .meta{{font-family:var(--mono);font-size:.74rem;color:var(--muted);
      display:flex;flex-wrap:wrap;gap:.4rem 1.4rem;margin:.8em 0 0;}}
    .rule{{height:1px;background:var(--line);border:0;margin:2.4em 0;}}
    strong{{color:var(--ink);font-weight:650;}}
    code{{font-family:var(--mono);font-size:.86em;background:var(--panel);
      border:1px solid var(--line);border-radius:4px;padding:.06em .34em;}}
    figure{{margin:1.6em 0;}}
    figure img,figure video{{width:100%;height:auto;border:1px solid var(--line);
      border-radius:8px;background:var(--panel);}}
    figcaption{{font-size:.82rem;color:var(--muted);margin-top:.6em;font-family:var(--sans);}}
    figcaption b{{color:var(--ink-soft);}}
    .tbl-scroll{{overflow-x:auto;margin:1.3em 0;}}
    table{{border-collapse:collapse;width:100%;font-family:var(--mono);
      font-size:.82rem;font-variant-numeric:tabular-nums;}}
    th,td{{text-align:right;padding:.42rem .7rem;border-bottom:1px solid var(--line);white-space:nowrap;}}
    th:first-child,td:first-child{{text-align:left;}}
    thead th{{color:var(--amber);border-bottom:1.5px solid var(--amber-soft);
      font-weight:600;font-size:.72rem;letter-spacing:.04em;text-transform:uppercase;}}
    tbody tr:hover{{background:color-mix(in srgb,var(--amber) 7%,transparent);}}
    .callout{{background:var(--panel);border:1px solid var(--line);
      border-left:3px solid var(--plum);border-radius:8px;padding:1rem 1.2rem;margin:1.4em 0;}}
    .callout .tag{{font-family:var(--mono);font-size:.7rem;letter-spacing:.12em;
      text-transform:uppercase;color:var(--plum);font-weight:600;}}
    .callout.verdict{{border-left-color:var(--teal);}}
    .callout.verdict .tag{{color:var(--teal);}}
    .pill{{display:inline-block;font-family:var(--mono);font-size:.72rem;
      padding:.1em .5em;border-radius:999px;border:1px solid currentColor;font-weight:600;}}
    .pill.confirm{{color:var(--good);}} .pill.partial{{color:var(--amber);}}
    .pill.correct{{color:var(--bad);}}
    ul,ol{{padding-left:1.2rem;}} li{{margin:.35em 0;}}
    .foot{{margin-top:3.5em;font-family:var(--mono);font-size:.74rem;color:var(--muted);
      border-top:1px solid var(--line);padding-top:1.2em;}}
    .kbd{{font-family:var(--mono);color:var(--teal);}}
    </style>

    <div class="wrap">
      <div class="eyebrow">PAULA · ALERM — homeostatic dynamics study</div>
      <h1>Does the learning window stabilize?</h1>
      <p class="lede">Two questions about the PAULA metaplastic window <code>t_ref</code>: whether it
      fails to settle in dense networks the way it settles in sparse ones, and how long a network must
      look at something before it carries a usable memory of it.</p>
      <div class="meta">
        <span>CIFAR-10 grayscale</span><span>3 dense layers × 64</span>
        <span>16 000 ticks</span><span>τ<sub>F</sub> ≈ 1000 (β=0.999)</span>
        <span>input auto-calibrated to drive/threshold 1.8</span><span>legacy rule · no model edits</span>
      </div>

      <p>The whole study rests on one fact read straight from <code>neuron.py</code>: <strong>t_ref is
      used at exactly one place</strong> — the plasticity direction switch
      <code>dir = +1 if Δt ≤ t_ref else −1</code>. It never feeds back into firing. So it is a
      <em>passive readout</em> of the firing-rate EMA <code>F_avg</code> through
      <code>t_ref = clip(ceil − (ceil−floor)·clip(F_avg·c,0,1))</code>. "t_ref stabilizes" is therefore
      identical to "the population firing rate reaches a fixed point." Every metric below is a
      per-neuron <em>saturation</em> <code>(t_ref−floor)/(ceil−floor) ∈ [0,1]</code> — comparable across
      architectures whose ceilings <code>c·num_inputs</code> differ.</p>

      <h2><span class="eyebrow">Task A</span><br>Sparse settles quietly; dense settles hot</h2>
      <p>Four MLPs, connectivity 0.1 → 1.0, each held on a constant image for 16 000 ticks with the real
      homeostatic time constant. Every arm reaches a low-slope tail — <strong>nothing diverges</strong>.
      But the way each arm gets there separates cleanly by density.</p>

      <figure class="fig-wide"><img alt="t_ref saturation, efficacy, participation and rolling-std traces for four connectivities" src="{A_static}">
        <figcaption><b>The damped-overshoot signature.</b> Top-left: all four arms drop from the
        t_ref ceiling, <b>overshoot</b> toward the floor as weights grow and the net over-fires, then
        recover to a plateau — denser arms dip deeper (0.66→0.18) and recover slower. Bottom-left:
        during the transient, dense fires in <b>synchronized bursts to 0.22</b>, far above the 0.1
        per-neuron cap. Bottom-right: residual non-stationarity (rolling std of mean saturation) — the
        cleanest stability metric — decays to ~0 for sparse but stays elevated for dense.</figcaption>
      </figure>

      <div class="tbl-scroll"><table>
        <thead><tr><th>connectivity</th><th>final sat</th><th>settle @</th><th>tail jitter (std)</th><th>weight growth</th><th>Δ(stream−const)</th></tr></thead>
        <tbody>
          <tr><td>0.1 — sparse</td><td>0.747</td><td>24%</td><td>0.0001</td><td>1.00→1.75</td><td>0.0021</td></tr>
          <tr><td>0.25</td><td>0.734</td><td>41%</td><td>0.0012</td><td>→2.13</td><td>—</td></tr>
          <tr><td>0.5</td><td>0.725</td><td>66%</td><td>0.0039</td><td>→2.65</td><td>0.0126</td></tr>
          <tr><td>1.0 — dense</td><td>0.661</td><td>68%</td><td>0.0042</td><td>→3.51</td><td>0.0106</td></tr>
        </tbody>
      </table></div>

      <figure class="fig-wide"><video controls muted loop playsinline src="{A_anim}"></video>
        <figcaption><b>Population t_ref saturation over time</b>, sparse (left) vs dense (right). The
        dense histogram sweeps farther toward the floor and stays broader — the visual form of the
        higher residual jitter.</figcaption></figure>

      <div class="callout verdict"><span class="tag">Verdict — the paper claim</span>
        <p style="margin:.4em 0 0"><span class="pill partial">PARTIALLY CONFIRMED</span> &nbsp;Denser MLPs
        are quantifiably <strong>less stable</strong> on three independent, monotone axes — residual
        jitter grows <strong>~40×</strong> (0.0001→0.0042), settling shifts later (24%→68%), and
        input-sensitivity grows with density. But at the 15k-tick scale <strong>none outright fail</strong>:
        every arm reaches a near-frozen tail. Dense is "stable-but-hot," not unstable. A strictly
        divergent regime would need higher drive or a longer horizon.</p></div>

      <h2><span class="eyebrow">Task B</span><br>How long until a network remembers what it saw</h2>
      <p>The only thing that survives a state reset is the synaptic weight vector <code>u_i.info</code>
      (verified in <code>reset_simulation</code>). So the memory <em>is</em> the weights. We expose a
      fresh net for a variable number of ticks, snapshot its weights, then measure how quickly a reset
      copy loaded with those weights re-enters the fully-adapted regime — averaged over 3 nets, since the
      dynamics are deterministic but sensitively dependent on the exact weight pattern.</p>

      <div style="display:grid;grid-template-columns:1fr;gap:1rem" class="fig-wide">
        <figure style="margin:0"><img alt="memory formation, sparse" src="{F1}">
          <figcaption><b>Sparse (conn 0.1).</b> Weight memory (right) forms between ~150 and ~400 ticks
          and plateaus near 1.75; re-entry time (left) collapses from the 2500-tick cold baseline to a
          ~400-tick floor over the same window.</figcaption></figure>
        <figure style="margin:0"><img alt="memory formation, denser" src="{F2}">
          <figcaption><b>Denser (conn 0.5).</b> The same two curves share a knee — but a <b>later,
          larger</b> one (~800–1500 ticks, plateau ~2.7). Denser nets accumulate more memory over a
          longer window, exactly tracking their slower weight growth.</figcaption></figure>
      </div>

      <div class="callout verdict"><span class="tag">Verdict — memory formation</span>
        <p style="margin:.4em 0 0">A usable memory forms in <strong>hundreds of ticks</strong>, set by
        the weight-growth time τ<sub>W</sub>: <strong>~150–400 ticks</strong> sparse, <strong>~800–1500
        ticks</strong> dense. Crucially τ<sub>W</sub> is <em>shorter</em> than the t_ref homeostat's own
        settling time (τ<sub>F</sub>≈1000) — so the weights are the <strong>fast</strong> memory and the
        metaplastic window is the <strong>slow regularizer</strong> riding on top of them.</p></div>

      <h3>What makes a pre-adapted network re-enter faster — and when it doesn't</h3>
      <p>You asked which timescale makes the effect strong and how the parameters relate. Sweeping the
      homeostat speed (β → τ<sub>F</sub>) against the weight speed (η<sub>post</sub> → τ<sub>W</sub>):</p>
      <figure class="fig-wide"><img alt="savings ratio collapses onto tau_W over tau_F" src="{TS}">
        <figcaption>The re-entry speedup <b>collapses onto the ratio τ<sub>W</sub>/τ<sub>F</sub></b>
        (left) and reads directly off the (β, η) grid (right): strong (×5–7) only when the homeostat is
        fast and weights grow slowly (τ<sub>W</sub>≳τ<sub>F</sub>); it vanishes to ×1 when the homeostat
        is slow. <b>At the model default β=0.999 there is essentially no savings</b> — τ<sub>F</sub>≈1000
        ≫ τ<sub>W</sub>, so settling is homeostat-limited and the weight head-start is masked.</figcaption>
      </figure>

      <hr class="rule">
      <h2>What I had to correct along the way</h2>
      <p>Three results looked clean and were wrong; each correction changed a conclusion.</p>

      <div class="callout"><span class="tag">Correction 1 — the input was blind</span>
        <p style="margin:.4em 0 0">At gain 1 the input layer sat at drive/threshold ratio <strong>10.9</strong>,
        far past the refractory-limited ceiling of 2.54: every input neuron fired at its max rate for
        <em>every</em> image, so <strong>image content never entered the network</strong> — streaming and
        constant input gave byte-identical traces. Wiring in the existing <code>--auto-gain</code>
        calibration (target ratio 1.8) fixed it; only then did two different images produce different
        dynamics. All streaming/specificity numbers were re-run.</p></div>

      <div class="callout"><span class="tag">Correction 2 — the ×10 speedup was an artifact</span>
        <p style="margin:.4em 0 0">In the saturated regime a pre-adapted net re-entered <strong>~×10</strong>
        faster than a fresh one. Calibrated, that shrinks to a modest <strong>~15–20%</strong>: the ×10
        was the pathological saturated dynamics taking thousands of ticks to settle, not a genuine memory
        advantage. The honest deliverable is the weight-formation curve, not the raw speedup.</p></div>

      <div class="callout"><span class="tag">Correction 3 — "noise" that was really structure</span>
        <p style="margin:.4em 0 0">The tick dynamics are <strong>fully deterministic</strong> (every RNG
        call is at build time; two runs are byte-identical). The jumpy settle-times were <strong>sensitive
        dependence on the weight pattern</strong> — 25 ticks of LTP reshapes weights enough to divert the
        trajectory. This is the <em>same</em> mechanism as Task A's "denser → jitterier," and it is why
        warm-settle is averaged over seeds.</p></div>

      <hr class="rule">
      <h2>Mapping the model to the brain</h2>
      <p>The measurements pin down which biological role each PAULA / ALERM variable actually plays —
      cellular first, then architectural.</p>
      <div class="tbl-scroll"><table>
        <thead><tr><th>model quantity</th><th>biological counterpart</th><th>timescale (measured)</th></tr></thead>
        <tbody>
          <tr><td>u_i.info (weight)</td><td>synaptic efficacy / LTP–LTD trace</td><td>τ<sub>W</sub> ≈ 150–1500 ticks — <b>the memory</b></td></tr>
          <tr><td>t_ref window</td><td>metaplasticity (BCM sliding threshold)</td><td>τ<sub>F</sub> ≈ 1000 — slow regularizer</td></tr>
          <tr><td>F_avg (EMA)</td><td>Ca²⁺-averaged activity history</td><td>set by β; gates the window</td></tr>
          <tr><td>c (cooldown)</td><td>absolute refractory period</td><td>caps rate at 1/c = 0.1</td></tr>
          <tr><td>r, b thresholds ← M_vector</td><td>intrinsic excitability / neuromodulation</td><td>fast, per-tick</td></tr>
          <tr><td>M0 / M1 (ALERM)</td><td>stress vs reward neuromodulators</td><td>γ-leaky, volume transmission</td></tr>
          <tr><td>connectivity density</td><td>cortical vs cerebellar-style fan-in</td><td>sets damping of the loop</td></tr>
        </tbody>
      </table></div>
      <p>The headline architectural read: <strong>controllable integration windows give the substrate
      two memory systems at once</strong> — fast synaptic weights (working/short-term trace) beneath a
      slow metaplastic homeostat (the stability regularizer). Sparsity is the brain's damping knob:
      low fan-in keeps the weight↔firing↔window loop critically damped; high fan-in makes it ring.</p>

      <div class="callout verdict"><span class="tag">Next — the mini-brain</span>
        <p style="margin:.4em 0 0">Build on calibrated, sparse substrates (auto-gain to ratio ~1.8,
        connectivity ≤0.25 for stability). Perception → cortex → motor as PAULA components, an ALERM
        teacher supplying M1/M0. Because savings vanishes at the default homeostat speed, short-term
        adaptation must live in the <strong>weights</strong> (fast) while the t_ref homeostat holds the
        macro-regime — exactly the two-timescale split these experiments measured.</p></div>

      <div class="foot">Raw data · static + animated renders in
        <span class="kbd">foveation_results/tref_stability/</span> and
        <span class="kbd">foveation_results/savings/</span>. Experiments:
        <span class="kbd">experiment_tref_stability.py</span>,
        <span class="kbd">experiment_savings.py</span>. Measurement-only; <span class="kbd">neuron.py</span> unmodified.</div>
    </div>
    """
    with output.open("x", encoding="utf-8") as f:
        f.write(HISTORICAL_NOTICE + HTML)
    print("wrote", output, output.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
