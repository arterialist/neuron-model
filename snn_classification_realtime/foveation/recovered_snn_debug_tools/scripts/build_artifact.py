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
    parser.add_argument("--input-root", type=Path, default=repo / "foveation_results/minibrain")
    parser.add_argument("--output", type=Path, default=repo / "foveation_results/rebuilt_reports/minibrain_dynamics.historical.html")
    args = parser.parse_args()
    output = args.output.expanduser()
    if output.exists():
        parser.error(f"refusing to replace an existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    ROOT = args.input_root.expanduser()

    def uri(path):
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()

    IMG = {
        "retino": uri(f"{ROOT}/gallery/retino_gray.png"),
        "random": uri(f"{ROOT}/gallery/random_gray.png"),
        "conv":   uri(f"{ROOT}/gallery/conv_gray.png"),
        "gaze":   uri(f"{ROOT}/gallery/gaze_hunt.png"),
        "temporal": uri(f"{ROOT}/temporal/AGG_temporal.png"),
        "sep":    uri(f"{ROOT}/arch/AGG_sep.png"),
        "plast":  uri(f"{ROOT}/arch/AGG_plast.png"),
        "gaze_agg": uri(f"{ROOT}/arch/AGG_gaze.png"),
        "powered": uri(f"{ROOT}/pool_powered/AGG_powered.png"),
    }

    HTML = f"""<title>Mini-Brain — Reading the Dynamics</title>
    <style>
      :root {{
        --bg:#f1efe9; --surface:#faf8f3; --surface-2:#eceae2; --ink:#161a20; --muted:#5b636e;
        --faint:#8b929c; --line:#dbd7cd; --accent:#0d818f; --accent-ink:#0a6470; --accent-soft:#d3e9ec;
        --ok:#2f7d52; --partial:#a9781f; --no:#ad433a; --neuro:#9c3a74;
        --mono:ui-monospace,"SF Mono","JetBrains Mono",Menlo,monospace;
        --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; --measure:66ch;
      }}
      @media (prefers-color-scheme:dark){{:root{{
        --bg:#0e1218; --surface:#151a22; --surface-2:#1b212b; --ink:#e7eaee; --muted:#93a0af;
        --faint:#6b7684; --line:#28303b; --accent:#3fc3d2; --accent-ink:#8fe0ea; --accent-soft:#10333a;
        --ok:#54b781; --partial:#d2a24c; --no:#d66459; --neuro:#d074aa;
      }}}}
      :root[data-theme="light"]{{--bg:#f1efe9;--surface:#faf8f3;--surface-2:#eceae2;--ink:#161a20;--muted:#5b636e;--faint:#8b929c;--line:#dbd7cd;--accent:#0d818f;--accent-ink:#0a6470;--ok:#2f7d52;--partial:#a9781f;--no:#ad433a;--neuro:#9c3a74;}}
      :root[data-theme="dark"]{{--bg:#0e1218;--surface:#151a22;--surface-2:#1b212b;--ink:#e7eaee;--muted:#93a0af;--faint:#6b7684;--line:#28303b;--accent:#3fc3d2;--accent-ink:#8fe0ea;--ok:#54b781;--partial:#d2a24c;--no:#d66459;--neuro:#d074aa;}}
      *{{box-sizing:border-box}}
      body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6;-webkit-font-smoothing:antialiased;font-size:16px}}
      .wrap{{max-width:1000px;margin:0 auto;padding:0 24px}}
      p{{max-width:var(--measure)}} a{{color:var(--accent-ink)}}
      .eyebrow{{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--accent-ink);margin:0 0 10px}}
      h1{{font-size:clamp(2.1rem,5.4vw,3.6rem);line-height:1.03;letter-spacing:-.022em;font-weight:800;margin:.1em 0 .3em;text-wrap:balance}}
      h2{{font-size:clamp(1.4rem,3vw,1.95rem);letter-spacing:-.015em;font-weight:750;margin:0 0 .2em;text-wrap:balance}}
      h3{{font-size:1.02rem;font-weight:700;margin:0 0 .3em}}
      .lede{{font-size:1.2rem;color:var(--muted);max-width:62ch}}
      .num{{font-family:var(--mono);font-variant-numeric:tabular-nums}}
      header.hero{{padding:66px 0 34px;border-bottom:1px solid var(--line)}}
      .kpis{{display:flex;flex-wrap:wrap;gap:26px;margin-top:30px;padding-top:18px;border-top:1px solid var(--line)}}
      .kpi .n{{font-family:var(--mono);font-weight:600;font-size:2.1rem;letter-spacing:-.02em;color:var(--accent-ink);line-height:1}}
      .kpi .l{{font-size:.78rem;color:var(--faint);text-transform:uppercase;letter-spacing:.08em;margin-top:8px;max-width:22ch}}
      .sec{{padding:50px 0;border-bottom:1px solid var(--line)}}
      .sechead{{display:flex;align-items:baseline;gap:14px;margin-bottom:14px}}
      .secidx{{font-family:var(--mono);font-size:.8rem;color:var(--faint)}}
      figure{{margin:22px 0;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px;overflow:hidden}}
      figure img{{width:100%;height:auto;border-radius:6px;display:block}}
      figcaption{{font-size:.86rem;color:var(--muted);margin-top:10px;font-family:var(--mono)}}
      .gal{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;margin:22px 0}}
      .gal figure{{margin:0}}
      .badge{{font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;font-weight:600;color:var(--accent-ink);display:inline-flex;align-items:center;gap:6px}}
      .badge::before{{content:"";width:7px;height:7px;border-radius:50%;background:var(--accent)}}
      .cards{{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));margin:22px 0}}
      .card{{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:18px 18px 18px 22px;position:relative;overflow:hidden}}
      .card::before{{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--stripe,var(--accent))}}
      .card.ok{{--stripe:var(--ok)}} .card.partial{{--stripe:var(--partial)}} .card.no{{--stripe:var(--no)}}
      .tag{{font-family:var(--mono);font-size:.66rem;letter-spacing:.12em;text-transform:uppercase;font-weight:600;color:var(--stripe,var(--accent))}}
      .card .kpi2{{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:1.4rem;font-weight:600;margin-top:10px}}
      .card p{{font-size:.92rem;color:var(--muted);margin:.3em 0 0}}
      .tablewrap{{overflow-x:auto;margin:22px 0;border:1px solid var(--line);border-radius:9px}}
      table{{border-collapse:collapse;width:100%;font-size:.9rem;min-width:520px}}
      th,td{{text-align:left;padding:10px 15px;border-bottom:1px solid var(--line)}}
      thead th{{font-family:var(--mono);font-size:.68rem;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);font-weight:600;background:var(--surface)}}
      tbody tr:last-child td{{border-bottom:none}}
      td.n{{font-family:var(--mono);font-variant-numeric:tabular-nums;text-align:right}}
      .up{{color:var(--ok);font-weight:650}} .flat{{color:var(--faint)}}
      .pull{{border-left:3px solid var(--accent);padding:6px 0 6px 18px;margin:22px 0;font-size:1.12rem;color:var(--ink);max-width:60ch}}
      pre{{background:var(--surface-2);border:1px solid var(--line);border-radius:8px;padding:14px 16px;overflow-x:auto;font-family:var(--mono);font-size:.82rem;color:var(--ink)}}
      code{{font-family:var(--mono);font-size:.86em;background:var(--surface-2);padding:1px 5px;border-radius:4px}}
      .foot{{padding:38px 0 64px;color:var(--faint);font-family:var(--mono);font-size:.74rem}}
      strong{{font-weight:650}} em{{color:var(--ink);font-style:italic}}
    </style>

    <header class="hero"><div class="wrap">
      <p class="eyebrow">PAULA Mini-Brain · CIFAR-10 · Architecture &amp; Readout Study</p>
      <h1>Does the trajectory beat the mean? A hypothesis, a false positive, and the powered test that killed it.</h1>
      <p class="lede">We asked whether reading the substrate as a <em>tick-by-tick trajectory</em> (GRU/LSTM on the
      spike sequence) recovers class information a <em>mean over ticks</em> discards. An underpowered run said
      yes in the recurrent pool. A properly-powered run (4× the data, σ≈0.02) said <strong>no</strong> — the
      effect was ~1σ noise. On this task, the mean is a sufficient statistic. This page keeps the honest arc.</p>
      <div class="kpis">
        <div class="kpi"><div class="n">≈0</div><div class="l">powered pool Δ (temporal − mean), all 3 topologies</div></div>
        <div class="kpi"><div class="n">−0.019</div><div class="l">underpowered +0.042 → powered: gone</div></div>
        <div class="kpi"><div class="n">0.194</div><div class="l">reward_hebb+teacher substrate separability (vs 0.08 frozen) — the one positive</div></div>
        <div class="kpi"><div class="n">flat</div><div class="l">static mean-lens across all architectures</div></div>
      </div>
    </div></header>

    <section class="sec"><div class="wrap">
      <div class="sechead"><span class="secidx">01</span><h2>Watch the substrate think</h2></div>
      <p>Each clip is one continuously-running mini-brain: a foveated retina glimpses CIFAR through a fixed
      retinal-convolution front-end into a sparse PAULA substrate, tick-synced across the moving fovea, the
      retina views, the <strong>pool liquid-state map</strong>, the layered <strong>spike raster</strong>,
      per-layer participation + teacher m0/m1, and the decoder posterior. The pool raster is where the
      effect below lives — its <em>timing</em>, not its average, carries the class. (Full videos delivered separately.)</p>
      <div class="gal">
        <figure><span class="badge">retinotopic reservoir</span><img src="{IMG['retino']}" alt="retinotopic dynamics"><figcaption>Local receptive fields — preserves conv locality. Best separability (0.267).</figcaption></figure>
        <figure><span class="badge">random reservoir</span><img src="{IMG['random']}" alt="random dynamics"><figcaption>Dense scrambling pool — the classic LSM baseline (0.217).</figcaption></figure>
        <figure><span class="badge">convolutional PAULA</span><img src="{IMG['conv']}" alt="conv dynamics"><figcaption>Fewer, structured units — lowest readout dimensionality here (0.16).</figcaption></figure>
        <figure><span class="badge">gaze hunt · cluttered</span><img src="{IMG['gaze']}" alt="gaze hunt"><figcaption>Fovea searching a 72px canvas for the hidden object (klinotaxis drift).</figcaption></figure>
      </div>
    </div></section>

    <section class="sec"><div class="wrap">
      <div class="sechead"><span class="secidx">02</span><h2>The load-bearing test: trajectory vs mean</h2></div>
      <p>On identical frozen substrates we read the representation three ways and asked which recovers class
      structure: the <strong>mean</strong> over the fixation, a <strong>temporal-linear</strong> readout
      (time-binned rates + membrane/t_ref standing state + population power-spectrum), and a trained
      <strong>GRU/LSTM</strong> on the raw spike trajectory — the continuous <span class="num">(ticks × neurons)</span>
      shape the real pipeline records, classified so the mean can't destroy it.</p>

      <h3 style="color:var(--partial)">First pass (underpowered) — a tempting signal</h3>
      <figure><img src="{IMG['temporal']}" alt="temporal vs mean, underpowered"><figcaption>400 img, σ≈0.033. The pool panel looked like the prediction: temporal &gt; mean for conv (+0.042), random (+0.036), retino (+0.015), while the feedforward input showed nothing. Cleanest possible shape for “dynamics live in the pool” — but each Δ was only ~1σ, and the RNNs had just 280 training trajectories.</figcaption></figure>

      <h3 style="color:var(--no)">Powered pass (4× data, σ≈0.02) — the effect vanishes</h3>
      <figure><img src="{IMG['powered']}" alt="temporal vs mean, powered"><figcaption>1200 img/arm × 2 seeds. With proper power the pool Δ collapses to zero-or-negative for all three topologies. The ~1σ pool signal was noise.</figcaption></figure>

      <div class="tablewrap"><table>
        <thead><tr><th>pool layer</th><th class="n">underpowered Δ</th><th class="n">powered Δ</th><th></th></tr></thead>
        <tbody>
          <tr><td>conv</td><td class="n">+0.042</td><td class="n">−0.019</td><td class="flat">gone</td></tr>
          <tr><td>random</td><td class="n">+0.036</td><td class="n">−0.013</td><td class="flat">gone</td></tr>
          <tr><td>retino</td><td class="n">+0.015</td><td class="n">−0.001</td><td class="flat">gone</td></tr>
        </tbody>
      </table></div>
      <p class="pull">Verdict: on this task, reading the spike trajectory — temporal-linear <em>or</em> GRU/LSTM —
      recovers no more CIFAR-class information than the mean. The mean is a sufficient statistic here. The pool
      effect was an underpowered false positive.</p>
      <p>Two boundaries worth stating precisely. <strong>Refuted:</strong> that a downstream classifier extracts
      more <em>class-label</em> info from the trajectory than the mean on this substrate — and since the cheap,
      well-powered <code>temporal_lin</code> also shows nothing, this isn't merely an undertrained RNN.
      <strong>Not refuted:</strong> that the representation is dynamical in the deeper sense — the dynamics can
      still carry the concept and drive the substrate's computation while the mean happens to capture as much
      <em>label</em> info as the trajectory for this easy readout target. The lone weak crumb — LSTM on the
      reservoir <em>input</em> layer (+0.03, ~1.5σ, absent in conv) — is marginal and not where predicted.</p>
    </div></section>

    <section class="sec"><div class="wrap">
      <div class="sechead"><span class="secidx">03</span><h2>The static lens is a flat null — and that's the point</h2></div>
      <p>Under the mean lens, <strong>nothing distinguishes the architectures</strong>: retinotopic ≈ random ≈
      0.21, conv lower, the richer conv bank doesn't help, and — tellingly — a trained <strong>MLP readout is
      worse than linear</strong> (0.208 → 0.170). A fancier readout of a mean can't recover what the mean
      discarded; it only overfits. That flatness is not a boring negative — it is the fingerprint of averaging
      away a temporal code.</p>
      <figure><img src="{IMG['sep']}" alt="encoder x readout separability"><figcaption>Exp 1 — frozen-encoder separability (mean lens). Linear vs MLP across the encoder matrix; everything collapses into a ~0.20 band.</figcaption></figure>
    </div></section>

    <section class="sec"><div class="wrap">
      <div class="sechead"><span class="secidx">04</span><h2>Does the substrate learn? The teacher-gated rule says: a little.</h2></div>
      <p>A gated, non-saturating native plasticity rule (<code>reward_hebb</code>, added to <code>neuron.py</code>
      behind a default-off flag) with the ALERM teacher produced the <strong>highest frozen-probe separability
      of any plastic arm</strong>. The teacher matters: with it, 0.194; without it, 0.139; frozen alone, 0.083–0.139.
      A hint that reward-gated plasticity is <em>constructive</em> in the locality-preserving topology — the
      opposite of the old dense-reservoir saturation. (Compare the frozen vs reward_hebb videos: the m1/m0 panel
      is actively reshaping the pool in the latter.)</p>
      <div class="cards">
        <div class="card ok"><span class="tag">Works</span><div class="kpi2">0.194</div><p>reward_hebb + teacher — highest plastic separability</p></div>
        <div class="card partial"><span class="tag">Teacher matters</span><div class="kpi2">0.139</div><p>reward_hebb, no teacher — the reward gate carries it</p></div>
        <div class="card no"><span class="tag">Baseline</span><div class="kpi2">0.08–0.14</div><p>frozen substrate — the reference</p></div>
      </div>
      <figure><img src="{IMG['plast']}" alt="plasticity trajectories"><figcaption>Exp 2 — substrate-plasticity trajectories over the 400-image run (frozen / reward_hebb ± teacher / external oja·bcm·rmhebb / association layer).</figcaption></figure>
    </div></section>

    <section class="sec"><div class="wrap">
      <div class="sechead"><span class="secidx">05</span><h2>Learning where to look — homes, but doesn't lock</h2></div>
      <p>On a cluttered canvas the learned gaze policies (linear / MLP / recurrent REINFORCE) all
      <strong>home toward the object</strong> — 15.9–17.5 px final distance vs a random walk's 26.8 px — but
      classification stays near chance (oracle itself only 0.133 under the mean-based memorize readout). The
      fancier policies did not beat linear. Homing works; foveation-grade locking, and the mean-based readout,
      remain the ceiling.</p>
      <figure><img src="{IMG['gaze_agg']}" alt="gaze accuracy and distance"><figcaption>Exp 3 — cluttered gaze: online accuracy (left) and gaze→object distance (right). Learned policies reduce distance toward the object; accuracy stays low.</figcaption></figure>
    </div></section>

    <section class="sec"><div class="wrap">
      <div class="sechead"><span class="secidx">06</span><h2>What would actually test the thesis</h2></div>
      <p>“Temporal decoding = mean decoding” on CIFAR does not mean the representation is static — it means this
      <em>task</em> doesn't force the dynamics to matter for the <em>label</em>. A foveated CIFAR glimpse is
      nearly a static-image problem; the mean can carry whatever little class signal exists. To actually probe
      whether the representation is spatiotemporal you need a setting where the mean provably cannot win:</p>
      <div class="cards">
        <div class="card"><span class="tag">Task</span><h3>Dynamics-only stimuli</h3><p>Inputs that differ <em>only</em> in their time course (sequence order, motion, flicker phase) so a mean is information-free by construction.</p></div>
        <div class="card"><span class="tag">Readout</span><h3>Regime identity, not rate</h3><p>Target the specific invariant — oscillation phase relationships, attractor ID, limit-cycle geometry — rather than binned rates a mean already summarizes.</p></div>
        <div class="card"><span class="tag">Evidence</span><h3>Computation, not decode</h3><p>Show the dynamics are <em>causal</em> for the substrate's behavior (perturb the trajectory, not the mean, and watch output change) — the thesis is about what the network <em>does</em>, not just what a probe can read.</p></div>
      </div>
      <p class="pull">The rigorous takeaway: the mean is a sufficient statistic for CIFAR class on this substrate.
      Demonstrating the spatiotemporal thesis needs a task the mean cannot solve — not a harder readout of an
      easy one.</p>
    </div></section>

    <footer class="foot"><div class="wrap">
      <p>Measurement + gated, default-off additions only. neuron.py: new <code>plasticity_mode="reward_hebb"</code>
      behind a flag (default legacy unchanged). Params: retinal-conv front-end, sparse PAULA substrate, dwell 500 /
      warmup 500 / 400 img/arm sweep; temporal readout records (firings, avg_S, avg_t_ref) as (ticks×neurons),
      GRU/LSTM on MPS. Held-out, seed-checked where noted; every number above is measured, not asserted.</p>
    </div></footer>
    """

    with output.open("x", encoding="utf-8") as f:
        f.write(HISTORICAL_NOTICE + HTML)
    print("wrote", output, output.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
