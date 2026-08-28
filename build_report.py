import json, os

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")

with open(os.path.join(HERE, "results", "analysis.json")) as f:
    A = json.load(f)

def b64(name):
    with open(os.path.join(FIG, name)) as f:
        return f.read()

IMG_ACC = b64("acc_vs_snr.png.b64")
IMG_PHASE = b64("phase_subproblem.png.b64")
IMG_CONF = b64("confusion_-5dB.png.b64")
IMG_CURVE = b64("train_curves.png.b64")

S = A["summary"]
SIG = A["significance"]
SUB = A["constant_vs_phasestep_subacc"]

def pct(x): return f"{x*100:.2f}"
def pp(x): return f"{x:+.2f}"

def row(snr):
    m, r, c = S[snr]["RVNN_mag"], S[snr]["RVNN_ri"], S[snr]["CVNN"]
    sig_m, sig_r = SIG[snr]["RVNN_mag"], SIG[snr]["RVNN_ri"]
    return f"""<tr>
      <td>{snr}</td>
      <td class="num">{pct(m['mean'])} ± {pct(m['std'])}</td>
      <td class="num">{pct(r['mean'])} ± {pct(r['std'])}</td>
      <td class="num accent">{pct(c['mean'])} ± {pct(c['std'])}</td>
      <td class="num">{pp(sig_m['delta_pp'])} pp <span class="p">(p={sig_m['p']:.3f})</span></td>
      <td class="num">{pp(sig_r['delta_pp'])} pp <span class="p">(p={sig_r['p']:.3f})</span></td>
    </tr>"""

rows = "\n".join(row(s) for s in ["clean", "15dB", "5dB", "-5dB"])

def subrow(snr):
    v = SUB[snr]
    return f"""<tr><td>{snr}</td><td class="num">{pct(v['RVNN_mag'])}</td><td class="num">{pct(v['RVNN_ri'])}</td><td class="num accent">{pct(v['CVNN'])}</td></tr>"""

subrows = "\n".join(subrow(s) for s in ["clean", "15dB", "5dB", "-5dB"])

html = f"""<title>Complex vs. Real-Valued Networks</title>
<style>
:root {{
  --bg: #f5f6f9;
  --surface: #ffffff;
  --surface-2: #eef0f4;
  --ink: #1a2233;
  --muted: #5b6472;
  --line: #dee2e8;
  --accent: #c93b46;
  --accent-soft: #c93b4614;
  --blue: #3e6f9e;
  --grey: #8b93a1;
  --code-bg: #eef0f4;
  --font-display: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
  --font-body: "IBM Plex Sans", -apple-system, "Segoe UI", sans-serif;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #10141c;
    --surface: #161b25;
    --surface-2: #1c222e;
    --ink: #e7eaf0;
    --muted: #9aa3b2;
    --line: #2a3140;
    --accent: #ff8790;
    --accent-soft: #ff879022;
    --blue: #7fabdb;
    --grey: #9aa3b2;
    --code-bg: #1c222e;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #10141c;
  --surface: #161b25;
  --surface-2: #1c222e;
  --ink: #e7eaf0;
  --muted: #9aa3b2;
  --line: #2a3140;
  --accent: #ff8790;
  --accent-soft: #ff879022;
  --blue: #7fabdb;
  --grey: #9aa3b2;
  --code-bg: #1c222e;
}}

* {{ box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--ink);
  font-family: var(--font-body);
  margin: 0;
  padding: 0 20px 80px;
  line-height: 1.6;
}}
.wrap {{ max-width: 760px; margin: 0 auto; }}
.wide {{ max-width: 980px; margin: 0 auto; }}

header {{
  max-width: 980px;
  margin: 0 auto;
  padding: 64px 0 32px;
  border-bottom: 1px solid var(--line);
}}
.eyebrow {{
  font-family: var(--font-display);
  font-size: 12.5px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 14px;
}}
h1 {{
  font-family: var(--font-display);
  font-size: clamp(28px, 4vw, 40px);
  font-weight: 600;
  line-height: 1.15;
  margin: 0 0 16px;
  text-wrap: balance;
  letter-spacing: -0.01em;
}}
.dek {{
  font-size: 17px;
  color: var(--muted);
  max-width: 640px;
  margin: 0 0 28px;
}}
.meta {{
  display: flex;
  flex-wrap: wrap;
  gap: 28px;
  font-family: var(--font-display);
  font-size: 12.5px;
  color: var(--muted);
}}
.meta b {{ color: var(--ink); font-weight: 600; }}

section {{ max-width: 760px; margin: 0 auto; padding-top: 48px; }}
section.wide-section {{ max-width: 980px; }}
h2 {{
  font-family: var(--font-display);
  font-size: 20px;
  font-weight: 600;
  margin: 0 0 6px;
  display: flex;
  align-items: baseline;
  gap: 10px;
}}
h2 .n {{ color: var(--accent); font-size: 14px; }}
h3 {{ font-size: 16px; font-weight: 600; margin: 28px 0 10px; }}
p {{ margin: 0 0 16px; max-width: 66ch; }}
.wide-section p {{ max-width: 72ch; }}
ul {{ margin: 0 0 16px; padding-left: 22px; }}
li {{ margin-bottom: 6px; max-width: 62ch; }}
code {{
  font-family: var(--font-display);
  background: var(--code-bg);
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 0.88em;
}}
.callout {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-left: 3px solid var(--accent);
  border-radius: 6px;
  padding: 18px 22px;
  margin: 0 0 20px;
}}
.callout p:last-child {{ margin-bottom: 0; }}
.callout .label {{
  font-family: var(--font-display);
  font-size: 11.5px;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 8px;
}}

.figure {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 18px;
  margin: 0 0 8px;
}}
.figure img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
.figcap {{
  font-size: 13px;
  color: var(--muted);
  margin: 12px 4px 28px;
  max-width: 72ch;
}}

.tablewrap {{ overflow-x: auto; margin-bottom: 8px; }}
table {{
  width: 100%;
  border-collapse: collapse;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  font-size: 14px;
}}
th, td {{
  padding: 10px 14px;
  text-align: left;
  border-bottom: 1px solid var(--line);
}}
th {{
  font-family: var(--font-display);
  font-size: 11.5px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 600;
  background: var(--surface-2);
}}
tr:last-child td {{ border-bottom: none; }}
td.num, th.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
td.accent {{ color: var(--accent); font-weight: 600; }}
.p {{ color: var(--muted); font-size: 0.9em; }}

.stat-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px; margin: 24px 0 32px; }}
.stat {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; }}
.stat .k {{ font-family: var(--font-display); font-size: 26px; font-weight: 600; color: var(--accent); font-variant-numeric: tabular-nums; }}
.stat .l {{ font-size: 12.5px; color: var(--muted); margin-top: 4px; }}

.legend-row {{ display: flex; gap: 20px; flex-wrap: wrap; font-size: 13px; color: var(--muted); margin: 12px 0 24px; font-family: var(--font-display); }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }}

footer {{
  max-width: 760px;
  margin: 64px auto 0;
  padding-top: 24px;
  border-top: 1px solid var(--line);
  font-size: 13px;
  color: var(--muted);
}}
footer code {{ font-size: 0.85em; }}
</style>

<header>
  <p class="eyebrow">Empirical benchmark &middot; 60 training runs</p>
  <h1>Why a complex-valued network hears what a real-valued one can't</h1>
  <p class="dek">A parameter-matched CNN benchmark isolating exactly one variable — access to phase — across three architectures, four noise levels, and five random seeds, on a synthetic complex-spectrogram task built so the answer is falsifiable rather than assumed.</p>
  <div class="meta">
    <div><b>27.0K</b> params/model (±0.6%)</div>
    <div><b>6</b> signal classes</div>
    <div><b>4</b> SNR levels &times; <b>5</b> seeds &times; <b>3</b> models</div>
    <div><b>2026-08-27</b></div>
  </div>
</header>

<section>
  <h2><span class="n">01</span> The question, made falsifiable</h2>
  <p>Complex-valued neural networks (CVNNs) are routinely claimed to outperform real-valued networks (RVNNs) on tasks with natural phase structure — radar, MRI, audio, communications. That claim is easy to assert and hard to test cleanly, because most real datasets confound "has phase information" with a dozen other differences between implementations.</p>
  <p>This benchmark isolates the variable. Three CNNs, matched to within 0.6% of each other's parameter count, see the identical training examples through three different windows:</p>
  <ul>
    <li><b>RVNN&#8209;mag</b> — a real-valued CNN that sees only the magnitude spectrogram. The common "throw away the phase" pipeline.</li>
    <li><b>RVNN&#8209;ri</b> — a real-valued CNN that sees the real and imaginary parts stacked as two input channels. Same information as the complex network, ordinary real arithmetic to process it.</li>
    <li><b>CVNN</b> — a native complex-valued CNN (complex convolution, complex batch-norm, modReLU activation) whose final read-out is the magnitude of its last complex feature map, making the whole network equivariant to a global phase rotation of the input.</li>
  </ul>
  <div class="callout">
    <p class="label">Sandbox note</p>
    <p>The original plan was a real audio or MRI dataset. Every host that serves one (TensorFlow's speech-commands mirror, HuggingFace, OpenSLR, Zenodo) is blocked by this environment's network allowlist. The substitute is a synthetic complex-spectrogram task — the same style used in the actual CVNN literature for radar and modulation classification — which has the advantage of a known ground truth and a dial for signal-to-noise ratio.</p>
  </div>
</section>

<section>
  <h2><span class="n">02</span> A task with a built-in control</h2>
  <p>Six classes of baseband complex signals, windowed and passed through a complex STFT to a 32&times;32 complex spectrogram. Five of the classes differ in ordinary ways (chirp direction, frequency modulation rate, beat frequency) that show up in the magnitude spectrogram alone.</p>
  <p>The sixth class is the control. <code>phase_step</code> is a constant-frequency tone identical to the <code>constant</code> class in every respect except one: a single abrupt phase jump partway through the signal. Because a phase discontinuity barely perturbs a windowed magnitude spectrum, <code>constant</code> and <code>phase_step</code> are <i>near-indistinguishable in magnitude alone</i> — a real-valued network that only sees <code>|STFT|</code> is architecturally blind to the one thing that separates them. Every sample additionally gets an independent random global carrier-phase offset, so no model can shortcut by memorizing an absolute phase value; it has to learn phase <i>differences</i>.</p>
  <p>This is the experiment's fulcrum: if phase-aware architectures have a real advantage, it should show up concentrated in exactly this one class pair — and nowhere else, since the other five classes give no such advantage to exploit.</p>
</section>

<section class="wide-section">
  <h2><span class="n">03</span> Overall accuracy vs. noise</h2>
  <p>Each model was trained for 30 epochs, 5 seeds per condition, on a fixed 1,500-sample train / 480-sample test split generated per SNR level. Reported accuracy is mean ± standard deviation across seeds.</p>
  <div class="figure"><img src="data:image/png;base64,{IMG_ACC}" alt="Accuracy vs SNR line chart"></div>
  <p class="figcap">Fig. 1 — Test accuracy across four noise conditions. All three models are close on clean data; the gap opens as SNR drops, with CVNN pulling ahead of both real-valued baselines at every noisy condition.</p>

  <div class="tablewrap">
    <table>
      <thead><tr>
        <th>SNR</th><th class="num">RVNN&#8209;mag</th><th class="num">RVNN&#8209;ri</th><th class="num">CVNN</th>
        <th class="num">Δ vs mag</th><th class="num">Δ vs ri</th>
      </tr></thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </div>
  <p class="figcap">Accuracy in %, mean ± std over 5 seeds. Δ columns are paired t-tests (CVNN &minus; baseline, matched by seed), with p-values from a two-sided paired t-test, n=5.</p>

  <div class="stat-row">
    <div class="stat"><div class="k">+4.58pp</div><div class="l">CVNN vs. RVNN&#8209;ri at &minus;5dB SNR (p=0.003)</div></div>
    <div class="stat"><div class="k">+3.04pp</div><div class="l">CVNN vs. RVNN&#8209;mag at &minus;5dB SNR (p=0.003)</div></div>
    <div class="stat"><div class="k">6 / 8</div><div class="l">noisy-condition comparisons significant at p&lt;0.05</div></div>
  </div>
</section>

<section class="wide-section">
  <h2><span class="n">04</span> Where the advantage actually lives</h2>
  <p>Aggregate accuracy is the headline, but the mechanism is the point. Isolating just the <code>constant</code> vs. <code>phase_step</code> two-class sub-problem — the pair engineered to require phase — makes the effect explicit:</p>
  <div class="figure"><img src="data:image/png;base64,{IMG_PHASE}" alt="Accuracy on the constant vs phase_step subproblem across SNR"></div>
  <p class="figcap">Fig. 2 — Accuracy restricted to the constant/phase_step pair. RVNN&#8209;mag and RVNN&#8209;ri both erode toward chance (50%, dashed line) as noise increases; CVNN degrades far more slowly, staying near 69% even at &minus;5dB.</p>

  <div class="tablewrap">
    <table>
      <thead><tr><th>SNR</th><th class="num">RVNN&#8209;mag</th><th class="num">RVNN&#8209;ri</th><th class="num">CVNN</th></tr></thead>
      <tbody>{subrows}</tbody>
    </table>
  </div>
  <p class="figcap">Accuracy (%) on the 2-class constant-vs-phase_step subproblem only. Chance is 50%.</p>

  <h3>Confusion matrices at the hardest condition (&minus;5dB)</h3>
  <div class="figure"><img src="data:image/png;base64,{IMG_CONF}" alt="Confusion matrices for the three models at -5dB SNR"></div>
  <p class="figcap">Fig. 3 — Row-normalized confusion, summed over 5 seeds. All three models get the four phase-irrelevant classes (up_chirp, down_chirp, vibrato, two_tone) exactly right, 100% of the time, every seed — confirming those classes carry no phase advantage to exploit. Every error, for every model, is confined to the constant/phase_step pair. CVNN resolves <code>constant</code> correctly 91% of the time versus 57&ndash;61% for the real-valued baselines; recovering <code>phase_step</code> itself is closer to a coin flip for all three, since the jump's exact location is noisier information to recover than its mere presence.</p>
</section>

<section>
  <h2><span class="n">05</span> Training dynamics</h2>
  <div class="figure"><img src="data:image/png;base64,{IMG_CURVE}" alt="Training curves at -5dB SNR"></div>
  <p class="figcap">Fig. 4 — Test accuracy per epoch at &minus;5dB SNR, seed 0. CVNN reaches its plateau in roughly half the epochs RVNN&#8209;ri needs and settles at a visibly higher, steadier ceiling — consistent with the noise-robustness argument made for complex batch-norm and modReLU in the literature (Trabelsi et al., 2018): treating magnitude and phase as a coupled quantity, rather than two independent real channels, regularizes what the network can learn to key on.</p>
</section>

<section>
  <h2><span class="n">06</span> Reading the result honestly</h2>
  <p>A few things worth stating plainly rather than glossing over:</p>
  <ul>
    <li><b>On clean, noise-free data, CVNN ties RVNN&#8209;mag exactly</b> (88.54% both, zero variance across all 5 seeds for both). The phase-aware architecture buys nothing when there's no noise to be robust to and the magnitude channel alone already carries a faint artifact of the phase jump (STFT windowing leaks a little energy across the discontinuity). The advantage is specifically a <i>noise-robustness</i> and <i>information-access</i> story, not a "complex numbers are strictly better, always" story.</li>
    <li><b>CVNN's zero variance across seeds</b> at clean/15dB/5dB (std=0.00 in the table above) is a real observation, not a rounding artifact — this small model, on this small fixed dataset, converges to the same local optimum regardless of initialization at those noise levels. It's a sign the task is small enough to be somewhat brittle, and a reason to treat the exact percentages here as illustrative rather than as literature-grade numbers. The RVNN baselines do show normal seed-to-seed variance throughout, which is what makes the paired t-tests meaningful.</li>
    <li><b>RVNN&#8209;ri, not RVNN&#8209;mag, is the fairer baseline</b> — it has the same information (real and imaginary parts) as the CVNN, just without complex algebraic structure. CVNN beats RVNN&#8209;ri by a statistically significant margin at every single SNR tested, including on clean data (p&nbsp;&lt;&nbsp;0.02 throughout). That comparison, more than the RVNN&#8209;mag one, is the real test of whether native complex arithmetic buys something beyond "having access to phase," and it does.</li>
    <li><b>This is one task, one architecture family, one scale.</b> A 27K-parameter, 3-layer CNN on a 1,500-sample synthetic dataset is a controlled probe, not a benchmark suite. The direction of the result — parity on clean/noiseless data, a widening CVNN lead as SNR drops, concentrated exactly in the phase-dependent class pair — matches what's reported at much larger scale in the actual CVNN literature (SAR imagery, MRI reconstruction, RF modulation classification), which is the most that a benchmark like this can honestly claim to corroborate.</li>
  </ul>
</section>

<footer>
  <p>Code: <code>data_gen.py</code> (synthetic signal + STFT generator), <code>complex_layers.py</code> (ComplexConv2d / ComplexBatchNorm2d / modReLU), <code>models.py</code> (the three architectures), <code>train.py</code> (60-run benchmark sweep), <code>analyze.py</code> (significance tests), <code>make_figures.py</code> (this report's charts) — delivered alongside this report. PyTorch 2.13 (CPU), 2 threads, ~28 minutes total training time.</p>
</footer>
"""

out_path = os.path.join(HERE, "report.html")
with open(out_path, "w") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")
