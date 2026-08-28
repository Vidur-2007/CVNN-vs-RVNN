# CVNN vs. RVNN

A controlled benchmark testing whether complex-valued neural networks (CVNNs) actually outperform real-valued neural networks (RVNNs) on phase-sensitive data — and, more specifically, *where* and *why*.

Three parameter-matched CNNs (~27.0K params each, matched within 0.6%) are trained on an identical synthetic complex-spectrogram classification task and compared across four noise levels and five random seeds:

| Model | Sees | Notes |
|---|---|---|
| `RVNN_mag` | Magnitude spectrogram only (1 channel) | The common "throw away the phase" pipeline |
| `RVNN_ri` | Real + imaginary parts (2 channels) | Same information as CVNN, ordinary real arithmetic |
| `CVNN` | Native complex input (1 complex channel) | Complex conv / complex batch-norm / modReLU, magnitude read-out |

**Result:** the three models tie on clean data, but CVNN pulls significantly ahead as noise increases — +3.0 to +4.6 percentage points at −5dB SNR (p ≈ 0.003, paired t-test, n=5 seeds) — with the advantage concentrated almost entirely in the one class pair engineered to require phase information to distinguish. Full writeup in [`report.md`](report.md) (or the rendered [`report.html`](report.html)).

## Why synthetic data

The original plan used a real audio or MRI dataset. Every common host for one (TensorFlow's speech-commands mirror, HuggingFace, OpenSLR, Zenodo) was network-blocked in the sandbox this was built in, so the benchmark instead uses a synthetic complex-spectrogram task in the same style used in the actual CVNN literature (radar micro-Doppler, modulation classification). This has a real upside: known ground truth and a controllable SNR dial, which is what makes the class-pair ablation in §4 of the report possible. See `data_gen.py` for the full generation logic if you want to swap in real data — `generate_dataset()` is the only function the rest of the pipeline depends on.

## Repo layout

```
data_gen.py        6-class synthetic complex-signal + STFT dataset generator
complex_layers.py  ComplexConv2d / ComplexBatchNorm2d / modReLU — the CVNN building blocks
models.py          RVNN_Mag, RVNN_RI, CVNN — the three architectures (param-matched)
train.py           Benchmark runner: trains all 3 models x 4 SNR levels x 5 seeds (60 runs)
analyze.py         Aggregates results, paired t-tests, confusion matrices -> results/analysis.json
make_figures.py    Renders the charts used in the report -> figures/*.png
build_report.py    Assembles report.html from results/analysis.json + figures/
results/           all_results.json (raw per-run results), analysis.json (aggregated stats)
figures/           acc_vs_snr.png, phase_subproblem.png, confusion_-5dB.png, train_curves.png
report.md          Full write-up (plain Markdown)
report.html         Same write-up as a styled, self-contained HTML page
```

## Running it

```bash
pip install torch torchaudio numpy scipy matplotlib

python3 train.py          # ~30 min on 2 CPU cores; writes results/all_results.json
python3 analyze.py        # writes results/analysis.json, prints tables to stdout
python3 make_figures.py   # writes figures/*.png
python3 build_report.py   # writes report.html
```

Each script is independent and reads only from `results/` / `figures/`, so you can re-run just the analysis or figures without retraining if you already have `all_results.json`.

### Changing the experiment

- **SNR levels / seed count**: edit `SNR_LEVELS` / `SEEDS` at the top of `train.py`.
- **Model width**: `models.py`'s `w=(...)` constructor args — re-run `python3 models.py` to check the three `count_params()` outputs stay matched before retraining.
- **Signal classes**: `data_gen.py`'s `_gen_waveform()`; `CLASSES` list must stay in sync with `NUM_CLASSES` in `models.py`.

## Honest caveats

This is a controlled probe on one small architecture and one synthetic task, not a benchmark suite — see report §6 ("Reading the result honestly") for the full list, including that CVNN ties `RVNN_mag` exactly on clean/noise-free data, and that CVNN's zero seed-to-seed variance at low noise levels reflects a small, somewhat brittle setup rather than a universal property of complex networks.

## Requirements

Python 3.9+, PyTorch ≥ 2.0 (CPU is fine — no GPU used in the original runs), NumPy, SciPy (for `scipy.stats.ttest_rel`), Matplotlib.
