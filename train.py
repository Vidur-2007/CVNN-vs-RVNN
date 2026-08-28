"""Benchmark runner: trains RVNN_mag, RVNN_RI, and CVNN across multiple
random seeds and multiple SNR levels, and logs results to results/*.json.
"""
import json
import time
import os
import numpy as np
import torch
import torch.nn as nn

from data_gen import generate_dataset, NUM_CLASSES, CLASSES
from models import RVNN_Mag, RVNN_RI, CVNN, count_params

DEVICE = "cpu"
torch.set_num_threads(2)

SNR_LEVELS = [None, 15, 5, -5]     # None = clean (no added noise)
SEEDS = [0, 1, 2, 3, 4]
N_TRAIN_PER_CLASS = 250
N_TEST_PER_CLASS = 80
EPOCHS = 30
BATCH_SIZE = 64
LR = 2e-3

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def to_tensors(X):
    """X: complex64 (N, F, T) -> re, im, mag each (N,1,F,T) float32 tensors."""
    re = torch.from_numpy(X.real.astype(np.float32)).unsqueeze(1)
    im = torch.from_numpy(X.imag.astype(np.float32)).unsqueeze(1)
    mag = torch.sqrt(re ** 2 + im ** 2)
    # per-sample normalization so amplitude scale isn't a trivial shortcut
    scale = mag.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    re, im, mag = re / scale, im / scale, mag / scale
    return re, im, mag


def make_model(name):
    if name == "RVNN_mag":
        return RVNN_Mag()
    if name == "RVNN_ri":
        return RVNN_RI()
    if name == "CVNN":
        return CVNN()
    raise ValueError(name)


def model_forward(name, model, re, im, mag):
    if name == "RVNN_mag":
        return model(mag)
    if name == "RVNN_ri":
        return model(torch.cat([re, im], dim=1))
    if name == "CVNN":
        return model((re, im))
    raise ValueError(name)


def run_one(model_name, snr_db, seed, log_curve=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # data is fixed per SNR level (shared across seeds/models); use a data
    # seed derived from snr only, decoupled from the model/training seed
    data_seed = 10_000 + (0 if snr_db is None else int(snr_db))
    Xtr, ytr = generate_dataset(N_TRAIN_PER_CLASS, snr_db, seed=data_seed)
    Xte, yte = generate_dataset(N_TEST_PER_CLASS, snr_db, seed=data_seed + 1)

    re_tr, im_tr, mag_tr = to_tensors(Xtr)
    re_te, im_te, mag_te = to_tensors(Xte)
    ytr_t = torch.from_numpy(ytr)
    yte_t = torch.from_numpy(yte)

    model = make_model(model_name).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()

    n = len(ytr_t)
    curve = []
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        tot_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            logits = model_forward(model_name, model, re_tr[idx], im_tr[idx], mag_tr[idx])
            loss = crit(logits, ytr_t[idx])
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
        sched.step()
        if log_curve:
            model.eval()
            with torch.no_grad():
                te_logits = model_forward(model_name, model, re_te, im_te, mag_te)
                te_acc = (te_logits.argmax(1) == yte_t).float().mean().item()
            curve.append({"epoch": epoch, "train_loss": tot_loss / n, "test_acc": te_acc})

    model.eval()
    with torch.no_grad():
        te_logits = model_forward(model_name, model, re_te, im_te, mag_te)
        pred = te_logits.argmax(1)
        acc = (pred == yte_t).float().mean().item()
        # per-class confusion matrix
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for t, p in zip(yte_t.numpy(), pred.numpy()):
            cm[t, p] += 1

    return {
        "model": model_name,
        "snr_db": snr_db,
        "seed": seed,
        "test_acc": acc,
        "confusion": cm.tolist(),
        "curve": curve,
        "n_params": count_params(model),
    }


def main():
    all_results = []
    t_start = time.time()
    total_runs = len(SNR_LEVELS) * 3 * len(SEEDS)
    run_i = 0
    for snr in SNR_LEVELS:
        for model_name in ["RVNN_mag", "RVNN_ri", "CVNN"]:
            for seed in SEEDS:
                run_i += 1
                t0 = time.time()
                # log full curve only for seed 0 (keeps result file small)
                res = run_one(model_name, snr, seed, log_curve=(seed == 0))
                dt = time.time() - t0
                all_results.append(res)
                print(f"[{run_i}/{total_runs}] snr={snr} model={model_name} seed={seed} "
                      f"acc={res['test_acc']:.3f} ({dt:.1f}s, total {time.time()-t_start:.0f}s)", flush=True)
                # incremental checkpoint
                with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
                    json.dump(all_results, f)
    print(f"DONE in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
