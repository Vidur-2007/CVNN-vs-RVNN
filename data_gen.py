"""
Synthetic complex-domain signal benchmark.

Real audio/MRI dataset hosts are network-blocked in this sandbox, so we
substitute a controlled synthetic complex-spectrogram classification task,
in the same style used in the actual CVNN literature (radar micro-Doppler,
modulation classification, phase-coded pulse detection - e.g. Trabelsi et al.
2018 "Deep Complex Networks"; Zhang et al. on complex-valued SAR/radar nets).

Six classes of baseband complex analytic signals x[n] = A(n) * exp(i*phase(n)):

  0 constant   - fixed-frequency tone
  1 up_chirp   - linearly increasing instantaneous frequency
  2 down_chirp - linearly decreasing instantaneous frequency
  3 vibrato    - sinusoidally frequency-modulated tone (vibrato / micro-Doppler)
  4 two_tone   - sum of two close tones -> amplitude beating
  5 phase_step - fixed-frequency tone with a mid-signal phase discontinuity

Critically, class 5 (phase_step) is constructed to have (almost) the SAME
magnitude spectrogram as class 0 (constant) -- the only difference is a phase
jump. A magnitude-only real-valued network is architecturally incapable of
separating these two classes; a network with access to phase (real+imag, or
native complex) is not.

Every sample also gets an independent random *global* carrier-phase offset
phi0 ~ U[0, 2*pi) and complex Gaussian noise at a configurable SNR. The
random global phase is a nuisance parameter: it forces a model to learn
phase-difference / phase-rotation-invariant structure rather than memorizing
an absolute phase value, which is exactly the inductive bias native complex
arithmetic (multiplication by e^{i*phi0} is a simple gain on a complex
network's internal representations) provides for free, and a plain
real-valued network does not get architecturally.
"""
import numpy as np

CLASSES = ["constant", "up_chirp", "down_chirp", "vibrato", "two_tone", "phase_step"]
NUM_CLASSES = len(CLASSES)

SR = 8000          # sample rate (Hz)
DUR = 0.512         # seconds
N = int(SR * DUR)   # time samples
F0 = 1000.0         # base frequency (Hz)


def _window(n):
    # raised-cosine (Hann) envelope so edge effects don't leak class info
    t = np.arange(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * t / (n - 1))


def _gen_waveform(cls, rng):
    t = np.arange(N) / SR
    env = _window(N)

    if cls == 0:  # constant
        phase = 2 * np.pi * F0 * t
        x = env * np.exp(1j * phase)

    elif cls == 1:  # up_chirp
        f1 = F0 + 700.0
        k = (f1 - F0) / DUR
        phase = 2 * np.pi * (F0 * t + 0.5 * k * t**2)
        x = env * np.exp(1j * phase)

    elif cls == 2:  # down_chirp
        f1 = F0 + 700.0
        k = (f1 - F0) / DUR
        phase = 2 * np.pi * (f1 * t - 0.5 * k * t**2)
        x = env * np.exp(1j * phase)

    elif cls == 3:  # vibrato
        fm = rng.uniform(6.0, 10.0)     # modulation rate (Hz)
        fd = rng.uniform(80.0, 160.0)   # modulation depth (Hz)
        phase = 2 * np.pi * (F0 * t - (fd / (2 * np.pi * fm)) * np.cos(2 * np.pi * fm * t))
        x = env * np.exp(1j * phase)

    elif cls == 4:  # two_tone (amplitude beating)
        delta = rng.uniform(90.0, 160.0)
        p1 = 2 * np.pi * (F0 - delta / 2) * t
        p2 = 2 * np.pi * (F0 + delta / 2) * t
        x = env * 0.5 * (np.exp(1j * p1) + np.exp(1j * p2))

    elif cls == 5:  # phase_step: identical magnitude to class 0, phase jump only
        phase = 2 * np.pi * F0 * t
        jump_idx = rng.integers(int(0.3 * N), int(0.7 * N))
        jump = rng.choice([np.pi / 2, np.pi, -np.pi / 2, -np.pi])
        phase = phase.copy()
        phase[jump_idx:] += jump
        x = env * np.exp(1j * phase)

    else:
        raise ValueError(cls)

    # random global carrier-phase nuisance offset
    phi0 = rng.uniform(0, 2 * np.pi)
    x = x * np.exp(1j * phi0)
    return x.astype(np.complex64)


def _add_noise(x, snr_db, rng):
    if snr_db is None or snr_db == np.inf:
        return x
    sig_power = np.mean(np.abs(x) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    noise = noise * np.sqrt(noise_power / 2)
    return (x + noise).astype(np.complex64)


def _stft_crop(x, n_fft=63, hop=16, n_freq=32, n_time=32):
    """Complex STFT of a complex (analytic, baseband) signal, cropped/padded
    to a fixed (n_freq, n_time) size."""
    import torch
    xt = torch.from_numpy(x)
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
    S = torch.stft(
        xt, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window,
        center=True, return_complex=True, onesided=False,
    )  # (n_fft, n_frames) complex, baseband signal -> keep low positive+negative freqs
    # center the frequency axis around 0 Hz and crop n_freq bins
    S = torch.fft.fftshift(S, dim=0)
    f_center = S.shape[0] // 2
    f0 = f_center - n_freq // 2
    S = S[f0:f0 + n_freq, :]
    # crop/pad time axis to n_time
    if S.shape[1] >= n_time:
        t0 = (S.shape[1] - n_time) // 2
        S = S[:, t0:t0 + n_time]
    else:
        pad = n_time - S.shape[1]
        S = torch.nn.functional.pad(S, (0, pad))
    return S.numpy().astype(np.complex64)


def generate_dataset(n_per_class, snr_db, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls in range(NUM_CLASSES):
        for _ in range(n_per_class):
            x = _gen_waveform(cls, rng)
            x = _add_noise(x, snr_db, rng)
            S = _stft_crop(x)
            X.append(S)
            y.append(cls)
    X = np.stack(X)  # (N, n_freq, n_time) complex64
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


if __name__ == "__main__":
    X, y = generate_dataset(n_per_class=20, snr_db=10, seed=0)
    print("X", X.shape, X.dtype, "y", y.shape)
    for c in range(NUM_CLASSES):
        idx = np.where(y == c)[0][0]
        print(CLASSES[c], "mag range", np.abs(X[idx]).min(), np.abs(X[idx]).max())
