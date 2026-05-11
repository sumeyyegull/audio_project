import os, sys
sys.path.insert(0, '.')
from noise_filter import load_wav, adaptive_noise_filter, full_metrics

data_dir   = 'data'
orig_dir   = os.path.join(data_dir, 'original')
noisy_dir  = os.path.join(data_dir, 'noisy')
METHOD     = sys.argv[1] if len(sys.argv) > 1 else 'wiener'
MAX_FILES  = 100

files = sorted(f for f in os.listdir(orig_dir) if f.endswith('.wav'))[:MAX_FILES]

all_metrics = []
for fname in files:
    noisy_path = os.path.join(noisy_dir, fname)
    if not os.path.isfile(noisy_path):
        continue
    sr, clean = load_wav(os.path.join(orig_dir, fname))
    _, noisy  = load_wav(noisy_path)
    N = min(len(clean), len(noisy))
    clean, noisy = clean[:N], noisy[:N]

    filtered = adaptive_noise_filter(noisy, sr, method=METHOD, known_noise=clean)
    m = full_metrics(clean, noisy, filtered, sr)
    print(f"{fname:<12} SNR: {m['snr_before_dB']:6.2f} → {m['snr_after_dB']:6.2f}  "
          f"Δ={m['snr_improvement_dB']:+.2f}  r={m['correlation']:.3f}")
    all_metrics.append(m)

import numpy as np
print(f"\nOrt. ΔSNR  : {np.mean([m['snr_improvement_dB'] for m in all_metrics]):+.2f} dB")
print(f"Ort. ΔRMSE : {np.mean([m['rmse_before']-m['rmse_after'] for m in all_metrics]):.5f}")
print(f"Ort. r     : {np.mean([m['correlation'] for m in all_metrics]):.3f}")