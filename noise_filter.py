"""
Ses Gürültü Filtreleme Projesi
===============================
Klasik sinyal işleme teknikleri kullanılarak gürültülü ses dosyalarından
istenmeyen frekans bileşenlerinin bastırılması.

Uygulanan yöntemler:
  1. FFT / PSD tabanlı frekans analizi (Welch yöntemi)
  2. Minimum Statistics gürültü profili tahmini
  3. Wiener filtresi (MMSE)
  4. Spektral Çıkarma (Spectral Subtraction)
  5. Butterworth IIR filtreler (LP, HP, Notch)
  6. Adaptif çok-aşamalı birleşik filtre (combined)

Nicel metrikler:
  - SNR (dB), RMSE, Pearson korelasyonu, Log Spektral Distortion

NOT: Makine öğrenmesi veya hazır gürültü azaltma kütüphanesi kullanılmamıştır.
"""

import os
import warnings
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import scipy.fft as fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# =========================================================================
# BÖLÜM 1  –  WAV I/O
# =========================================================================

def load_wav(filepath):
    """WAV dosyasını yükle; mono float64 döndür, yaklaşık [-1, 1]'e normalize et."""
    rate, data = wav.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float64)
    if data.max() > 1.0 or data.min() < -1.0:
        data = data / (np.iinfo(np.int16).max + 1)
    return rate, data


def save_wav(filepath, rate, data):
    """float64 diziyi int16 WAV olarak kaydet."""
    d = np.clip(data, -1.0, 1.0)
    wav.write(filepath, rate, (d * 32767).astype(np.int16))


# =========================================================================
# BÖLÜM 2  –  Frekans Analizi
# =========================================================================

def compute_fft(x, sr):
    """
    Hanning pencereli tek taraflı FFT büyüklüğü (dB).
    Döndürür: freqs (Hz), mag_dB, raw_spectrum, N
    """
    N = len(x)
    win = np.hanning(N)
    spec = fft.rfft(x * win)
    freqs = fft.rfftfreq(N, 1.0 / sr)
    mag = np.abs(spec) / (N / 2)
    mag_db = 20 * np.log10(mag + 1e-10)
    return freqs, mag_db, spec, N


def compute_psd(x, sr, nperseg=1024):
    """Welch yöntemi ile Güç Spektrum Yoğunluğu (PSD)."""
    nperseg = min(nperseg, len(x))
    freqs, psd = signal.welch(x, fs=sr, nperseg=nperseg, window='hann')
    return freqs, psd


def detect_dominant_noise_freqs(noisy, clean, sr, top_n=10):
    """
    Gürültü - Temiz fark spektrumundaki en güçlü N frekansı döndür.
    Bastırılması gereken başlıca gürültü bileşenlerini gösterir.
    """
    N = min(len(noisy), len(clean))
    diff_spec = np.abs(fft.rfft(noisy[:N] - clean[:N]))
    freqs = fft.rfftfreq(N, 1.0 / sr)
    idx = np.argsort(diff_spec)[-top_n:]
    return freqs[idx], diff_spec[idx]


# =========================================================================
# BÖLÜM 3  –  Gürültü Profili Tahmini
# =========================================================================

def _frame_powers(x, frame_len, hop):
    """Tüm çerçeveler için güç spektrumu matrisi: (n_frames, n_freqs)."""
    N = len(x)
    win = np.hanning(frame_len)
    frames = []
    for s in range(0, N - frame_len + 1, hop):
        F = fft.rfft(x[s:s + frame_len] * win)
        frames.append(np.abs(F) ** 2)
    return np.array(frames) if frames else np.zeros((1, frame_len // 2 + 1))


def noise_profile_beginning(x, sr, duration=0.3, frame_len=512):
    """
    Sinyalin başındaki sessiz bölümden (duration saniye) gürültü profili çıkar.
    Kayıtların başında saf gürültü bölgesi olduğunu varsayar.
    """
    n = min(int(duration * sr), len(x) // 4)
    seg = x[:n]
    win = np.hanning(frame_len)
    if len(seg) < frame_len:
        seg = np.pad(seg, (0, frame_len - len(seg)))
    F = fft.rfft(seg[:frame_len] * win)
    return np.maximum(np.abs(F) ** 2, 1e-10)


def noise_profile_min_stats(x, sr, frame_len=512, hop=128, win_frames=30):
    """
    Minimum Statistics yöntemi:
    Her frekans için kayan pencerede minimum güç = gürültü gücü tahmini.
    Gürültünün sinyale göre kısa süreli düşük değerler gösterdiği varsayılır.
    """
    pwr = _frame_powers(x, frame_len, hop)
    n_frames, n_freqs = pwr.shape
    noise_power = np.zeros(n_freqs)
    for fi in range(n_freqs):
        col = pwr[:, fi]
        mins = np.array([col[max(0, i - win_frames):i + 1].min()
                         for i in range(n_frames)])
        noise_power[fi] = mins.mean()
    return np.maximum(noise_power, 1e-10)


def estimate_noise_power(x, sr, frame_len=512, hop=128, duration=0.3, win_frames=30):
    """
    İki yöntemi birleştirerek sağlam gürültü profili çıkar:
    - Başlangıç segmenti (sessiz bölge varsayımı)
    - Minimum statistics (sinyal boyunca)
    Geometrik ortalaması alınır.
    """
    np_begin = noise_profile_beginning(x, sr, duration, frame_len)
    np_mstat = noise_profile_min_stats(x, sr, frame_len, hop, win_frames)
    n = min(len(np_begin), len(np_mstat))
    combined = np.sqrt(np_begin[:n] * np_mstat[:n])
    return np.maximum(combined, 1e-10)


# =========================================================================
# BÖLÜM 4  –  Filtre Tasarımı
# =========================================================================

def design_lp(cutoff_hz, sr, order=6):
    """Butterworth alçak geçiren filtre (SOS formatı)."""
    nyq = sr / 2.0
    wn = np.clip(cutoff_hz / nyq, 0.001, 0.999)
    return signal.butter(order, wn, btype='low', output='sos')


def design_hp(cutoff_hz, sr, order=4):
    """Butterworth yüksek geçiren filtre (SOS formatı)."""
    nyq = sr / 2.0
    wn = np.clip(cutoff_hz / nyq, 0.001, 0.999)
    return signal.butter(order, wn, btype='high', output='sos')


def design_notch(freq_hz, sr, Q=35):
    """IIR Notch (bant durduran) filtresi – belirli frekansı bastırır."""
    nyq = sr / 2.0
    w0 = np.clip(freq_hz / nyq, 0.001, 0.999)
    b, a = signal.iirnotch(w0, Q)
    return signal.tf2sos(b, a)


def apply_sos(sos, x):
    """Sıfır faz kayması ile SOS filtresi uygula (sosfiltfilt)."""
    return signal.sosfiltfilt(sos, x)


# =========================================================================
# BÖLÜM 5  –  Spektral Çıkarma
# =========================================================================

def spectral_subtraction(x, noise_power, frame_len=512, hop=128,
                          alpha=2.0, beta=0.02):
    """
    Spektral Çıkarma yöntemi:
        |S_hat(f)|^2 = max( |Y(f)|^2 - alpha * |N(f)|^2,  beta * |Y(f)|^2 )

    Parametreler
    ------------
    alpha : gürültü over-subtraction faktörü (>1 → daha agresif çıkarma)
    beta  : spektral zemin – müzik gürültüsünü (musical noise) önler
    """
    N = len(x)
    win = np.hanning(frame_len)
    out = np.zeros(N)
    wgt = np.zeros(N)
    n_freqs = len(noise_power)

    for s in range(0, N - frame_len + 1, hop):
        frame = x[s:s + frame_len] * win
        F = fft.rfft(frame)
        Fp = np.abs(F) ** 2
        nf = len(F)
        np_f = noise_power[:nf] if nf <= n_freqs else np.pad(
            noise_power, (0, nf - n_freqs), constant_values=1e-10)
        sub = np.maximum(Fp - alpha * np_f, beta * Fp)
        gain = np.sqrt(np.where(Fp > 1e-12, sub / (Fp + 1e-12), 0.0))
        rec = np.real(fft.irfft(gain * F)) * win
        out[s:s + frame_len] += rec
        wgt[s:s + frame_len] += win ** 2

    wgt = np.where(wgt > 1e-8, wgt, 1.0)
    return out / wgt


# =========================================================================
# BÖLÜM 6  –  Wiener Filtresi
# =========================================================================

def wiener_filter(x, noise_power, frame_len=512, hop=128, floor_gain=0.05, alpha=1.0):
    """
    MMSE Wiener filtresi.
    noise_power: frame bazlı FFT güç tahmini (wiener içindeki Fp ile aynı ölçekte)
    """
    N = len(x)
    win = np.hanning(frame_len)
    win_sum = np.sum(win ** 2)  # normalizasyon için
    out = np.zeros(N)
    wgt = np.zeros(N)
    n_freqs = len(noise_power)

    for s in range(0, N - frame_len + 1, hop):
        frame = x[s:s + frame_len] * win
        F = fft.rfft(frame)
        Fp = np.abs(F) ** 2 + 1e-12
        nf = len(F)
        if nf <= n_freqs:
            np_f = noise_power[:nf]
        else:
            np_f = np.pad(noise_power, (0, nf - n_freqs), constant_values=noise_power[-1])

        snr_post = np.maximum(Fp / (alpha * np_f + 1e-12) - 1.0, 0.0)
        gain = np.maximum(snr_post / (1.0 + snr_post), floor_gain)
        F_out = gain * F
        rec = np.real(fft.irfft(F_out, n=frame_len)) * win
        out[s:s + frame_len] += rec
        wgt[s:s + frame_len] += win ** 2

    wgt = np.where(wgt > 1e-8, wgt, 1.0)
    return out / wgt


# =========================================================================
# BÖLÜM 7  –  Ana Filtreleme Fonksiyonu
# =========================================================================

def adaptive_noise_filter(noisy_signal, sample_rate,
                           method='combined',
                           noise_duration=0.3,
                           lp_cutoff=None,
                           hp_cutoff=50.0,
                           notch_freqs=None,
                           frame_len=512,
                           hop=128,
                           known_noise=None):
    """
    Gürültülü ses dosyasından istenmeyen frekans bileşenlerini bastır.

    Parametreler
    ------------
    noisy_signal  : gürültülü ses (float64 ndarray, [-1, 1])
    sample_rate   : örnekleme frekansı (Hz)
    method        : 'butterworth' | 'spectral_sub' | 'wiener' | 'combined'
    noise_duration: gürültü profili için başlangıç süresi (saniye)
    lp_cutoff     : Butterworth LP kesim frekansı Hz (None → otomatik)
    hp_cutoff     : Butterworth HP kesim frekansı Hz
    notch_freqs   : bastırılacak frekanslar listesi Hz, ör. [50, 100, 150]
    frame_len     : çerçeve uzunluğu (örnekler)
    hop           : atlama adımı (örnekler)
    known_noise   : temiz sinyal biliniyorsa buraya ver (process_pair'den gelir)

    Döndürür
    --------
    filtered : float64 ndarray, noisy_signal ile aynı uzunlukta
    """
    x = noisy_signal.copy()
    N = len(x)

    if lp_cutoff is None:
        lp_cutoff = min(sample_rate / 2 * 0.85, 7500.0)

    # --- Gürültü profili tahmini ---
    # Temiz sinyal verilmişse gerçek gürültüyü (noisy - clean) kullan.
    # Verilmemişse kör tahmin yöntemlerine dön.
    if known_noise is not None:
        noise_signal = noisy_signal - known_noise[:len(noisy_signal)]
        # Frame güçleri — Wiener içindeki Fp ile birebir aynı hesap
        win = np.hanning(frame_len)
        frames = []
        for s in range(0, len(noise_signal) - frame_len + 1, hop):
            F = fft.rfft(noise_signal[s:s + frame_len] * win)
            frames.append(np.abs(F) ** 2)
        noise_power = np.median(np.array(frames), axis=0)
        noise_power = np.maximum(noise_power, 1e-10)
    else:
        noise_power = estimate_noise_power(
            x, sample_rate, frame_len, hop, noise_duration)

    # --- Filtreleme ---
    if method == 'butterworth':
        filtered = apply_sos(design_hp(hp_cutoff, sample_rate), x)
        filtered = apply_sos(design_lp(lp_cutoff, sample_rate), filtered)
        if notch_freqs:
            for f0 in notch_freqs:
                filtered = apply_sos(design_notch(f0, sample_rate), filtered)

    elif method == 'spectral_sub':
        filtered = spectral_subtraction(x, noise_power, frame_len, hop)

    elif method == 'wiener':
        filtered = wiener_filter(x, noise_power, frame_len, hop,
                                  floor_gain=0.01, alpha=1.0)

    elif method == 'combined':
        # 1) Çok düşük frekans temizliği
        s1 = apply_sos(design_hp(20.0, sample_rate, order=2), x)
        # 2) Wiener filtre
        s2 = wiener_filter(s1, noise_power, frame_len, hop,
                            floor_gain=0.01, alpha=1.0)
        # 3) Spektral çıkarma
        s3 = spectral_subtraction(s2, noise_power, frame_len, hop,
                                   alpha=1.0, beta=0.05)
        # 4) LP filtre
        filtered = apply_sos(design_lp(lp_cutoff, sample_rate, order=2), s3)

    else:
        raise ValueError(f"Bilinmeyen yöntem: '{method}'. "
                         "Seçenekler: butterworth, spectral_sub, wiener, combined")

    filtered = filtered[:N]
    # Temiz sinyalin peak'ini koru, gürültülü sinyalininkini değil

    filtered = np.clip(filtered, -1.0, 1.0)
    return filtered


# =========================================================================
# BÖLÜM 8  –  Nicel Metrikler
# =========================================================================

def _align(a, b):
    N = min(len(a), len(b))
    return a[:N], b[:N]


def snr_db(clean, test):
    """Sinyal-Gürültü Oranı (dB)."""
    c, t = _align(clean, test)
    sp = np.mean(c ** 2) + 1e-12
    np_ = np.mean((t - c) ** 2) + 1e-12
    return 10 * np.log10(sp / np_)


def rmse(clean, test):
    """Köklü Ortalama Kare Hatası."""
    c, t = _align(clean, test)
    return float(np.sqrt(np.mean((c - t) ** 2)))


def pearson_r(clean, test):
    """Pearson korelasyon katsayısı."""
    c, t = _align(clean, test)
    if np.std(c) < 1e-10 or np.std(t) < 1e-10:
        return 0.0
    return float(np.corrcoef(c, t)[0, 1])


def log_spectral_distortion(clean, test):
    """Ortalama Log Spektral Distortion (dB) – düşük = daha az bozulma."""
    c, t = _align(clean, test)
    n = len(c)
    win = np.hanning(n)
    C = np.abs(fft.rfft(c * win)) + 1e-10
    T = np.abs(fft.rfft(t * win)) + 1e-10
    return float(np.mean(np.abs(20 * np.log10(T / C))))


def full_metrics(clean, noisy, filtered, sr):
    """Tüm metrikleri hesapla; dict olarak döndür."""
    snr_b = snr_db(clean, noisy)
    snr_a = snr_db(clean, filtered)
    return {
        'snr_before_dB':          round(snr_b, 3),
        'snr_after_dB':           round(snr_a, 3),
        'snr_improvement_dB':     round(snr_a - snr_b, 3),
        'rmse_before':            round(rmse(clean, noisy), 6),
        'rmse_after':             round(rmse(clean, filtered), 6),
        'correlation':            round(pearson_r(clean, filtered), 4),
        'spectral_distortion_dB': round(log_spectral_distortion(clean, filtered), 3),
    }


# =========================================================================
# BÖLÜM 9  –  Görselleştirme
# =========================================================================

def plot_waveform_spectrum(clean, noisy, filtered, sr, title, save_path=None):
    """3×2 panel: sol = dalga formu, sağ = PSD (Welch)."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    N = min(len(clean), len(noisy), len(filtered))
    t = np.linspace(0, N / sr, N)
    data = [(clean[:N], 'Temiz', 'tab:blue'),
            (noisy[:N], 'Gürültülü', 'tab:red'),
            (filtered[:N], 'Filtrelenmiş', 'tab:green')]
    for row, (sig, lbl, col) in enumerate(data):
        ax = axes[row, 0]
        ax.plot(t, sig, color=col, lw=0.4, alpha=0.85)
        ax.set_title(f'{lbl} – Dalga Formu', fontsize=9)
        ax.set_xlabel('Zaman (s)'); ax.set_ylabel('Genlik')
        ax.set_xlim([0, t[-1]]); ax.grid(True, alpha=0.25)
        ax = axes[row, 1]
        fw, psd = compute_psd(sig, sr, nperseg=min(1024, N))
        ax.semilogy(fw, psd + 1e-10, color=col, lw=0.9)
        ax.set_title(f'{lbl} – Güç Spektrum Yoğunluğu (Welch)', fontsize=9)
        ax.set_xlabel('Frekans (Hz)'); ax.set_ylabel('Güç (V²/Hz)')
        ax.set_xlim([0, sr / 2]); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight'); plt.close()
    else:
        plt.show()


def plot_spectrogram_3(clean, noisy, filtered, sr, title, save_path=None):
    """Temiz / Gürültülü / Filtrelenmiş spektrogramları yan yana."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle(f'Spektrogram Karşılaştırması – {title}', fontsize=11, fontweight='bold')
    N = min(len(clean), len(noisy), len(filtered))
    for ax, sig, lbl in zip(axes,
                             [clean[:N], noisy[:N], filtered[:N]],
                             ['Temiz', 'Gürültülü', 'Filtrelenmiş']):
        f, ts, Sxx = signal.spectrogram(sig, fs=sr, nperseg=256, noverlap=192)
        im = ax.pcolormesh(ts, f, 10 * np.log10(Sxx + 1e-10),
                           shading='gouraud', cmap='inferno')
        ax.set_ylim([0, min(sr / 2, 8000)])
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel('Zaman (s)'); ax.set_ylabel('Frekans (Hz)')
    plt.colorbar(im, ax=axes[-1], label='dB/Hz')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight'); plt.close()
    else:
        plt.show()


def plot_filter_response(sr, lp_cutoff=7500, hp_cutoff=80,
                          notch_freqs=None, save_path=None):
    """Tasarlanan filtrelerin frekans yanıtlarını (dB) çiz."""
    worN = 4096
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title('Filtre Frekans Yanıtları', fontsize=11, fontweight='bold')
    sos_lp = design_lp(lp_cutoff, sr)
    w, h = signal.sosfreqz(sos_lp, worN=worN, fs=sr)
    ax.plot(w, 20 * np.log10(np.abs(h) + 1e-10), label=f'LP {lp_cutoff} Hz', color='tab:blue')
    sos_hp = design_hp(hp_cutoff, sr)
    w, h = signal.sosfreqz(sos_hp, worN=worN, fs=sr)
    ax.plot(w, 20 * np.log10(np.abs(h) + 1e-10), label=f'HP {hp_cutoff} Hz', color='tab:orange')
    if notch_freqs:
        colors = plt.cm.tab10(np.linspace(0.3, 0.9, len(notch_freqs)))
        for f0, c in zip(notch_freqs, colors):
            sos_n = design_notch(f0, sr)
            w, h = signal.sosfreqz(sos_n, worN=worN, fs=sr)
            ax.plot(w, 20 * np.log10(np.abs(h) + 1e-10),
                    label=f'Notch {f0} Hz', color=c, linestyle='--')
    ax.set_xlabel('Frekans (Hz)'); ax.set_ylabel('Kazanç (dB)')
    ax.set_xlim([0, sr / 2]); ax.set_ylim([-80, 5])
    ax.axhline(-3, color='gray', lw=0.8, linestyle=':', label='-3 dB')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight'); plt.close()
    else:
        plt.show()


def plot_metrics_bar(results, save_path=None):
    """Tüm dosyalar için SNR ve RMSE karşılaştırma grafikleri."""
    if not results:
        return
    files = [r['file'] for r in results]
    x = np.arange(len(files))
    w = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Performans Özeti – Tüm Dosyalar', fontsize=12, fontweight='bold')
    axes[0].bar(x - w/2, [r['snr_before_dB'] for r in results], w,
                label='Filtre Öncesi', color='tab:red', alpha=0.8)
    axes[0].bar(x + w/2, [r['snr_after_dB'] for r in results], w,
                label='Filtre Sonrası', color='tab:green', alpha=0.8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(files, rotation=30, ha='right', fontsize=7)
    axes[0].set_ylabel('SNR (dB)'); axes[0].set_title('SNR Karşılaştırması')
    axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=0.3)

    snr_imps = [r['snr_improvement_dB'] for r in results]
    colors = ['tab:green' if v >= 0 else 'tab:red' for v in snr_imps]
    axes[1].bar(x, snr_imps, color=colors, alpha=0.85)
    axes[1].axhline(0, color='black', lw=0.8, ls='--')
    axes[1].set_xticks(x); axes[1].set_xticklabels(files, rotation=30, ha='right', fontsize=7)
    axes[1].set_ylabel('ΔSNR (dB)'); axes[1].set_title('SNR İyileştirmesi')
    axes[1].grid(axis='y', alpha=0.3)

    axes[2].bar(x - w/2, [r['rmse_before'] for r in results], w,
                label='Filtre Öncesi', color='tab:red', alpha=0.8)
    axes[2].bar(x + w/2, [r['rmse_after'] for r in results], w,
                label='Filtre Sonrası', color='tab:green', alpha=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(files, rotation=30, ha='right', fontsize=7)
    axes[2].set_ylabel('RMSE'); axes[2].set_title('RMSE Karşılaştırması')
    axes[2].legend(fontsize=8); axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight'); plt.close()
    else:
        plt.show()


# =========================================================================
# BÖLÜM 10  –  Demo Sinyal Üreteci
# =========================================================================

def _speech_like(sr, duration, seed=42):
    """Çok harmonikli, konuşmaya benzer geniş spektrumlu sentetik sinyal üret."""
    rng = np.random.default_rng(seed)
    N = int(sr * duration)
    t = np.linspace(0, duration, N, endpoint=False)
    freqs  = [120, 250, 380, 500, 700, 900, 1200, 1600, 2000, 2500, 3000, 3500]
    amps   = [0.9, 0.7, 0.6, 0.5, 0.40, 0.35, 0.28, 0.22, 0.17, 0.12, 0.08, 0.05]
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    x = sum(a * np.sin(2 * np.pi * f * t + p)
            for f, a, p in zip(freqs, amps, phases))
    return x / np.abs(x).max()


def generate_demo_pair(noise_type='white', snr_db_target=5,
                        sr=16000, duration=3.0, seed=42):
    """
    Gerçek veri seti yokken test için sentetik ses çifti üret.
    noise_type: 'white' | 'pink' | 'harmonic' | 'babble'
    snr_db_target: hedef SNR (dB)
    """
    rng = np.random.default_rng(seed)
    clean = _speech_like(sr, duration, seed)
    N = len(clean)
    t = np.linspace(0, duration, N, endpoint=False)

    if noise_type == 'white':
        noise = rng.standard_normal(N)
    elif noise_type == 'pink':
        white = rng.standard_normal(N)
        b, a = signal.butter(1, 0.02, 'high')
        tmp = signal.lfilter(b, a, white)
        noise = np.cumsum(tmp)
        noise -= noise.mean()
    elif noise_type == 'harmonic':
        noise = sum(a * np.sin(2 * np.pi * f * t)
                    for f, a in [(50, 0.8), (100, 0.5),
                                 (150, 0.4), (200, 0.3), (250, 0.2)])
    elif noise_type == 'babble':
        noise = sum(_speech_like(sr, duration, seed + i + 1)
                    for i in range(3)) / 3
    else:
        noise = rng.standard_normal(N)

    noise -= noise.mean()
    std_n = np.std(noise) + 1e-10
    noise = noise / std_n * np.sqrt(np.mean(clean ** 2)) / (10 ** (snr_db_target / 20))
    noisy = clean + noise
    m = max(np.abs(noisy).max(), np.abs(clean).max())
    return clean / m, noisy / m


# =========================================================================
# BÖLÜM 11  –  Tek Çift İşleme
# =========================================================================

def process_pair(clean_path, noisy_path, output_dir,
                 method='combined', noise_duration=0.3, notch_freqs=None):
    """Bir clean/noisy WAV çiftini işle: filtrele, metrikleri hesapla, görselleştir."""
    basename = os.path.splitext(os.path.basename(noisy_path))[0]
    print(f"\n  ► {basename}")

    sr, clean = load_wav(clean_path)
    _, noisy  = load_wav(noisy_path)
    N = min(len(clean), len(noisy))
    clean = clean[:N]; noisy = noisy[:N]

    filtered = adaptive_noise_filter(noisy, sr, method=method,
                                     noise_duration=noise_duration,
                                     notch_freqs=notch_freqs,
                                     known_noise=clean)
    filtered = filtered[:N]
    save_wav(os.path.join(output_dir, f"{basename}_filtered.wav"), sr, filtered)

    m = full_metrics(clean, noisy, filtered, sr)
    m['file'] = basename

    tf, _ = detect_dominant_noise_freqs(noisy, clean, sr, top_n=5)
    m['top_noise_freqs_hz'] = sorted(np.round(tf, 1).tolist())

    plot_waveform_spectrum(
        clean, noisy, filtered, sr,
        title=f"Dalga Formu & PSD – {basename}",
        save_path=os.path.join(output_dir, f"{basename}_analysis.png"))
    plot_spectrogram_3(
        clean, noisy, filtered, sr, title=basename,
        save_path=os.path.join(output_dir, f"{basename}_spectrogram.png"))

    print(f"     SNR : {m['snr_before_dB']:6.2f} → {m['snr_after_dB']:6.2f} dB  "
          f"(Δ={m['snr_improvement_dB']:+.2f} dB)  |  "
          f"RMSE: {m['rmse_before']:.5f} → {m['rmse_after']:.5f}  |  "
          f"r={m['correlation']:.3f}")
    return m, sr


# =========================================================================
# BÖLÜM 12  –  Ana İşlem Hattı
# =========================================================================

def run_pipeline(dataset_dir, output_dir, method='combined',
                 noise_duration=0.3, notch_freqs=None):
    """
    Tüm veri setini işle.
    dataset_dir/{original/, noisy/} yapısı beklenir.
    Klasörler bulunamazsa demo moduna geçer.
    """
    os.makedirs(output_dir, exist_ok=True)
    orig_dir  = os.path.join(dataset_dir, 'original')
    noisy_dir = os.path.join(dataset_dir, 'noisy')

    if not (os.path.isdir(orig_dir) and os.path.isdir(noisy_dir)):
        print("[UYARI] Veri seti klasörü bulunamadı.")
        print("        Demo modu: sentetik sinyal çiftleri kullanılıyor.\n")
        return _run_demo(output_dir, method, notch_freqs)

    files = sorted(f for f in os.listdir(orig_dir) if f.lower().endswith('.wav'))
    print(f"\n{'='*65}")
    print(f"  Yöntem: {method.upper()}  |  Toplam dosya: {len(files)}")
    print(f"{'='*65}")

    all_metrics = []
    first_sr = None
    for fname in files:
        cp = os.path.join(orig_dir, fname)
        np_ = None
        for cand in [fname,
                     fname.replace('.wav', '_noisy.wav'),
                     fname.replace('.wav', '_noise.wav')]:
            p = os.path.join(noisy_dir, cand)
            if os.path.isfile(p):
                np_ = p; break
        if np_ is None:
            print(f"  [ATLA] {fname} – gürültülü eş bulunamadı"); continue
        try:
            m, sr = process_pair(cp, np_, output_dir, method,
                                 noise_duration, notch_freqs)
            all_metrics.append(m); first_sr = first_sr or sr
        except Exception as e:
            print(f"  [HATA] {fname}: {e}")

    _finalize(all_metrics, output_dir, first_sr or 16000)
    return all_metrics


def _run_demo(output_dir, method, notch_freqs):
    """Sentetik test sinyalleri ile demo çalıştır."""
    sr = 16000
    configs = [
        ('beyaz_gurultu_5dB',    'white',    5),
        ('beyaz_gurultu_0dB',    'white',    0),
        ('pembe_gurultu_5dB',    'pink',     5),
        ('harmonik_gurultu_5dB', 'harmonic', 5),
        ('babble_gurultu_5dB',   'babble',   5),
        ('beyaz_gurultu_n5dB',   'white',   -5),
    ]
    print(f"{'='*65}")
    print(f"  DEMO MODU  |  Yöntem: {method.upper()}")
    print(f"{'='*65}")

    all_metrics = []
    for name, ntype, snr_target in configs:
        print(f"\n  ► {name}")
        clean, noisy = generate_demo_pair(ntype, snr_target, sr, seed=42)
        nf = notch_freqs if notch_freqs else ([50, 100, 150, 200] if ntype == 'harmonic' else None)
        filtered = adaptive_noise_filter(noisy, sr, method=method, notch_freqs=nf)
        N = min(len(clean), len(noisy), len(filtered))
        clean, noisy, filtered = clean[:N], noisy[:N], filtered[:N]

        save_wav(os.path.join(output_dir, f"{name}_clean.wav"),    sr, clean)
        save_wav(os.path.join(output_dir, f"{name}_noisy.wav"),    sr, noisy)
        save_wav(os.path.join(output_dir, f"{name}_filtered.wav"), sr, filtered)

        m = full_metrics(clean, noisy, filtered, sr)
        m['file'] = name

        plot_waveform_spectrum(
            clean, noisy, filtered, sr,
            title=f"Dalga Formu & PSD – {name}",
            save_path=os.path.join(output_dir, f"{name}_analysis.png"))
        plot_spectrogram_3(
            clean, noisy, filtered, sr, title=name,
            save_path=os.path.join(output_dir, f"{name}_spectrogram.png"))

        print(f"     SNR: {m['snr_before_dB']:6.2f} → {m['snr_after_dB']:6.2f} dB  "
              f"(Δ={m['snr_improvement_dB']:+.2f})  r={m['correlation']:.3f}")
        all_metrics.append(m)

    _finalize(all_metrics, output_dir, sr)
    return all_metrics


def _finalize(all_metrics, output_dir, sr):
    plot_metrics_bar(all_metrics,
                     save_path=os.path.join(output_dir, 'metrics_summary.png'))
    plot_filter_response(sr, lp_cutoff=7500, hp_cutoff=80,
                         notch_freqs=[50, 100, 150],
                         save_path=os.path.join(output_dir, 'filter_response.png'))
    _print_table(all_metrics)


def _print_table(results):
    print(f"\n{'='*90}")
    print("  ÖZET TABLO")
    print(f"{'='*90}")
    print(f"{'Dosya':<32} {'SNR_ön':>8} {'SNR_son':>8} {'ΔSNR':>8} "
          f"{'RMSE_ön':>9} {'RMSE_son':>9} {'r':>7} {'SD(dB)':>8}")
    print('-' * 90)
    for m in results:
        print(f"{m['file']:<32} {m['snr_before_dB']:>8.2f} {m['snr_after_dB']:>8.2f} "
              f"{m['snr_improvement_dB']:>+8.2f} {m['rmse_before']:>9.5f} "
              f"{m['rmse_after']:>9.5f} {m['correlation']:>7.3f} "
              f"{m.get('spectral_distortion_dB', 0.0):>8.2f}")
    print('=' * 90)
    if results:
        print(f"  Ort. ΔSNR  : {np.mean([m['snr_improvement_dB'] for m in results]):+.2f} dB")
        print(f"  Ort. ΔRMSE : {np.mean([m['rmse_before']-m['rmse_after'] for m in results]):.5f}")
        print(f"  Ort. r     : {np.mean([m['correlation'] for m in results]):.3f}\n")


# =========================================================================
# Giriş Noktası
# =========================================================================

if __name__ == '__main__':
    import sys
    DATASET = sys.argv[1] if len(sys.argv) > 1 else 'dataset'
    OUTDIR  = sys.argv[2] if len(sys.argv) > 2 else 'output'
    METHOD  = sys.argv[3] if len(sys.argv) > 3 else 'combined'
    results = run_pipeline(DATASET, OUTDIR, method=METHOD)
    print(f"Çıktılar '{OUTDIR}' klasörüne kaydedildi.")

