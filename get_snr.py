import numpy as np

def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculate SNR (in dB) for complex-valued signal and noise."""
    power_signal = np.mean(np.abs(signal)**2)
    power_noise = np.mean(np.abs(noise)**2)
    snr = power_signal / power_noise
    return 10 * np.log10(snr)

def snr_db_from_received(received, clean):
    noise = received - clean
    return snr_db(clean, noise)