import numpy as np

def selection_combining(rx_signal):
    """
    Select the antenna with highest total signal power.
    Input: rx_signal shape = (N_antennas, T)
    """
    powers = np.sum(np.abs(rx_signal) ** 2, axis=1)
    best_idx = np.argmax(powers)
    return rx_signal[best_idx]


def switch_combining(rx_signal, threshold_ratio=0.8):
    """
    Use the best antenna if its power is significantly better than others.
    Else switch to average of top 2 antennas.
    """
    powers = np.sum(np.abs(rx_signal) ** 2, axis=1)
    best_idx = np.argmax(powers)
    sorted_idx = np.argsort(powers)[::-1]
    ratio = powers[sorted_idx[1]] / powers[best_idx]

    if ratio < threshold_ratio:
        return rx_signal[best_idx]
    else:
        combined = (rx_signal[sorted_idx[0]] + rx_signal[sorted_idx[1]]) / 2
        return combined


def equal_gain_combining(rx_signal):
    """
    Align phase of each antenna and sum.
    """
    phase = np.angle(rx_signal)
    aligned = rx_signal * np.exp(-1j * phase)  # Remove individual phase
    combined = np.sum(aligned, axis=0)
    return combined


def rms_gain_combining(rx_signals):
    combined = np.zeros_like(rx_signals[0], dtype=np.complex64)
    for sig in rx_signals:
        rms = np.sqrt(np.mean(np.abs(sig)**2))
        combined += sig / rms
    return combined

