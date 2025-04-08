import numpy as np
import matplotlib.pyplot as plt

# Define signals
x = np.array([1, 2, 3, 4])
y = np.array([1, 0, 0, 0, 0, 5])

# Linear cross-correlation (ground truth)
linear_corr = np.correlate(x, y, mode='full')

# Circular correlation using FFT without zero-padding
N = max(len(x), len(y))  # too short!
X = np.fft.fft(x, n=N)
Y = np.fft.fft(y, n=N)
circular_corr = np.fft.ifft(X * np.conj(Y))
circular_corr = np.real(circular_corr)

# Circular correlation using FFT with proper zero-padding
N_pad = len(x) + len(y) - 1  # to avoid wrap-around
X_pad = np.fft.fft(x, n=N_pad)
Y_pad = np.fft.fft(y, n=N_pad)
fixed_corr = np.fft.ifft(X_pad * np.conj(Y_pad))
fixed_corr = np.real(fixed_corr)

# Plotting
# plt.figure()
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

ax1.stem(linear_corr)
ax1.title("Linear Correlation\nnp.correlate (ground truth)")
ax1.xlabel("Lag")

ax2.stem(circular_corr)
ax2.title("Circular Correlation\nFFT (no padding) — WRAPPED")
ax2.xlabel("Lag")

ax3.stem(fixed_corr)
ax3.title("FFT-based Correlation\nWith Zero Padding (Correct)")
plt.xlabel("Lag")

plt.tight_layout()
plt.show()
