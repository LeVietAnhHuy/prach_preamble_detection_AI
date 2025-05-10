import numpy as np
import torch
import time

# Parameters
num_signals = 50
signal_length = 1228800
num_layers = 10
signals_per_layer = num_signals // num_layers

# Generate test signals and reshape to 3D
signals_np = np.random.randn(num_signals, signal_length).astype(np.complex64)
signals_np_3d = signals_np.reshape(num_layers, signals_per_layer, signal_length)
signals_torch = torch.from_numpy(signals_np_3d).to('cuda')

### --- NumPy (CPU) --- ###
print("=== NumPy (CPU) ===")

# FFT
start = time.time()
fft_np = np.fft.fft(signals_np_3d, axis=2)
end = time.time()
print(f"FFT: {end - start:.2f} seconds")

# IFFT
start = time.time()
ifft_np = np.fft.ifft(fft_np, axis=2)
end = time.time()
print(f"IFFT: {end - start:.2f} seconds")

# FFTShift + IFFTShift
start = time.time()
fftshift_np = np.fft.fftshift(fft_np, axes=2)
ifftshift_np = np.fft.ifftshift(fftshift_np, axes=2)
end = time.time()
print(f"FFTShift + IFFTShift: {end - start:.2f} seconds")

### --- PyTorch (GPU) --- ###
print("\n=== PyTorch (GPU) ===")

# Warm-up
_ = torch.fft.fft(signals_torch, dim=2)
torch.cuda.synchronize()

# FFT
torch.cuda.synchronize()
start = time.time()
fft_torch = torch.fft.fft(signals_torch, dim=2)
torch.cuda.synchronize()
end = time.time()
print(f"FFT: {end - start:.2f} seconds")

# IFFT
torch.cuda.synchronize()
start = time.time()
ifft_torch = torch.fft.ifft(fft_torch, dim=2)
torch.cuda.synchronize()
end = time.time()
print(f"IFFT: {end - start:.2f} seconds")

# FFTShift + IFFTShift
torch.cuda.synchronize()
start = time.time()
fftshift_torch = torch.fft.fftshift(fft_torch, dim=2)
ifftshift_torch = torch.fft.ifftshift(fftshift_torch, dim=2)
torch.cuda.synchronize()
end = time.time()
print(f"FFTShift + IFFTShift: {end - start:.2f} seconds")
