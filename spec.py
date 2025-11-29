import numpy as np
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

file_path = r"C:\Users\josia\Desktop\Signals\P7_ec_signal.npy"
data = np.load(file_path)

data = data.T    # shape becomes (channels, samples)
fs = 250  # sampling frequency in Hz

# Channel names
channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
                 'Fz', 'Cz', 'Pz', 'POz', 'FC1', 'FC2', 'CP1', 'CP2', 
                 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10']

%matplotlib notebook

ch_name = "FC1"   # <<< pick your channel here

fft_window_sec = 2
fft_window_samples = fft_window_sec * fs
nfft = fft_window_samples
viewer_window_sec = 120
viewer_window_len = viewer_window_sec // fft_window_sec  # # of spectrogram columns

# Extract the channel
channel_data = data[channel_names.index(ch_name)]

# Compute FFT windows
num_windows = channel_data.shape[0] // fft_window_samples
segments = channel_data[:num_windows * fft_window_samples].reshape(num_windows, fft_window_samples)
fft_vals = np.fft.rfft(segments, axis=1)
power = np.abs(fft_vals)**2
power_db = 10 * np.log10(power + 1e-12)

# Frequency axis (0–45 Hz)
freqs = np.fft.rfftfreq(nfft, 1/fs)
freq_mask = freqs <= 45
freqs = freqs[freq_mask]
power_db = power_db[:, freq_mask]

# Time axis (each window = 2 sec)
spec_times = np.arange(num_windows) * fft_window_sec

# Initial time window
start_idx = 0
end_idx = start_idx + viewer_window_len
power_slice = power_db[start_idx:end_idx, :].T

# Plot
%matplotlib notebook
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

im = ax.imshow(
    power_slice, aspect='auto', origin='lower',
    extent=[spec_times[start_idx], spec_times[end_idx-1]+fft_window_sec,
            freqs[0], freqs[-1]]
)

ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (s)")
ax.set_title(f"Interactive Spectrogram - {ch_name}")
cbar = plt.colorbar(im, ax=ax, label='Power (dB)')

# Slider widget
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, 'Start (s)', 0, spec_times[-1] - viewer_window_sec, valinit=0)

def update(val):
    start_sec = slider.val
    start_idx = int(start_sec // fft_window_sec)
    end_idx = start_idx + viewer_window_len

    end_idx = min(end_idx, len(spec_times))  # avoid overflow

    power_slice = power_db[start_idx:end_idx, :].T

    im.set_data(power_slice)
    im.set_extent([
        spec_times[start_idx],
        spec_times[end_idx-1] + fft_window_sec,
        freqs[0],
        freqs[-1]
    ])
    ax.set_xlim(spec_times[start_idx], spec_times[end_idx-1] + fft_window_sec)

    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()


