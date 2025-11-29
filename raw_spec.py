
# Raw Data & Spectrogram

import sys
import mne
import numpy as np

file_path = r"C:\Users\josia\Desktop\Signals\P7_ec_signal.npy"
data = np.load(file_path)
data = data.T
sfreq = 250  # sampling frequency in Hz

# Your channel names
channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
                 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 
                 'Fz', 'Cz', 'Pz', 'POz', 'FC1', 'FC2', 'CP1', 'CP2', 
                 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10', 'TP9', 'TP10']

n_channels = data.shape[0] #32
ch_names = channel_names
ch_types = ['eeg'] * n_channels

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(data, info)

# Plotting
raw.plot(n_channels=32, duration=5, scalings='auto', title='ec EEG Signals', show=True, block=True)