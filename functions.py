import os
import math
import numpy as np
import matplotlib.pyplot as plt
import librosa
from env_variables import LOG_FILE_PATH
from config import cfg

def samples_to_seconds(samples, sample_rate):
    out = len(samples)/sample_rate
    print(out)

def plot_signal(signal,name):
    plt.plot(signal)
    plt.savefig(LOG_FILE_PATH+f"/{name}.png")

def trim_signal(signal, sample_rate, secs):
    n_samples = math.floor(secs*sample_rate)
    signal = signal[:n_samples]
    return signal

def get_samples(speaker_files, speaker_id):
    path_list = speaker_files[speaker_id]
    samples = []
    total_seconds = 0
    for i in range(len(path_list)):
        speaker_samples, _ = librosa.load(path_list[i], mono=True, sr=cfg["General"]["SAMPLE_RATE"])
        total_seconds = total_seconds + (len(speaker_samples)*cfg["General"]["SAMPLE_RATE"])
        samples.extend(speaker_samples)
    print(total_seconds)
    return np.array(samples)