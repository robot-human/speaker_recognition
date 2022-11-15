import os
import math
import numpy as np
import matplotlib.pyplot as plt
import librosa
from env_variables import LOG_FILE_PATH, SAMPLE_RATE

def samples_to_seconds(samples, sample_rate):
    out = len(samples)/sample_rate
    print(out)

def plot_signal(signal,name):
    plt.plot(signal)
    plt.savefig(LOG_FILES_PATH+f"/{name}.png")

def trim_signal(signal, sample_rate, secs):
    n_samples = math.floor(secs*sample_rate)
    signal = signal[:n_samples]
    return signal

def get_samples(speaker_files, speaker_id):
    path_list = speaker_files[speaker_id]
    train = []
    valid = []
    test = []
    for i in range(8):
        train_samples, _ = librosa.load(path_list[i], mono=True, sr=SAMPLE_RATE)
        valid_samples, _ = librosa.load(path_list[i+8], mono=True, sr=SAMPLE_RATE)
        test_samples, _ = librosa.load(path_list[i+16], mono=True, sr=SAMPLE_RATE)
        train.extend(train_samples)
        valid.extend(valid_samples)
        test.extend(test_samples)
    return np.array(train),np.array(valid),np.array(test)