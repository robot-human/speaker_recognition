import os
import math
import numpy as np
import matplotlib.pyplot as plt
import librosa

log_files_path = os.getcwd() + "/log_files" 

def samples_to_seconds(samples, sample_rate):
    out = len(samples)/sample_rate
    print(out)

def plot_signal(signal,name):
    plt.plot(signal)
    plt.savefig(log_files_path+f"/{name}.png")

def trim_signal(signal, sample_rate, secs):
    n_samples = math.floor(secs*sample_rate)
    signal = signal[:n_samples]
    return signal

def get_samples(speaker_files,speaker_id,samples_limit):
    path_list = speaker_files[speaker_id]
    samples_num = 0
    idx = 0
    output_samples = []
    while((samples_num < samples_limit) and (idx < len(path_list))):
        samples, sample_rate = librosa.load(path_list[idx], mono=True, sr=8000)
        samples_num += len(samples)
        idx += 1
        output_samples.extend(samples)
    return np.array(output_samples)