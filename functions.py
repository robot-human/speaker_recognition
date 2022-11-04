import os
import math
import matplotlib.pyplot as plt

log_files_path = os.getcwd() + "/log_files" 

def samples_to_seconds(samples, sample_rate):
    out = samples/sample_rate
    print(out)

def plot_signal(signal,name):
    plt.plot(signal)
    plt.savefig(log_files_path+f"/{name}.png")

def trim_signal(signal, sample_rate, secs):
    n_samples = math.floor(secs*sample_rate)
    signal = signal[:n_samples]
    return signal