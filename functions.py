import matplotlib.pyplot as plt
log_files_path = "/media/Data/pedro_tesis/log_files" 

def samples_to_seconds(samples, sample_rate):
    out = samples/sample_rate
    print(out)

def plot_signal(signal,name):
    plt.plot(signal)
    plt.savefig(log_files_path+f"/{name}.png")