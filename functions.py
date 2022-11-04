import matplotlib.pyplot as plt

def samples_to_seconds(samples, sample_rate):
    out = samples/sample_rate
    print(out)

def plot_signal(signal):
    plt.plot(signal)
    plt.show()