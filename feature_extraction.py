import numpy as np
import math
from scipy.fftpack import dct



######################################################################################################
# Signal Information and Summary
def print_properties(s, sr):
    print("Shape        ",s.shape)
    print("Type         ",s.dtype)
    print("Sample rate  ",sr)
    print("Seconds      ",s.shape[0]/sr)
    return None


######################################################################################################
# Signal Processing
def vad(mono_signal, threshold, buff_size=1000):
    total_s = int(mono_signal.shape[0]/buff_size)
    signal = []
    for i in range(total_s):
        sig = mono_signal[i*buff_size:(i+1)*buff_size]
        rms = math.sqrt(np.square(sig).mean())
        if(rms > threshold):
            signal = np.append(signal,sig)
    return signal

def pre_emphasis(signal, pre_emphasis_coef):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis_coef * signal[:-1])
    return emphasized_signal

def framing(signal, sample_rate, frame_size, frame_stride):
    nsamples_signal = len(signal)
    nsamples_frame = int(sample_rate*frame_size)
    nsamples_stride = int(sample_rate*frame_stride)
    n_frames = int(np.ceil((nsamples_signal-nsamples_frame)/nsamples_stride) + 1)
    nsamples_padding = ((n_frames - 1)*nsamples_stride + nsamples_frame) - nsamples_signal
    z = np.zeros(nsamples_padding)
    signal = np.append(signal, z)
    frames = np.empty((n_frames,nsamples_frame))
    for i in range(n_frames):
        left = i*nsamples_stride
        right = left + nsamples_frame
        frame = signal[left:right]
        frames[i] = frame
    return frames

def window(frames, frame_length, window_type = 'hamming'):
    if(window_type == 'hanning'):
        out = frames*np.hanning(frame_length)
    elif(window_type == 'blackman'):
        out = frames*np.blackman(frame_length)
    else:
        out = frames*np.hamming(frame_length)
    return out

def frame_matrix(signal,sample_rate, pre_emphasis_coef, frame_in_secs, overlap_in_secs, window_type):
    emph_signal = pre_emphasis(signal, pre_emphasis_coef)
    frames = framing(emph_signal, sample_rate, frame_in_secs, overlap_in_secs)
    window_frames = window(frames, int(frame_in_secs*sample_rate), window_type)
    return window_frames

def voice_signal_processing(samples, attr):
    vad_signal = vad(samples,attr["VAD_TRESHOLD"])
    emph_signal = pre_emphasis(vad_signal, attr["PRE_EMPHASIS_COEF"])
    frames = framing(emph_signal, attr["sample_rate"], attr["FRAME_IN_SECS"], attr["OVERLAP_IN_SECS"])
    window_frames = window(frames, int(attr["FRAME_IN_SECS"]*attr["sample_rate"]), attr["WINDOW"])
    return window_frames

def get_window_frames_dict(speaker_ids, signal_dict, attr):
    window_frames_dict = {}
    for id in speaker_ids:
        speaker_dict = {}
        speaker_dict['train'] = voice_signal_processing(signal_dict[id]['train'], attr)
        speaker_dict['valid'] = voice_signal_processing(signal_dict[id]['valid'], attr)
        speaker_dict['test'] = voice_signal_processing(signal_dict[id]['test'], attr)
        window_frames_dict[id] = speaker_dict
    return window_frames_dict

def get_pow_frames_dict(speaker_ids, window_frames_dict, NFFT):
    pow_frames_dict = {}
    for id in speaker_ids:
        speaker_dict = {}
        speaker_dict['train'] = (np.absolute(np.fft.rfft(window_frames_dict[id]['train'], NFFT))** 2)/NFFT
        speaker_dict['valid'] = (np.absolute(np.fft.rfft(window_frames_dict[id]['valid'], NFFT))** 2)/NFFT
        speaker_dict['test'] = (np.absolute(np.fft.rfft(window_frames_dict[id]['test'], NFFT))** 2)/NFFT
        pow_frames_dict[id] = speaker_dict
    return pow_frames_dict
######################################################################################################
# MFCC
def freq_to_mel(freq):
    mel = 1127*math.log(1 + freq/700)
    return mel
def mel_to_freq(mel):
    freq = 700*(math.exp(mel/1127) - 1)
    return freq
def freq_to_bin(freq, nfft, sample_rate):
    bin_freq = int(np.floor(((nfft + 1)*freq/sample_rate)))
    return bin_freq
def triangular_filter(bin_freqs, n_bin, length):
    filt = np.zeros(length)
    l = int(bin_freqs[n_bin - 1])
    c = int(bin_freqs[n_bin])
    r = int(bin_freqs[n_bin + 1])
    for i in range(l, c):
        filt[i] = (i - l) / (c - l) 
    for i in range(c, r):
        filt[i] = (i - r) / (c - r)
    return filt
def filter_banks(bin_freqs, nfft):
    n_filt = len(bin_freqs) - 2
    length = int(np.floor(nfft / 2 + 1))
    fbank = np.zeros((n_filt,length))
    for i in range(1, n_filt + 1): 
        fbank[i - 1] = triangular_filter(bin_freqs, i, length)
    return fbank

def MFCC(pow_frames, attr):
    mels = np.linspace(freq_to_mel(0), freq_to_mel(attr["sample_rate"]/2), attr["n_filt"] + 2)
    freqs = [mel_to_freq(mel) for mel in mels]
    bin_freqs = [freq_to_bin(f, attr["NFFT"], attr["sample_rate"]) for f in freqs]
    fbank = filter_banks(bin_freqs,  attr["NFFT"])
    
    f_banks = np.dot(pow_frames, fbank.T)
    f_banks = np.where(f_banks == 0, np.finfo(float).eps, f_banks) 
    f_banks = 20 * np.log10(f_banks)
    mfcc = dct(f_banks, type=2, axis=1, norm='ortho')[:, 1 : (attr["num_ceps"] + 1)]

    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (attr["cep_lifter"] / 2) * np.sin(np.pi * n / attr["cep_lifter"])
    mfcc *= lift
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc

def get_mfcc_feats(speaker_ids, pow_frames_dict, attr):
    mfcc_dict = {}
    for id in speaker_ids:
        speaker_dict = {}
        speaker_dict['train'] = MFCC(pow_frames_dict[id]['train'], attr)
        speaker_dict['valid'] = MFCC(pow_frames_dict[id]['valid'], attr)
        speaker_dict['test'] = MFCC(pow_frames_dict[id]['test'], attr)
        mfcc_dict[id] = speaker_dict
    return mfcc_dict
######################################################################################################
# LPC
def correlations(frames, p, N):
    phi = np.zeros(p+1)
    for i in range(1,p+2):
        sigma = 0
        for m in range(N-i+1):
            sigma += frames[m]*frames[m+i-1]
        phi[i-1] = sigma
    return phi

def Levinson_Durbin(R, p):
    E = np.zeros(p+1)
    k = np.zeros(p)
    alpha = np.zeros((p,p))
    i = 1
    E[i-1] = R[i-1]
    k[i-1] = R[i]/E[i-1]
    alpha[i-1][i-1] = k[i-1]
    E[i] = (1 - k[i-1]**2)*E[i-1]

    for i in range(2,p+1):
        suma = 0
        for l in range(1,i):
            suma += alpha[i-2][l-1]*R[i-l]
        k[i-1] = (R[i] - suma)/E[i-1]
        alpha[i-1][i-1] = k[i-1]
        for j in range(1,i):
            alpha[i-1][j-1] = alpha[i-2][j-1] - k[i-1]*alpha[i-2][i-j-1]
        E[i] = (1 - k[i-1]**2)*E[i-1]
    return alpha[p-1].tolist()

def LPC(window_frames, p):
    lpc = []
    for n_frame in range(window_frames.shape[0]):
        R = correlations(window_frames[n_frame], p, window_frames.shape[1])
        lpc.append(Levinson_Durbin(R, p))
    return np.array(lpc)

def get_lpc_feats(speaker_ids, window_frames_dict, p):
    lpc_dict = {}
    for id in speaker_ids:
        speaker_dict = {}
        speaker_dict['train'] = LPC(window_frames_dict[id]['train'], p)
        speaker_dict['valid'] = LPC(window_frames_dict[id]['valid'], p)
        speaker_dict['test'] = LPC(window_frames_dict[id]['test'], p)
        lpc_dict[id] = speaker_dict
    return lpc_dict
######################################################################################################
# PLP
def sample_num_to_freq(x, sample_rate, n_points):
    return(x*sample_rate/(2*(n_points-1)))

def freq_to_angular_freq(freq):
    omega = 2*math.pi*freq
    return omega

def angular_freq_to_freq(afreq):
    freq = afreq/(2*math.pi)
    return freq

def freq_to_bark(freq):
    omega = freq_to_angular_freq(freq)
    coef = omega/(1200*math.pi)
    bark = 6*math.log(coef + (coef**2 + 1)**0.5)
    return bark

def bark_to_freq(bark):
    freq = angular_freq_to_freq(1200*math.pi*math.sinh(bark/6))
    return freq

def psi(omega):
    if(omega <= -0.5):
        y = 10**(2.5*(omega + 0.5))
    elif((omega > -0.5) and (omega < 0.5)):
        y = 1
    elif((omega >= 0.5) and (omega <= 2.5)):
        y = 10**(-1*(omega - 0.5))
    else:
        y = 0
    return y

def found_index(warped_label, index):
    left = warped_label[index] - 1.3
    right = warped_label[index] + 2.5
    left_index = index - 1
    while((left_index > 0) and (warped_label[left_index] > left)):
        left_index = left_index - 1
    right_index = index + 1
    while((right_index < len(warped_label)) and (warped_label[right_index] <= right)):
        right_index = right_index + 1
    if(left_index < 0):
        left_index = 0
    if(right_index >= len(warped_label)):
        right_index = len(warped_label)-1
    return(left_index,right_index)

def convolution(warped_label,pow_frame,index,left_index,right_index):
    sigma = 0
    for i in range(left_index,right_index+1):
        coef = warped_label[i] - warped_label[index]
        product = psi(coef)*pow_frame[i]
        sigma = sigma + product
    return sigma

def critical_band_convolution(warped_label,pow_frame):
    omega = []
    for index in range(len(warped_label)):
        left,right = found_index(warped_label, index)
        sigma = convolution(warped_label,pow_frame,index,left,right)
        omega.append(sigma)
    return omega

def equal_loudness_value(omega):
    k1 = 56.8*10**6
    k2 = 6.3*10**6
    k3 = 0.38*10**9
    omega_2 = omega**2
    omega_4 = omega**4
    num = (omega_2 + k1)*omega_4
    den = ((omega_2 + k2)**2)*(omega_2 + k3)
    return num/den

def equal_loudness(bark_convolution, afreq_label):
    eq_loudness = []
    for i in range(len(bark_convolution)):
        product = equal_loudness_value(afreq_label[i])*bark_convolution[i]
        eq_loudness.append(product)
    return eq_loudness

def amplitud_compression(eq_loudness):
    amp_compression = []
    for eq in eq_loudness:
        amp_compression.append(eq**(1/3))
    return amp_compression

def PLP_slow(pow_frames,  p, sample_rate):
    freq_label = np.linspace(0, sample_rate/2, len(pow_frames[0]))
    afreq_label = [freq_to_angular_freq(i) for i in freq_label]
    #sample_label = [i for i in range(len(freq_label))]
    warped_label = [freq_to_bark(i) for i in freq_label]
    
    perceptual_coeffs = []
    for frame in pow_frames:
        bark_convolution = critical_band_convolution(warped_label,frame)
        eq_loudness = equal_loudness(bark_convolution,afreq_label)
        amp_compression = amplitud_compression(eq_loudness)
        inverse_fourier = np.fft.irfft(amp_compression)
        perceptual_coeffs.append(inverse_fourier)
    
    plp = LPC(np.array(perceptual_coeffs), p)
    return plp

def get_PLP_filters(sample_rate, NFFT=512):
    n_samples = int(np.floor(NFFT / 2 + 1))
    x_samples = [i for i in range(n_samples)]
    x_freq = [sample_num_to_freq(i,sample_rate,n_samples) for i in x_samples]
    x_bark = [freq_to_bark(i) for i in x_freq]
    
    n_filters = int(freq_to_bark(sample_rate/2))+1
    filters = []
    for i in range(1,n_filters):
        array = np.array([psi(i-x) for x in x_bark])
        array = array*equal_loudness_value(freq_to_angular_freq(bark_to_freq(i)))/equal_loudness_value(freq_to_angular_freq(bark_to_freq(16)))
        filters.append(array)
    return filters

def PLP(pow_frames,  p, filters):
    perceptual_coeffs = []
    for frame in pow_frames:
        new_frame = []
        for filt in filters:
            new_frame.append((frame*filt).sum())
        amp_compression = amplitud_compression(new_frame)
        inverse_fourier = np.fft.irfft(amp_compression)
        perceptual_coeffs.append(inverse_fourier)
    plp = LPC(np.array(perceptual_coeffs), p)
    return plp

def get_plp_feats(speaker_ids, pow_frames_dict, p, filters):
    plp_dict = {}
    for id in speaker_ids:
        speaker_dict = {}
        speaker_dict['train'] = PLP(pow_frames_dict[id]['train'], p, filters)
        speaker_dict['valid'] = PLP(pow_frames_dict[id]['valid'], p, filters)
        speaker_dict['test'] = PLP(pow_frames_dict[id]['test'], p, filters)
        plp_dict[id] = speaker_dict
    return plp_dict