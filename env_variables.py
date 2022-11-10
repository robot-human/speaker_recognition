LOG_FILES_PATH = "/media/Data/pedro_tesis/log_files" 
DATABASE_PATH = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
SIGNAL_DURATION_IN_SECONDS = 1.0
VAD_TRESHOLD = 0.012
N_SPEAKERS = 30
SAMPLE_RATE = 10000
N_SAMPLES = SAMPLE_RATE*2
NFFT = 512

FRAMES_ATTR = {
    "NFFT": NFFT,
    "sample_rate": SAMPLE_RATE,
    "VAD_TRESHOLD": VAD_TRESHOLD,
    "PRE_EMPHASIS_COEF": 0.95,
    "FRAME_IN_SECS": 0.025,
    "OVERLAP_IN_SECS": 0.01,
    "WINDOW": 'hanning'
}
MFCC_ATTR = {
    "NFFT": NFFT,
    "sample_rate": SAMPLE_RATE,
    "n_filt": 22,
    "num_ceps": 22,
    "cep_lifter": 22
}

P = 12

N_MIXTURES = 50
N_CODEWORDS = 50
EPOCHS = 50