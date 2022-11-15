LOG_FILES_PATH = "/media/Data/pedro_tesis/log_files" 
DATABASE_PATH = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"

N_SPEAKERS = 5
SIGNAL_DURATION_IN_SECONDS = 5.0

SAMPLE_RATE = 10000
NFFT = 512
VAD_TRESHOLD = 0.012
N_SAMPLES = SAMPLE_RATE*2

execution_times = {
    'File reading' : 0,
    'Pre-processing' : 0,
    'MFCC' : 0,
    'LPC' : 0,
    'PLP' : 0,
    'VQ MFCC' : 0,
    'GMM MFCC' : 0,
    'SVM MFCC' : 0,
    'VQ LPC' : 0,
    'GMM LPC' : 0,
    'SVM LPC' : 0,
    'VQ PLP' : 0,
    'GMM PLP' : 0,
    'SVM PLP' : 0
}

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

N_MIXTURES = 200
N_CODEWORDS = 400
EPOCHS = 50