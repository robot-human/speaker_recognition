LOG_FILES_PATH = "/media/Data/pedro_tesis/log_files" 
DATABASE_PATH = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
N_SPEAKERS = 10
SAMPLE_RATE = 10000
N_SAMPLES = SAMPLE_RATE*2

FRAMES_ATTR = {
    "NFFT": 512,
    "sample_rate": 10000,
    "VAD_TRESHOLD": 0.012,
    "PRE_EMPHASIS_COEF": 0.95,
    "FRAME_IN_SECS": 0.025,
    "OVERLAP_IN_SECS": 0.01,
    "WINDOW": 'hanning'
}
MFCC_ATTR = {
    "NFFT": 512,
    "sample_rate": 10000,
    "n_filt": 22,
    "num_ceps": 22,
    "cep_lifter": 22
}

P = 12

