import math

N_SPEAKERS_LIST = [10]
SAMPLE_RATE = 10000
NFFT = 512

SIGNAL_DURATION_IN_SECONDS = 10.0
VAD_TRESHOLD = 0.02
LOG_FILE_PATH = "/media/Data/pedro_tesis/speaker_recognition/log_files/"
DATABASE_PATH = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
N_VECTOR_SAMPLES = math.floor(SIGNAL_DURATION_IN_SECONDS*SAMPLE_RATE)
WINDOW_SIZE = 0.025

GENERAL = {
    "N_SPEAKERS" : 0,  
    "SIGNAL_DURATION_IN_SECONDS" : SIGNAL_DURATION_IN_SECONDS,
}
EXECUTION_TIMES = {
    'File reading' : 0,
    'Pre-processing' : 0,
    'MFB' : 0,
    'MFCC' : 0,
    'LPC' : 0,
    'PLP' : 0,
    'VQ MFB' : 0,
    'GMM MFB' : 0,
    'SVM MFB' : 0,
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
    "SAMPLE_RATE": SAMPLE_RATE,
    "VAD_TRESHOLD": VAD_TRESHOLD,
    "PRE_EMPHASIS_COEF": 0.95,
    "FRAME_IN_SECS": WINDOW_SIZE,
    "OVERLAP_IN_SECS": 0.01,
    "WINDOW": 'hanning'
}
MFCC_ATTR = {
    "NFFT": NFFT,
    "SAMPLE_RATE": SAMPLE_RATE,
    "N_FILT": 40,
    "N_CEPS": 22,
    "CEP_LIFTER": 22
}
LPC_ATTR = {
    "P" : 12
}
PLP_ATTR = {
    "P" : 12
}
MODEL_ATTR = {
    "VQ" : {"N_CODEWORDS" : 40, "EPOCHS" : 50},
    "GMM" : {"N_MIXTURES" : min(350,math.floor(0.75*WINDOW_SIZE*SAMPLE_RATE)), "EPOCHS" : 5000},
    "SVM" : {"EPOCHS" : 1000}
}
