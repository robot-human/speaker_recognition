from os.path import isfile
import os
from configparser import ConfigParser
import math

cfg = None
                         
if not isfile(os.path.join(os.getcwd(), 'config.ini')):
    print("config file doesn't exist")
else:
    cfgParser = ConfigParser()
    cfgParser.read(os.path.join(os.getcwd(), 'config.ini'))

    N_SPEAKERS = int(cfgParser.get('config', 'N_SPEAKERS'))
    SAMPLE_RATE = int(cfgParser.get('config', 'SAMPLE_RATE'))
    NFFT = int(cfgParser.get('config', 'NFFT'))
    SIGNAL_DURATION_IN_SECONDS = float(cfgParser.get('config', 'SIGNAL_DURATION_IN_SECONDS'))
    VAD_TRESHOLD = float(cfgParser.get('config', 'VAD_TRESHOLD'))

    PRE_EMPHASIS_COEF = cfgParser.get('config', 'PRE_EMPHASIS_COEF').split(",")
    FRAME_IN_SECS = cfgParser.get('config', 'FRAME_IN_SECS').split(",")
    OVERLAP_PCT = cfgParser.get('config', 'OVERLAP_PCT').split(",")
    WINDOW = cfgParser.get('config', 'WINDOW')
    N_FILT = int(cfgParser.get('config', 'N_FILT'))
    N_CEPS = int(cfgParser.get('config', 'N_CEPS'))
    CEP_LIFTER = int(cfgParser.get('config', 'CEP_LIFTER'))

    P = int(cfgParser.get('config', 'P'))
    N_CODEWORDS = int(cfgParser.get('config', 'N_CODEWORDS'))
    EPOCHS_VQ = int(cfgParser.get('config', 'EPOCHS_VQ'))
    N_MIXTURES = int(cfgParser.get('config', 'N_MIXTURES'))
    EPOCHS_GMM = int(cfgParser.get('config', 'EPOCHS_GMM'))
    EPOCHS_SVM = int(cfgParser.get('config', 'EPOCHS_SVM'))

    GENERAL = {
        "N_SPEAKERS" : N_SPEAKERS,
        "SAMPLE_RATE": SAMPLE_RATE,
        "NFFT": NFFT,
        "SIGNAL_DURATION_IN_SECONDS" : SIGNAL_DURATION_IN_SECONDS
    }
    FRAMES_ATTR = {
        "SAMPLE_RATE": SAMPLE_RATE,
        "NFFT": NFFT,
        "VAD_TRESHOLD": VAD_TRESHOLD,
        "PRE_EMPHASIS_COEF": PRE_EMPHASIS_COEF,
        "FRAME_IN_SECS": FRAME_IN_SECS,
        "OVERLAP_PCT": OVERLAP_PCT,
        "WINDOW": WINDOW
    }
    MFCC_ATTR = {
        "NFFT": NFFT,
        "SAMPLE_RATE": SAMPLE_RATE,
        "N_FILT": N_FILT,
        "N_CEPS": N_CEPS,
        "CEP_LIFTER": CEP_LIFTER
    }
    LPC_ATTR = {
        "P" : P
    }
    PLP_ATTR = {
        "P" : P
    }
    MODEL_ATTR = {
        "VQ" : {"N_CODEWORDS" : N_CODEWORDS, "EPOCHS" : EPOCHS_VQ},
        "GMM" : {"N_MIXTURES" : 200, "EPOCHS" : EPOCHS_GMM},
        "SVM" : {"EPOCHS" : EPOCHS_SVM}
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

    RESULTS = {
    "Model" : "",
    "Features" : "",
    "Confusion matrix" : []
    }

    cfg = {
        "General" : GENERAL,
        "Frames" : FRAMES_ATTR,
        "MFB" : MFCC_ATTR,
        "MFCC" : MFCC_ATTR, 
        "LPC" : LPC_ATTR, 
        "PLP" : PLP_ATTR,
        "Model attr" : MODEL_ATTR,
        "Execution times" : EXECUTION_TIMES,
        "Results" : RESULTS
    }
    for k in (cfg.keys()):
        print(k)
        print(cfg[k])
        print("")