import random
import numpy as np
import time
import csv
import datetime
from sklearn.preprocessing import StandardScaler
import feature_extraction as feats
import classification_models as models
from read_files import get_speaker_files, get_speaker_signals_dict
from env_variables import GENERAL, EXECUTION_TIMES, FRAMES_ATTR, MFCC_ATTR, LPC_ATTR, PLP_ATTR

FRAMES_ATTR["NFFT"]

random.seed(10)



date = datetime.datetime.now()

MODELS_LIST = ['GMM','VQ']
FEATURES_LIST = ['LPC']

results_file = open(GENERAL["LOG_FILE_PATH"], 'w')
writer = csv.writer(results_file)
writer.writerow(GENERAL["FILE_HEADER"])
results_file.close()

results_file = open(GENERAL["LOG_FILE_PATH"], 'a+')
writer = csv.writer(results_file)

start_time = time.time()
ids, speaker_files = get_speaker_files(GENERAL["DATABASE_PATH"])
speaker_ids = random.sample(ids, k=GENERAL["N_SPEAKERS"])
signal_dict = get_speaker_signals_dict(speaker_files, speaker_ids)
end_time = time.time()

EXECUTION_TIMES['File reading'] = round(end_time - start_time,2)

start_time = time.time()
plp_filters = feats.get_PLP_filters(FRAMES_ATTR["SAMPLE_RATE"], FRAMES_ATTR["NFFT"])
window_frames = feats.get_window_frames_dict(speaker_ids, signal_dict , FRAMES_ATTR)
pow_frames = feats.get_pow_frames_dict(speaker_ids, window_frames, FRAMES_ATTR["NFFT"])
end_time = time.time()

EXECUTION_TIMES['Pre-processing'] = round(end_time - start_time,2)

print(speaker_ids)
########################################################################################################
## MFCC
if("MFCC" in FEATURES_LIST):
    print("MFCC")
    start_time = time.time()
    features = feats.get_mfcc_feats(speaker_ids, pow_frames, MFCC_ATTR)
    classes = []
    train_set = []
    for enum, id in enumerate(speaker_ids):
        for val in features[id]['train']:
            train_set.extend(features[id]['train'])
            for i in range(len(features[id]['train'])):
                classes.append(enum)

    scaler = StandardScaler()
    scaler.fit(train_set)
    scaled_train = scaler.transform(train_set)
    end_time = time.time()
    EXECUTION_TIMES['MFCC'] = round(end_time - start_time,2)

    if("VQ" in MODELS_LIST):
        print("MFFC with VQ")
        start_time = time.time()
        models.run_VQ_model(speaker_ids, features)
        end_time = time.time()
        EXECUTION_TIMES['VQ MFCC'] = round(end_time - start_time,2)
        print("")
    
    if("GMM" in MODELS_LIST):
        print("MFFC with GMM")
        start_time = time.time()
        models.run_GMM_model(speaker_ids, features, scaler)
        end_time = time.time()
        EXECUTION_TIMES['GMM MFCC'] = round(end_time - start_time,2)
        print("")

    if("SVM" in MODELS_LIST):
        print("MFFC with SVM")
        start_time = time.time()
        models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
        end_time = time.time()
        EXECUTION_TIMES['SVM MFCC'] = round(end_time - start_time,2)
        print("")
# ########################################################################################################
# ## LPC
if("LPC" in FEATURES_LIST):
    print("LPC")
    start_time = time.time()
    features = feats.get_lpc_feats(speaker_ids, window_frames, LPC_ATTR["P"])
    classes = []
    train_set = []
    for enum, id in enumerate(speaker_ids):
        for val in features[id]['train']:
            train_set.extend(features[id]['train'])
            for i in range(len(features[id]['train'])):
                classes.append(enum)

    scaler = StandardScaler()
    scaler.fit(train_set)
    scaled_train = scaler.transform(train_set)
    end_time = time.time()
    EXECUTION_TIMES['LPC'] = round(end_time - start_time,2)

    if("VQ" in MODELS_LIST):
        str_data = []
        print("LPC with VQ")
        start_time = time.time()
        models.run_VQ_model(speaker_ids, features)
        end_time = time.time()
        EXECUTION_TIMES['VQ LPC'] = round(end_time - start_time,2)
        print("")

        str_data.append(date)
        str_data.append(GENERAL["N_SPEAKERS"])
        str_data.append(GENERAL["SIGNAL_DURATION_IN_SECONDS"])
        str_data.append("VQ")
        str_data.append("LPC")
        str_data.append(EXECUTION_TIMES['Pre-processing'])
        str_data.append("NA")
        str_data.append(EXECUTION_TIMES['LPC'])
        str_data.append("NA")
        str_data.append(EXECUTION_TIMES['VQ LPC'])
        str_data.append("NA")
        str_data.append("NA")

        writer.writerow(str_data)
    if("GMM" in MODELS_LIST):
        str_data = []
        print("LPC with GMM")
        start_time = time.time()
        models.run_GMM_model(speaker_ids, features, scaler)
        end_time = time.time()
        EXECUTION_TIMES['GMM LPC'] = round(end_time - start_time,2)
        print("")

        str_data.append(date)
        str_data.append(GENERAL["N_SPEAKERS"])
        str_data.append(GENERAL["SIGNAL_DURATION_IN_SECONDS"])
        str_data.append("GMM")
        str_data.append("LPC")
        str_data.append(EXECUTION_TIMES['Pre-processing'])
        str_data.append("NA")
        str_data.append(EXECUTION_TIMES['LPC'])
        str_data.append("NA")
        str_data.append("NA")
        str_data.append(EXECUTION_TIMES['GMM LPC'])
        str_data.append("NA")
        writer.writerow(str_data)

    if("SVM" in MODELS_LIST):
        print("LPC with SVM")
        start_time = time.time()
        models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
        end_time = time.time()
        EXECUTION_TIMES['SVM LPC'] = round(end_time - start_time,2)
        print("")
# ########################################################################################################
# ## PLP
if("PLP" in FEATURES_LIST):
    print("PLP")
    start_time = time.time()
    features = feats.get_plp_feats(speaker_ids, pow_frames, PLP_ATTR["P"], plp_filters)
    classes = []
    train_set = []
    for enum, id in enumerate(speaker_ids):
        for val in features[id]['train']:
            train_set.extend(features[id]['train'])
            for i in range(len(features[id]['train'])):
                classes.append(enum)

    scaler = StandardScaler()
    scaler.fit(train_set)
    scaled_train = scaler.transform(train_set)
    end_time = time.time()
    EXECUTION_TIMES['PLP'] = round(end_time - start_time,2)

    if("VQ" in MODELS_LIST):
        print("PLP with VQ")
        start_time = time.time()
        models.run_VQ_model(speaker_ids, features)
        end_time = time.time()
        EXECUTION_TIMES['VQ PLP'] = round(end_time - start_time,2)
        print("")

    if("GMM" in MODELS_LIST):
        print("PLP with GMM")
        start_time = time.time()
        models.run_GMM_model(speaker_ids, features, scaler)
        end_time = time.time()
        EXECUTION_TIMES['GMM PLP'] = round(end_time - start_time,2)
        print("")

    if("SVM" in MODELS_LIST):
        print("PLP with SVM")
        start_time = time.time()
        models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
        end_time = time.time()
        EXECUTION_TIMES['SVM PLP'] = round(end_time - start_time,2)
        print("")

results_file.close()
for k in  EXECUTION_TIMES.keys():
    print(f"{k} :  {EXECUTION_TIMES[k]}")