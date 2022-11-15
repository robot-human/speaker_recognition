import random
import numpy as np
import time
import feature_extraction as feats
import classification_models as models
from read_files import get_speaker_files, get_speaker_signals_dict
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from env_variables import DATABASE_PATH, N_SPEAKERS, SAMPLE_RATE, NFFT, FRAMES_ATTR, MFCC_ATTR, P, N_CODEWORDS, EPOCHS, N_MIXTURES, execution_times


random.seed(10)


MODELS_LIST = ['SVM']
FEATURES_LIST = ['LPC']


start_time = time.time()
ids, speaker_files = get_speaker_files(DATABASE_PATH)
speaker_ids = random.sample(ids, k=N_SPEAKERS)
signal_dict = get_speaker_signals_dict(speaker_files, speaker_ids)
end_time = time.time()

execution_times['File reading'] = end_time - start_time

start_time = time.time()
plp_filters = feats.get_PLP_filters(SAMPLE_RATE, NFFT)
window_frames = feats.get_window_frames_dict(speaker_ids, signal_dict , FRAMES_ATTR)
pow_frames = feats.get_pow_frames_dict(speaker_ids, window_frames, NFFT)
end_time = time.time()

execution_times['Pre-processing'] = end_time - start_time

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
    execution_times['MFCC'] = end_time - start_time
#model_svm = svm.SVC(kernel='rbf')
#model_svm.fit(scaled_train,classes)

    if("VQ" in MODELS_LIST):
        print("MFFC with VQ")
        start_time = time.time()
        models.run_VQ_model(speaker_ids, features)
        end_time = time.time()
        execution_times['VQ MFCC'] = end_time - start_time
        print("")

    if("SVM" in MODELS_LIST):
        print("MFFC with SVM")
        start_time = time.time()
        models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
        end_time = time.time()
        execution_times['SVM MFCC'] = end_time - start_time
        print("")

    if("GMM" in MODELS_LIST):
        print("MFFC with GMM")
        start_time = time.time()
        models.run_GMM_model(speaker_ids, features, scaler)
        end_time = time.time()
        execution_times['GMM MFCC'] = end_time - start_time
        print("")

# ########################################################################################################
# ## LPC
if("LPC" in FEATURES_LIST):
    print("LPC")
    start_time = time.time()
    features = feats.get_lpc_feats(speaker_ids, window_frames, P)
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
    execution_times['LPC'] = end_time - start_time
#model_svm = svm.SVC(kernel='rbf')
#model_svm.fit(scaled_train,classes)
    if("VQ" in MODELS_LIST):
        print("LPC with VQ")
        start_time = time.time()
        models.run_VQ_model(speaker_ids, features)
        end_time = time.time()
        execution_times['VQ LPC'] = end_time - start_time
        print("")

    if("SVM" in MODELS_LIST):
        print("LPC with SVM")
        start_time = time.time()
        models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
        end_time = time.time()
        execution_times['SVM LPC'] = end_time - start_time
        print("")

    if("GMM" in MODELS_LIST):
        print("LPC with GMM")
        start_time = time.time()
        models.run_GMM_model(speaker_ids, features, scaler)
        end_time = time.time()
        execution_times['GMM LPC'] = end_time - start_time
        print("")
# ########################################################################################################
# ## PLP
if("PLP" in FEATURES_LIST):
    print("PLP")
    start_time = time.time()
    features = feats.get_plp_feats(speaker_ids, pow_frames, P, plp_filters)
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
    execution_times['PLP'] = end_time - start_time
#model_svm = svm.SVC(kernel='rbf')
#model_svm.fit(scaled_train,classes)
    if("VQ" in MODELS_LIST):
        print("PLP with VQ")
        start_time = time.time()
        models.run_VQ_model(speaker_ids, features)
        end_time = time.time()
        execution_times['VQ PLP'] = end_time - start_time
        print("")

    if("SVM" in MODELS_LIST):
        print("PLP with SVM")
        start_time = time.time()
        models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
        end_time = time.time()
        execution_times['SVM PLP'] = end_time - start_time
        print("")

    if("GMM" in MODELS_LIST):
        print("PLP with GMM")
        start_time = time.time()
        models.run_GMM_model(speaker_ids, features, scaler)
        end_time = time.time()
        execution_times['GMM PLP'] = end_time - start_time
        print("")
    
print(execution_times)