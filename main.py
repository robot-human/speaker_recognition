import random
import numpy as np
import feature_extraction as feats
import classification_models as models
from read_files import get_speaker_files, get_speaker_signals_dict
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from env_variables import DATABASE_PATH, N_SPEAKERS, SAMPLE_RATE, NFFT, FRAMES_ATTR, MFCC_ATTR, P, N_CODEWORDS, EPOCHS, N_MIXTURES

random.seed(10)


MODELS_LIST = ['SVM']
FEATURES_LIST = ['LPC']

ids, speaker_files = get_speaker_files(DATABASE_PATH)
speaker_ids = random.sample(ids, k=N_SPEAKERS)
signal_dict = get_speaker_signals_dict(speaker_files, speaker_ids)
plp_filters = feats.get_PLP_filters(SAMPLE_RATE, NFFT)
              
window_frames = feats.get_window_frames_dict(speaker_ids, signal_dict , FRAMES_ATTR)
pow_frames = feats.get_pow_frames_dict(speaker_ids, window_frames, NFFT)

print(ids)
########################################################################################################
## MFCC
if("MFCC" in FEATURES_LIST):
    print("MFCC")
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
#model_svm = svm.SVC(kernel='rbf')
#model_svm.fit(scaled_train,classes)

    if("VQ" in MODELS_LIST):
        print("MFFC with VQ")
        models.run_VQ_model(speaker_ids, features)
        print("")

    if("SVM" in MODELS_LIST):
        print("MFFC with SVM")
        models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
        print("")

    if("GMM" in MODELS_LIST):
        print("MFFC with GMM")
        models.run_GMM_model(speaker_ids, features, scaler)
        print("")

# ########################################################################################################
# ## LPC
if("LPC" in FEATURES_LIST):
    print("LPC")
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
#model_svm = svm.SVC(kernel='rbf')
#model_svm.fit(scaled_train,classes)
    if("VQ" in MODELS_LIST):
        print("LPC with VQ")
        models.run_VQ_model(speaker_ids, features)
        print("")

    if("SVM" in MODELS_LIST):
        print("LPC with SVM")
        models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
        print("")

    if("GMM" in MODELS_LIST):
        print("LPC with GMM")
        models.run_GMM_model(speaker_ids, features, scaler)
        print("")
# ########################################################################################################
# ## PLP
if("PLP" in FEATURES_LIST):
    print("PLP")
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
#model_svm = svm.SVC(kernel='rbf')
#model_svm.fit(scaled_train,classes)
    if("VQ" in MODELS_LIST):
        print("PLP with VQ")
        models.run_VQ_model(speaker_ids, features)
        print("")

    if("SVM" in MODELS_LIST):
        print("PLP with SVM")
        models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
        print("")

    if("GMM" in MODELS_LIST):
        print("PLP with GMM")
        models.run_GMM_model(speaker_ids, features, scaler)
        print("")