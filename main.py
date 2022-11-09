import random
import numpy as np
import feature_extraction as feats
import classification_models as models
from read_files import get_speaker_files, get_speaker_signals_dict
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

log_files_path = "/media/Data/pedro_tesis/log_files" 
database_path = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
N_SPEAKERS = 6
SAMPLE_RATE = 10000
random.seed(10)


frames_attr = {
    "NFFT": 512,
    "sample_rate": 10000,
    "VAD_TRESHOLD": 0.012,
    "PRE_EMPHASIS_COEF": 0.95,
    "FRAME_IN_SECS": 0.025,
    "OVERLAP_IN_SECS": 0.01,
    "WINDOW": 'hanning'
}
mfcc_attr={
    "NFFT": 512,
    "sample_rate": 10000,
    "n_filt": 22,
    "num_ceps": 22,
    "cep_lifter": 22
}


ids, speaker_files = get_speaker_files(database_path)
speaker_ids = random.sample(ids, k=N_SPEAKERS)
signal_dict = get_speaker_signals_dict(speaker_files, speaker_ids)
plp_filters = feats.get_PLP_filters(SAMPLE_RATE, 512)
              
window_frames = feats.get_window_frames_dict(speaker_ids, signal_dict , frames_attr)
pow_frames = feats.get_pow_frames_dict(speaker_ids, window_frames, 512)


########################################################################################################
## MFCC
print("MFCC")
features = feats.get_mfcc_feats(speaker_ids, pow_frames, mfcc_attr)
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
model_svm = svm.SVC(kernel='rbf')
model_svm.fit(scaled_train,classes)


# print("MFFC with VQ")
# n_codewords = 50
# epochs = 50
# speaker_models = []
# for enum, id in enumerate(speaker_ids):
#     codebook, classes = models.vector_quantization_trainning(features[id]['train'], n_codewords, epochs)
#     speaker_models.append(codebook)
    
# for id in speaker_ids:
#     speaker = -1
#     dist = 1/0.00000001
#     for enum, speaker_model in enumerate(speaker_models):
#         classes = models.assign_classes(features[id]['test'], speaker_model)
#         speaker_dist = models.featureset_distortion(features[id]['test'], classes, speaker_model)
#         if(speaker_dist < dist):
#             dist = speaker_dist
#             speaker = enum
#     print(speaker)
# print("")

# print("MFFC with SVM")
# for enum, id in enumerate(speaker_ids):
#     test_data = scaler.transform(features[id]['test'])
#     test_classes = model_svm.predict(test_data)
#     counts = np.bincount(test_classes)
#     speaker = np.argmax(counts)
#     print(speaker)
# print("")


print("MFFC with GMM")
N_COMPONENTS = 200
scaled_separate_set = []
for id in speaker_ids:
    print(len(features[id]['train'][0]))
    scaled_separate_set.append(scaler.transform(features[id]['train']))

speaker_gm_models = []
for sp in scaled_separate_set:
    gm = GaussianMixture(n_components=N_COMPONENTS, random_state=0).fit(sp)
    speaker_gm_models.append(gm)

for id in speaker_ids:
    dist = -1/0.000000001
    speaker = -1
    test_data = scaler.transform(features[id]['test'])
    for enum, model in enumerate(speaker_gm_models):
        speaker_dist = model.score(test_data)
        if(speaker_dist > dist):
            dist = speaker_dist
            speaker = enum
    print(speaker)
print("")

# ########################################################################################################
# ## LPC
# print("LPC with SVM")
# features = feats.get_lpc_feats(speaker_ids, window_frames, 12)
# classes = []
# train_set = []
# for enum, id in enumerate(speaker_ids):
#     for val in features[id]['train']:
#         train_set.extend(features[id]['train'])
#         for i in range(len(features[id]['train'])):
#             classes.append(enum)



# scaler = StandardScaler()
# scaler.fit(train_set)
# scaled_train = scaler.transform(train_set)
# model = svm.SVC(kernel='rbf')
# model.fit(scaled_train,classes)

# for enum, id in enumerate(speaker_ids):
#     test_data = scaler.transform(features[id]['test'])
#     test_classes = model.predict(test_data)
#     counts = np.bincount(test_classes)
#     speaker = np.argmax(counts)
#     print(speaker)
# print("")
# ########################################################################################################
# ## PLP
# print("PLP with SVM")
# features = feats.get_plp_feats(speaker_ids, pow_frames, 12, plp_filters)
# classes = []
# train_set = []
# for enum, id in enumerate(speaker_ids):
#     for val in features[id]['train']:
#         train_set.extend(features[id]['train'])
#         for i in range(len(features[id]['train'])):
#             classes.append(enum)



# scaler = StandardScaler()
# scaler.fit(train_set)
# scaled_train = scaler.transform(train_set)
# model = svm.SVC(kernel='rbf')
# model.fit(scaled_train,classes)

# for enum, id in enumerate(speaker_ids):
#     test_data = scaler.transform(features[id]['test'])
#     test_classes = model.predict(test_data)
#     counts = np.bincount(test_classes)
#     speaker = np.argmax(counts)
#     print(speaker)