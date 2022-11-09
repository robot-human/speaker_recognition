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


ids, speaker_files = get_speaker_files(DATABASE_PATH)
speaker_ids = random.sample(ids, k=N_SPEAKERS)
signal_dict = get_speaker_signals_dict(speaker_files, speaker_ids)
plp_filters = feats.get_PLP_filters(SAMPLE_RATE, NFFT)
              
window_frames = feats.get_window_frames_dict(speaker_ids, signal_dict , FRAMES_ATTR)
pow_frames = feats.get_pow_frames_dict(speaker_ids, window_frames, NFFT)


########################################################################################################
## MFCC
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
model_svm = svm.SVC(kernel='rbf')
model_svm.fit(scaled_train,classes)


print("MFFC with VQ")
models.run_VQ_model(speaker_ids, features)

print("MFFC with SVM")
for enum, id in enumerate(speaker_ids):
    test_data = scaler.transform(features[id]['test'])
    test_classes = model_svm.predict(test_data)
    counts = np.bincount(test_classes)
    speaker = np.argmax(counts)
    print(speaker)
print("")


print("MFFC with GMM")
scaled_separate_set = []
for id in speaker_ids:
    scaled_separate_set.append(scaler.transform(features[id]['train']))

speaker_gm_models = []
for sp in scaled_separate_set:
    gm = GaussianMixture(n_components=N_MIXTURES, random_state=0).fit(sp)
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
model_svm = svm.SVC(kernel='rbf')
model_svm.fit(scaled_train,classes)

print("LPC with VQ")
speaker_models = []
for enum, id in enumerate(speaker_ids):
    codebook, classes = models.vector_quantization_trainning(features[id]['train'], N_CODEWORDS, EPOCHS)
    speaker_models.append(codebook)
    
for id in speaker_ids:
    speaker = -1
    dist = 1/0.00000001
    for enum, speaker_model in enumerate(speaker_models):
        classes = models.assign_classes(features[id]['test'], speaker_model)
        speaker_dist = models.featureset_distortion(features[id]['test'], classes, speaker_model)
        if(speaker_dist < dist):
            dist = speaker_dist
            speaker = enum
    print(speaker)
print("")

print("LPC with SVM")
for enum, id in enumerate(speaker_ids):
    test_data = scaler.transform(features[id]['test'])
    test_classes = model_svm.predict(test_data)
    counts = np.bincount(test_classes)
    speaker = np.argmax(counts)
    print(speaker)
print("")


print("LPC with GMM")
scaled_separate_set = []
for id in speaker_ids:
    scaled_separate_set.append(scaler.transform(features[id]['train']))

speaker_gm_models = []
for sp in scaled_separate_set:
    gm = GaussianMixture(n_components=N_MIXTURES, random_state=0).fit(sp)
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
# ## PLP
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
model_svm = svm.SVC(kernel='rbf')
model_svm.fit(scaled_train,classes)

print("PLP with VQ")
speaker_models = []
for enum, id in enumerate(speaker_ids):
    codebook, classes = models.vector_quantization_trainning(features[id]['train'], N_CODEWORDS, EPOCHS)
    speaker_models.append(codebook)
    
for id in speaker_ids:
    speaker = -1
    dist = 1/0.00000001
    for enum, speaker_model in enumerate(speaker_models):
        classes = models.assign_classes(features[id]['test'], speaker_model)
        speaker_dist = models.featureset_distortion(features[id]['test'], classes, speaker_model)
        if(speaker_dist < dist):
            dist = speaker_dist
            speaker = enum
    print(speaker)
print("")

print("PLP with SVM")
for enum, id in enumerate(speaker_ids):
    test_data = scaler.transform(features[id]['test'])
    test_classes = model_svm.predict(test_data)
    counts = np.bincount(test_classes)
    speaker = np.argmax(counts)
    print(speaker)
print("")


print("PLP with GMM")
scaled_separate_set = []
for id in speaker_ids:
    scaled_separate_set.append(scaler.transform(features[id]['train']))

speaker_gm_models = []
for sp in scaled_separate_set:
    gm = GaussianMixture(n_components=N_MIXTURES, random_state=0).fit(sp)
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