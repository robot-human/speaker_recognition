import random
import numpy as np
import feature_extraction as feats
from read_files import get_speaker_files, get_speaker_signals_dict
from sklearn import svm
from sklearn.preprocessing import StandardScaler

log_files_path = "/media/Data/pedro_tesis/log_files" 
database_path = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
N_SPEAKERS = 4
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
    "n_filt": 20,
    "num_ceps": 12,
    "cep_lifter": 22
}


ids, speaker_files = get_speaker_files(database_path)
speaker_ids = random.sample(ids, k=N_SPEAKERS)
signal_dict = get_speaker_signals_dict(speaker_files, speaker_ids)

window_frames = feats.get_window_frames_dict(speaker_ids, signal_dict , frames_attr)
pow_frames = feats.get_pow_frames_dict(speaker_ids, window_frames, 512)
mfcc = feats.get_mfcc_feats(speaker_ids, pow_frames, mfcc_attr)

classes = []
train_set = []
for enum, id in enumerate(speaker_ids):
    for val in mfcc[id]['train']:
        train_set.append(mfcc[id]['train'])
        classes.append(enum)

print(len(train_set),len(classes))
scaler = StandardScaler()
scaler.fit(train_set)
scaled_train = scaler.transform(train_set)
model = svm.SVC(kernel='rbf')
model.fit(scaled_train,classes)

# for enum, id in enumerate(speaker_ids):
#     test_data = scaler.transform(mfcc[id]['test'])
#     test_classes = model.predict(test_data)
#     counts = np.bincount(test_classes)
#     speaker = np.argmax(counts)
#     print(speaker)