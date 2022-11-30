import os
import math
import random
import functions
import feature_extraction as feats
from config import cfg


def get_speaker_files(database_path):
    database_dir = os.listdir(database_path)
    speaker_files = {}
    speaker_ids = []
    for speaker in database_dir:
        speaker_path = database_path + "/" + speaker
        sessions_dir = os.listdir(speaker_path)
        speaker_files_list = []
        for session in sessions_dir:
            audio_files_path = speaker_path + "/" + session
            audio_files_dir = os.listdir(audio_files_path)
            for audio_file in audio_files_dir:
                if(audio_file.find(".flac") > 0):
                    speaker_files_list.append(audio_files_path+"/"+audio_file)
        speaker_ids.append(speaker)
        speaker_files[speaker] = speaker_files_list
    return speaker_ids, speaker_files

def get_speaker_signals_dict(speaker_files, speaker_ids, pct=1.0):
    n_vector_samples = math.floor(cfg["General"]["SIGNAL_DURATION_IN_SECONDS"]*cfg["General"]["SAMPLE_RATE"])
    signal_dict = {}
    for id in speaker_ids:
        signal_data = {}
        speaker_samples = functions.get_samples(speaker_files,id)
        vad_samples = feats.vad(speaker_samples, cfg["Frames"]["VAD_TRESHOLD"])
        signal_data['train'] = vad_samples[:n_vector_samples]
        vectors_left = math.floor((len(vad_samples) - n_vector_samples)/n_vector_samples)
        test_vectors = []
        for i in range(vectors_left):
            test_vectors.append(vad_samples[n_vector_samples*i:n_vector_samples*(i + 1)])
        signal_data['test'] = test_vectors[:math.floor(len(test_vectors)*pct)]
        signal_data['valid'] = test_vectors[math.floor(len(test_vectors)*pct):]
        signal_dict[id] = signal_data
    return signal_dict

