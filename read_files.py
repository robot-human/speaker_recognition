import os
import math
import random
import functions
import feature_extraction as feats
import librosa
import numpy as np

log_files_path = "/media/Data/pedro_tesis/log_files" 
database_path = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"

N_SPEAKERS = 1
SAMPLE_RATE = 10000
N_SAMPLES = SAMPLE_RATE*2



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

def get_speaker_signals_dict(speaker_files, speaker_ids):
    signal_dict = {}
    for id in speaker_ids:
        signal_data = {}
        train_samples,valid_samples,test_samples = functions.get_samples(speaker_files,id)
        train_vad_samples = functions.trim_signal(feats.vad(train_samples, 0.01),SAMPLE_RATE,1.0)
        valid_vad_samples = functions.trim_signal(feats.vad(valid_samples, 0.01),SAMPLE_RATE,1.0)
        test_vad_samples = functions.trim_signal(feats.vad(test_samples, 0.01),SAMPLE_RATE,1.0)

        signal_data['train'] = train_vad_samples
        signal_data['valid'] = valid_vad_samples
        signal_data['test'] = test_vad_samples
        signal_dict[id] = signal_data
    return signal_dict


ids, speaker_files = get_speaker_files(database_path)

random.seed(10)
speaker_ids = random.sample(ids, k=N_SPEAKERS)

signal_dict = get_speaker_signals_dict(speaker_files, speaker_ids)
functions.samples_to_seconds(signal_dict[speaker_ids[0]]['train'],SAMPLE_RATE)
functions.samples_to_seconds(signal_dict[speaker_ids[0]]['valid'],SAMPLE_RATE)
functions.samples_to_seconds(signal_dict[speaker_ids[0]]['test'],SAMPLE_RATE)

