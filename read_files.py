import os
import math
import random
import functions
import feature_extraction as feats
from env_variables import SAMPLE_RATE, GENERAL, VAD_TRESHOLD


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
        train_vad_samples = functions.trim_signal(feats.vad(train_samples, VAD_TRESHOLD),SAMPLE_RATE,GENERAL["SIGNAL_DURATION_IN_SECONDS"])
        valid_vad_samples = functions.trim_signal(feats.vad(valid_samples, VAD_TRESHOLD),SAMPLE_RATE,GENERAL["SIGNAL_DURATION_IN_SECONDS"])
        test_vad_samples = functions.trim_signal(feats.vad(test_samples, VAD_TRESHOLD),SAMPLE_RATE,GENERAL["SIGNAL_DURATION_IN_SECONDS"])

        signal_data['train'] = train_vad_samples
        signal_data['valid'] = valid_vad_samples
        signal_data['test'] = test_vad_samples
        signal_dict[id] = signal_data
    return signal_dict

