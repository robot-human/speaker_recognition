import os
import librosa
import functions
import feature_extraction as feats
import matplotlib.pyplot as plt

log_files_path = "/media/Data/pedro_tesis/log_files" 
database_path = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
database_dir = os.listdir(database_path)
N_SPEAKERS = 2

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


speakers_signal = {}
for i in range(N_SPEAKERS):
    samples = functions.get_samples(speaker_files,speaker_ids[i],4000000)
    vad_samples = feats.vad(samples, 0.01)
    speakers_signal[id] = vad_samples

