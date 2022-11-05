import random
from read_files import get_speaker_files, get_speaker_signals_dict

log_files_path = "/media/Data/pedro_tesis/log_files" 
database_path = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
N_SPEAKERS = 4
random.seed(10)

ids, speaker_files = get_speaker_files(database_path)
speaker_ids = random.sample(ids, k=N_SPEAKERS)
signal_dict = get_speaker_signals_dict(speaker_files, speaker_ids)

for id in speaker_ids:
    print(signal_dict[id]['test'])
