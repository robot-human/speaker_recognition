import random
import feature_extraction as feats
from read_files import get_speaker_files, get_speaker_signals_dict

log_files_path = "/media/Data/pedro_tesis/log_files" 
database_path = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
N_SPEAKERS = 4
SAMPLE_RATE = 10000
random.seed(10)

ids, speaker_files = get_speaker_files(database_path)
speaker_ids = random.sample(ids, k=N_SPEAKERS)
signal_dict = get_speaker_signals_dict(speaker_files, speaker_ids)



window_frames_dict = {}
attr = {
    "VAD_TRESHOLD": 0.012,
    "PRE_EMPHASIS_COEF": 0.95,
    "FRAME_IN_SECS": 0.025,
    "OVERLAP_IN_SECS": 0.01,
    "WINDOW": 'hanning'
}
for id in speaker_ids:
    speaker_dict = {}
    speaker_dict['train'] = feats.voice_signal_processing(signal_dict[id]['train'], SAMPLE_RATE, attr)
    speaker_dict['valid'] = feats.voice_signal_processing(signal_dict[id]['valid'], SAMPLE_RATE, attr)
    speaker_dict['test'] = feats.voice_signal_processing(signal_dict[id]['test'], SAMPLE_RATE, attr)
    window_frames_dict[id] = speaker_dict
