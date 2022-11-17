import random
import numpy as np
import time
import csv
import datetime
import json
from sklearn.preprocessing import StandardScaler
import feature_extraction as feats
import classification_models as models
from read_files import get_speaker_files, get_speaker_signals_dict
from env_variables import GENERAL, EXECUTION_TIMES, FRAMES_ATTR, MFCC_ATTR, LPC_ATTR, PLP_ATTR, LOG_FILE_PATH, DATABASE_PATH

random.seed(10)
date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")


MODELS_LIST = ['GMM','VQ']
FEATURES_LIST = ['LPC']

results = {
    "Models" : MODELS_LIST,
    "Features" : FEATURES_LIST
}


start_time = time.time()
ids, speaker_files = get_speaker_files(DATABASE_PATH)
speaker_ids = random.sample(ids, k=GENERAL["N_SPEAKERS"])
signal_dict = get_speaker_signals_dict(speaker_files, speaker_ids)
end_time = time.time()
EXECUTION_TIMES['File reading'] = round(end_time - start_time,2)

for speaker in speaker_files.keys():
    print(speaker_files[speaker])

# start_time = time.time()
# plp_filters = feats.get_PLP_filters(FRAMES_ATTR["SAMPLE_RATE"], FRAMES_ATTR["NFFT"])
# window_frames = feats.get_window_frames_dict(speaker_ids, signal_dict , FRAMES_ATTR)
# pow_frames = feats.get_pow_frames_dict(speaker_ids, window_frames, FRAMES_ATTR["NFFT"])
# end_time = time.time()
# EXECUTION_TIMES['Pre-processing'] = round(end_time - start_time,2)

# print(speaker_ids)
# ########################################################################################################
# ## MFCC
# if("MFCC" in FEATURES_LIST):
#     start_time = time.time()
#     print("MFCC")
#     features, scaled_train, classes, scaler = feats.prepared_scaled_mfcc_feats(speaker_ids, pow_frames, MFCC_ATTR)
#     end_time = time.time()
#     EXECUTION_TIMES['MFCC'] = round(end_time - start_time,2)

#     if("VQ" in MODELS_LIST):
#         start_time = time.time()
#         print("MFFC with VQ")
#         models.run_VQ_model(speaker_ids, features)
#         end_time = time.time()
#         EXECUTION_TIMES['VQ MFCC'] = round(end_time - start_time,2)
#         print("")
    
#     if("GMM" in MODELS_LIST):
#         start_time = time.time()
#         print("MFFC with GMM")
#         models.run_GMM_model(speaker_ids, features, scaler)
#         end_time = time.time()
#         EXECUTION_TIMES['GMM MFCC'] = round(end_time - start_time,2)
#         print("")

#     if("SVM" in MODELS_LIST):
#         start_time = time.time()
#         print("MFFC with SVM")
#         models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
#         end_time = time.time()
#         EXECUTION_TIMES['SVM MFCC'] = round(end_time - start_time,2)
#         print("")
# # ########################################################################################################
# # ## LPC
# if("LPC" in FEATURES_LIST):
#     print("LPC")
#     start_time = time.time()
#     features, scaled_train, classes, scaler = feats.prepared_scaled_lpc_feats(speaker_ids, window_frames, LPC_ATTR)
#     end_time = time.time()
#     EXECUTION_TIMES['LPC'] = round(end_time - start_time,2)

#     if("VQ" in MODELS_LIST):
#         print("LPC with VQ")
#         start_time = time.time()
#         models.run_VQ_model(speaker_ids, features)
#         end_time = time.time()
#         EXECUTION_TIMES['VQ LPC'] = round(end_time - start_time,2)
#         print("")

#     if("GMM" in MODELS_LIST):
#         print("LPC with GMM")
#         start_time = time.time()
#         models.run_GMM_model(speaker_ids, features, scaler)
#         end_time = time.time()
#         EXECUTION_TIMES['GMM LPC'] = round(end_time - start_time,2)
#         print("")

#     if("SVM" in MODELS_LIST):
#         print("LPC with SVM")
#         start_time = time.time()
#         models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
#         end_time = time.time()
#         EXECUTION_TIMES['SVM LPC'] = round(end_time - start_time,2)
#         print("")
# # ########################################################################################################
# # ## PLP
# if("PLP" in FEATURES_LIST):
#     print("PLP")
#     start_time = time.time()
#     features, scaled_train, classes, scaler = feats.prepared_scaled_plp_feats(speaker_ids, pow_frames, PLP_ATTR, plp_filters)
#     end_time = time.time()
#     EXECUTION_TIMES['PLP'] = round(end_time - start_time,2)

#     if("VQ" in MODELS_LIST):
#         print("PLP with VQ")
#         start_time = time.time()
#         models.run_VQ_model(speaker_ids, features)
#         end_time = time.time()
#         EXECUTION_TIMES['VQ PLP'] = round(end_time - start_time,2)
#         print("")

#     if("GMM" in MODELS_LIST):
#         print("PLP with GMM")
#         start_time = time.time()
#         models.run_GMM_model(speaker_ids, features, scaler)
#         end_time = time.time()
#         EXECUTION_TIMES['GMM PLP'] = round(end_time - start_time,2)
#         print("")

#     if("SVM" in MODELS_LIST):
#         print("PLP with SVM")
#         start_time = time.time()
#         models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
#         end_time = time.time()
#         EXECUTION_TIMES['SVM PLP'] = round(end_time - start_time,2)
#         print("")


# file_path = LOG_FILE_PATH + f"session_{date}.json"
# results_dict = {
#     "General" : GENERAL,
#     "Frames" : FRAMES_ATTR,
#     "MFCC" : MFCC_ATTR, 
#     "LPC" : LPC_ATTR, 
#     "PLP" : PLP_ATTR,
#     "Results" : results
# }
# with open(file_path, 'w') as json_file:
#   json.dump(results_dict, json_file)

# for k in  EXECUTION_TIMES.keys():
#     print(f"{k} :  {EXECUTION_TIMES[k]}")