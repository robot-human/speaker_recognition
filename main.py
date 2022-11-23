import random
import numpy as np
import time
import csv
import math
import datetime
import json
from sklearn.preprocessing import StandardScaler
import feature_extraction as feats
import classification_models as models
from env_variables import LOG_FILE_PATH,DATABASE_PATH,MODELS_LIST,FEATURES_LIST,PRINT_FILES
from read_files import get_speaker_files, get_speaker_signals_dict
from config import cfg


random.seed(10)
date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")

ids, speaker_files = get_speaker_files(DATABASE_PATH)
n_speakers = cfg["General"]["N_SPEAKERS"]
speaker_ids = random.sample(ids, k=n_speakers)
start_time = time.time()
signal_dict = get_speaker_signals_dict(speaker_files, speaker_ids)
end_time = time.time()
cfg["Execution times"]['File reading'] = round(end_time - start_time,2)

results_dict = cfg
for pre_emph in cfg["Frames"]["PRE_EMPHASIS_COEF"]:
    for frame_size in cfg["Frames"]["FRAME_IN_SECS"]:
        for frame_overlap in cfg["Frames"]["OVERLAP_PCT"]:
            results_dict["Frames"]["PRE_EMPHASIS_COEF"] = float(pre_emph)
            results_dict["Frames"]["FRAME_IN_SECS"] = float(frame_size)
            results_dict["Frames"]["OVERLAP_PCT"] = float(frame_overlap)
            results_dict["Frames"]["OVERLAP_IN_SECS"] = float(frame_size)*float(frame_overlap)
            results_dict["Model attr"]["GMM"]["N_MIXTURES"] = min(350,math.floor(0.75*int(results_dict["General"]["SAMPLE_RATE"]*results_dict["Frames"]["FRAME_IN_SECS"])))
            print(results_dict["Model attr"]["GMM"]["N_MIXTURES"])
            print(f"Pre_emph: {pre_emph}, Frame size: {frame_size}, Frame overlap: {frame_overlap}")

            start_time = time.time()
            plp_filters = feats.get_PLP_filters(results_dict["General"]["SAMPLE_RATE"], results_dict["General"]["NFFT"])
            window_frames = feats.get_window_frames_dict(speaker_ids, signal_dict , results_dict["Frames"])
            pow_frames = feats.get_pow_frames_dict(speaker_ids, window_frames, results_dict["General"]["NFFT"])
            end_time = time.time()
            results_dict["Execution times"]['Pre-processing'] = round(end_time - start_time,2)

# print(speaker_ids)
# ########################################################################################################
## MFB
            if("MFB" in FEATURES_LIST):
                feature_name = "MFB"
                start_time = time.time()
                print("MFB")
                features, scaled_train, classes, scaler = feats.prepared_scaled_mfb_feats(speaker_ids, pow_frames, results_dict["MFCC"])
                end_time = time.time()
                results_dict["Execution times"]['MFB'] = round(end_time - start_time,2)

                if("VQ" in MODELS_LIST):
                    model_name = "VQ"
                    start_time = time.time()
                    print("MFB with VQ")
                    confusion_matrix = models.run_VQ_model(speaker_ids, features)
                    end_time = time.time()
                    results_dict["Execution times"]['VQ MFB'] = round(end_time - start_time,2)
                    print("")
            
                    results_dict["Results"]["Model"] = "VQ"
                    results_dict["Results"]["Features"] = "MFB"
                    results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                    if(PRINT_FILES):
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)
    
                if("GMM" in MODELS_LIST):
                    model_name = "GMM"
                    start_time = time.time()
                    print("MFB with GMM")
                    confusion_matrix = models.run_GMM_model(speaker_ids, features, scaler, results_dict["Model attr"]["GMM"])
                    end_time = time.time()
                    results_dict["Execution times"]['GMM MFB'] = round(end_time - start_time,2)
                    print("")

                    results_dict["Results"]["Model"] = "GMM"
                    results_dict["Results"]["Features"] = "MFB"
                    results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                    if(PRINT_FILES):
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)

                if("SVM" in MODELS_LIST):
                    model_name = "SVM"
                    start_time = time.time()
                    print("MFB with SVM")
                    confusion_matrix = models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
                    end_time = time.time()
                    results_dict["Execution times"]['SVM MFB'] = round(end_time - start_time,2)
                    print("")

                    results_dict["Results"]["Model"] = "SVM"
                    results_dict["Results"]["Features"] = "MFB"
                    results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                    if(PRINT_FILES):
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)
# ########################################################################################################
## MFCC
            if("MFCC" in FEATURES_LIST):
                feature_name = "MFCC"
                start_time = time.time()
                print("MFCC")
                features, scaled_train, classes, scaler = feats.prepared_scaled_mfcc_feats(speaker_ids, pow_frames, results_dict["MFCC"])
                end_time = time.time()
                results_dict["Execution times"]['MFCC'] = round(end_time - start_time,2)

                if("VQ" in MODELS_LIST):
                    model_name = "VQ"
                    start_time = time.time()
                    print("MFFC with VQ")
                    confusion_matrix = models.run_VQ_model(speaker_ids, features)
                    end_time = time.time()
                    results_dict["Execution times"]['VQ MFCC'] = round(end_time - start_time,2)
                    print("")
            
                    results_dict["Results"]["Model"] = "VQ"
                    results_dict["Results"]["Features"] = "MFCC"
                    results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                    if(PRINT_FILES):
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)
    
                if("GMM" in MODELS_LIST):
                    model_name = "GMM"
                    start_time = time.time()
                    print("MFFC with GMM")
                    confusion_matrix = models.run_GMM_model(speaker_ids, features, scaler, results_dict["Model attr"]["GMM"])
                    end_time = time.time()
                    results_dict["Execution times"]['GMM MFCC'] = round(end_time - start_time,2)
                    print("")

                    results_dict["Results"]["Model"] = "GMM"
                    results_dict["Results"]["Features"] = "MFCC"
                    results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                    if(PRINT_FILES):
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)

                if("SVM" in MODELS_LIST):
                    model_name = "SVM"
                    start_time = time.time()
                    print("MFFC with SVM")
                    confusion_matrix = models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
                    end_time = time.time()
                    results_dict["Execution times"]['SVM MFCC'] = round(end_time - start_time,2)
                    print("")

                    results_dict["Results"]["Model"] = "SVM"
                    results_dict["Results"]["Features"] = "MFCC"
                    results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                    if(PRINT_FILES):
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)
# ########################################################################################################
# ## LPC
            if("LPC" in FEATURES_LIST):
                feature_name = "LPC"
                print("LPC")
                start_time = time.time()
                features, scaled_train, classes, scaler = feats.prepared_scaled_lpc_feats(speaker_ids, window_frames, results_dict["LPC"])
                end_time = time.time()
                results_dict["Execution times"]['LPC'] = round(end_time - start_time,2)

                if("VQ" in MODELS_LIST):
                    model_name = "VQ"
                    print("LPC with VQ")
                    start_time = time.time()
                    confusion_matrix = models.run_VQ_model(speaker_ids, features)
                    end_time = time.time()
                    results_dict["Execution times"]['VQ LPC'] = round(end_time - start_time,2)
                    print("")

                    results_dict["Results"]["Model"] = "VQ"
                    results_dict["Results"]["Features"] = "LPC"
                    results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                    if(PRINT_FILES):
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)

                if("GMM" in MODELS_LIST):
                    model_name = "GMM"
                    print("LPC with GMM")
                    start_time = time.time()
                    confusion_matrix = models.run_GMM_model(speaker_ids, features, scaler, results_dict["Model attr"]["GMM"])
                    end_time = time.time()
                    results_dict["Execution times"]['GMM LPC'] = round(end_time - start_time,2)
                    print("")

                    results_dict["Results"]["Model"] = "GMM"
                    results_dict["Results"]["Features"] = "LPC"
                    results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                    if(PRINT_FILES):
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)

                if("SVM" in MODELS_LIST):
                    model_name = "SVM"
                    print("LPC with SVM")
                    start_time = time.time()
                    confusion_matrix = models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
                    end_time = time.time()
                    results_dict["Execution times"]['SVM LPC'] = round(end_time - start_time,2)
                    print("")

                    results_dict["Results"]["Model"] = "SVM"
                    results_dict["Results"]["Features"] = "LPC"
                    results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                    if(PRINT_FILES):
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)
# ########################################################################################################
# ## PLP
            if("PLP" in FEATURES_LIST):
                feature_name = "PLP"
                print("PLP")
                start_time = time.time()
                features, scaled_train, classes, scaler = feats.prepared_scaled_plp_feats(speaker_ids, pow_frames, results_dict["PLP"], plp_filters)
                end_time = time.time()
                results_dict["Execution times"]['PLP'] = round(end_time - start_time,2)

                if("VQ" in MODELS_LIST):
                    model_name = "VQ"
                    print("PLP with VQ")
                    start_time = time.time()
                    confusion_matrix = models.run_VQ_model(speaker_ids, features)
                    end_time = time.time()
                    results_dict["Execution times"]['VQ PLP'] = round(end_time - start_time,2)
                    print("")

                    results_dict["Results"]["Model"] = "VQ"
                    results_dict["Results"]["Features"] = "PLP"
                    results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                    if(PRINT_FILES):
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)

                if("GMM" in MODELS_LIST):
                    model_name = "GMM"
                    print("PLP with GMM")
                    start_time = time.time()
                    confusion_matrix = models.run_GMM_model(speaker_ids, features, scaler, results_dict["Model attr"]["GMM"])
                    end_time = time.time()
                    results_dict["Execution times"]['GMM PLP'] = round(end_time - start_time,2)
                    print("")

                    results_dict["Results"]["Model"] = "GMM"
                    results_dict["Results"]["Features"] = "PLP"
                    results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                    if(PRINT_FILES):
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)

                if("SVM" in MODELS_LIST):
                    model_name = "SVM"
                    print("PLP with SVM")
                    start_time = time.time()
                    confusion_matrix = models.run_SVM_model(speaker_ids, features, scaled_train, classes, scaler)
                    end_time = time.time()
                    results_dict["Execution times"]['SVM PLP'] = round(end_time - start_time,2)
                    print("")
                    results_dict["Results"]["Model"] = "SVM"
                    results_dict["Results"]["Features"] = "PLP"
                    results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                    if(PRINT_FILES):
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)


    

    #for k in  cfg["Execution times"].keys():
        #print(f"{k} :  {cfg["Execution times"][k]}")