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


random.seed(cfg["General"]["N_SPEAKERS"])


ids, speaker_files = get_speaker_files(DATABASE_PATH)
n_speakers = cfg["General"]["N_SPEAKERS"]
speaker_ids = random.sample(ids, k=n_speakers)
start_time = time.time()
signal_dict = get_speaker_signals_dict(speaker_files, speaker_ids)
end_time = time.time()
cfg["Execution times"]['File reading'] = round(end_time - start_time,2)

preemph_list =cfg["Frames"]["PRE_EMPHASIS_COEF"]
frmes_size_list =cfg["Frames"]["FRAME_IN_SECS"]
overlap_list =cfg["Frames"]["OVERLAP_PCT"]
n_ceps_list = cfg["MFCC"]["N_CEPS"]
p_order_list = cfg["LPC"]["P"]

results_dict = cfg
for pre_emph in preemph_list:
    for frame_size in frmes_size_list:
        for frame_overlap in overlap_list:
            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
            
            results_dict["Frames"]["PRE_EMPHASIS_COEF"] = float(pre_emph)
            results_dict["Frames"]["FRAME_IN_SECS"] = float(frame_size)
            results_dict["Frames"]["OVERLAP_PCT"] = float(frame_overlap)
            results_dict["Frames"]["OVERLAP_IN_SECS"] = float(frame_size)*float(frame_overlap)
            n_frames = int((math.floor((results_dict["General"]["SIGNAL_DURATION_IN_SECONDS"]-results_dict["Frames"]["FRAME_IN_SECS"])/results_dict["Frames"]["OVERLAP_IN_SECS"])))
            results_dict["Model attr"]["GMM"]["N_MIXTURES"] = min(350,math.floor(0.70*n_frames ))
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
                        date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)
                        print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                        print("MFB VQ file saved")
    
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
                        date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)
                        print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                        print("MFB GMM file saved")

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
                        date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                        file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                        with open(file_path, 'w') as json_file:
                            json.dump(results_dict, json_file)
                        print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                        print("MFB SVM file saved")
# ########################################################################################################
## MFCC
            if("MFCC" in FEATURES_LIST):
                for n_ceps in n_ceps_list:
                    results_dict["MFCC"]["N_CEPS"] = int(n_ceps)
                    feature_name = "MFCC"
                    start_time = time.time()
                    print(f"MFCC con {n_ceps} coeficientes")
                    mfcc, deltas, ddeltas = feats.prepared_scaled_mfcc_feats(speaker_ids, pow_frames, int(n_ceps), results_dict["MFCC"])
                    end_time = time.time()
                    results_dict["Execution times"]['MFCC'] = round(end_time - start_time,2)

                    results_dict["MFCC"]["DELTAS"] = False
                    results_dict["MFCC"]["DOBLE_DELTAS"] = False
                    if("VQ" in MODELS_LIST):
                        model_name = "VQ"
                        start_time = time.time()
                        print("MFFC with VQ")
                        confusion_matrix = models.run_VQ_model(speaker_ids, mfcc[0])
                        end_time = time.time()
                        results_dict["Execution times"]['VQ MFCC'] = round(end_time - start_time,2)
                        print("")
            
                        results_dict["Results"]["Model"] = "VQ"
                        results_dict["Results"]["Features"] = "MFCC"
                        results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                        if(PRINT_FILES):
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"MFCC {n_ceps} VQ file saved")
    
                    if("GMM" in MODELS_LIST):
                        model_name = "GMM"
                        start_time = time.time()
                        print("MFFC with GMM")
                        confusion_matrix = models.run_GMM_model(speaker_ids, mfcc[0], mfcc[3], results_dict["Model attr"]["GMM"])
                        end_time = time.time()
                        results_dict["Execution times"]['GMM MFCC'] = round(end_time - start_time,2)
                        print("")

                        results_dict["Results"]["Model"] = "GMM"
                        results_dict["Results"]["Features"] = "MFCC"
                        results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                        if(PRINT_FILES):
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"MFCC {n_ceps} GMM file saved")

                    if("SVM" in MODELS_LIST):
                        model_name = "SVM"
                        start_time = time.time()
                        print("MFFC with SVM")
                        confusion_matrix = models.run_SVM_model(speaker_ids, mfcc[0], mfcc[1], mfcc[2], mfcc[3])
                        end_time = time.time()
                        results_dict["Execution times"]['SVM MFCC'] = round(end_time - start_time,2)
                        print("")

                        results_dict["Results"]["Model"] = "SVM"
                        results_dict["Results"]["Features"] = "MFCC"
                        results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                        if(PRINT_FILES):
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"MFCC {n_ceps} SVM file saved")

                    results_dict["MFCC"]["DELTAS"] = True
                    results_dict["MFCC"]["DOBLE_DELTAS"] = False
                    if("VQ" in MODELS_LIST):
                        model_name = "VQ"
                        start_time = time.time()
                        print("MFFC Deltas with VQ")
                        confusion_matrix = models.run_VQ_model(speaker_ids, deltas[0])
                        end_time = time.time()
                        results_dict["Execution times"]['VQ MFCC'] = round(end_time - start_time,2)
                        print("")
            
                        results_dict["Results"]["Model"] = "VQ"
                        results_dict["Results"]["Features"] = "MFCC"
                        results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                        if(PRINT_FILES):
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"MFCC {n_ceps} DELTA VQ file saved")
    
                    if("GMM" in MODELS_LIST):
                        model_name = "GMM"
                        start_time = time.time()
                        print("MFFC Deltas with GMM")
                        confusion_matrix = models.run_GMM_model(speaker_ids, deltas[0], deltas[3], results_dict["Model attr"]["GMM"])
                        end_time = time.time()
                        results_dict["Execution times"]['GMM MFCC'] = round(end_time - start_time,2)
                        print("")

                        results_dict["Results"]["Model"] = "GMM"
                        results_dict["Results"]["Features"] = "MFCC"
                        results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                        if(PRINT_FILES):
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"MFCC {n_ceps} DELTA GMM file saved")

                    if("SVM" in MODELS_LIST):
                        model_name = "SVM"
                        start_time = time.time()
                        print("MFFC Deltas with SVM")
                        confusion_matrix = models.run_SVM_model(speaker_ids, deltas[0], deltas[1], deltas[2], deltas[3])
                        end_time = time.time()
                        results_dict["Execution times"]['SVM MFCC'] = round(end_time - start_time,2)
                        print("")

                        results_dict["Results"]["Model"] = "SVM"
                        results_dict["Results"]["Features"] = "MFCC"
                        results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                        if(PRINT_FILES):
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"MFCC {n_ceps} DELTA SVM file saved")
                
                    results_dict["MFCC"]["DELTAS"] = False
                    results_dict["MFCC"]["DOBLE_DELTAS"] = True
                    if("VQ" in MODELS_LIST):
                        model_name = "VQ"
                        start_time = time.time()
                        print("MFFC Doble Deltas with VQ")
                        confusion_matrix = models.run_VQ_model(speaker_ids,ddeltas[0])
                        end_time = time.time()
                        results_dict["Execution times"]['VQ MFCC'] = round(end_time - start_time,2)
                        print("")
            
                        results_dict["Results"]["Model"] = "VQ"
                        results_dict["Results"]["Features"] = "MFCC"
                        results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                        if(PRINT_FILES):
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"MFCC {n_ceps} D DELTA VQ file saved")
    
                    if("GMM" in MODELS_LIST):
                        model_name = "GMM"
                        start_time = time.time()
                        print("MFFC Doble Deltas with GMM")
                        confusion_matrix = models.run_GMM_model(speaker_ids, ddeltas[0], ddeltas[3], results_dict["Model attr"]["GMM"])
                        end_time = time.time()
                        results_dict["Execution times"]['GMM MFCC'] = round(end_time - start_time,2)
                        print("")

                        results_dict["Results"]["Model"] = "GMM"
                        results_dict["Results"]["Features"] = "MFCC"
                        results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                        if(PRINT_FILES):
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"MFCC {n_ceps} D DELTA GMM file saved")

                    if("SVM" in MODELS_LIST):
                        model_name = "SVM"
                        start_time = time.time()
                        print("MFFC Doble Deltas with SVM")
                        confusion_matrix = models.run_SVM_model(speaker_ids, ddeltas[0], ddeltas[1], ddeltas[2], ddeltas[3])
                        end_time = time.time()
                        results_dict["Execution times"]['SVM MFCC'] = round(end_time - start_time,2)
                        print("")

                        results_dict["Results"]["Model"] = "SVM"
                        results_dict["Results"]["Features"] = "MFCC"
                        results_dict["Results"]["Confusion matrix"] = confusion_matrix
            
                        if(PRINT_FILES):
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"MFCC {n_ceps} D DELTA SVM file saved")
# ########################################################################################################
# ## LPC
            if("LPC" in FEATURES_LIST):
                for p in p_order_list:
                    results_dict["LPC"]["P"] = int(p)
                    feature_name = "LPC"
                    print(f"LPC de orden {p}")
                    start_time = time.time()
                    features, scaled_train, classes, scaler = feats.prepared_scaled_lpc_feats(speaker_ids, window_frames, int(p))
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
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"LPC {p} VQ file saved")

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
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"LPC {p} GMM file saved")

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
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"LPC {p} SVM file saved")
# ########################################################################################################
# ## PLP
            if("PLP" in FEATURES_LIST):
                for p in p_order_list:
                    results_dict["PLP"]["P"] = int(p)
                    feature_name = "PLP"
                    print(f"PLP de orden {p}")
                    start_time = time.time()
                    features, scaled_train, classes, scaler = feats.prepared_scaled_plp_feats(speaker_ids, pow_frames, int(p), plp_filters)
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
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"PLP {p} VQ file saved")

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
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"PLP {p} GMM file saved")

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
                            date = datetime.datetime.now().strftime("%m/%d/%H/%M/%S").replace("/","_")
                            file_path = LOG_FILE_PATH + f"{feature_name}_{model_name}_speakers_{n_speakers}_session_{date}.json"
                            with open(file_path, 'w') as json_file:
                                json.dump(results_dict, json_file)
                            print(f"Pre-emph: {pre_emph}, Frame size: {frame_size}, Frame overlap {frame_overlap}")
                            print(f"PLP {p} SVM file saved")
