import os

database_path = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
database_dir = os.listdir(database_path)

speaker_sessions = {}
for speaker in database_dir:
    path = database_path + "/" + speaker
    files_dir = os.listdir(path)
    print(files_dir)
    #speaker_sessions[speaker] = files_dir
    #print(files_dir)
    #print("*****************************")
    #print(" ")

#new_path = database_path+"/911/"+speaker_sessions['911'][0]
#new_path_dir = os.listdir(new_path)
#for files in new_path_dir:
#    print(files)