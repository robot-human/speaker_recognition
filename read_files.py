import os

database_path = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
database_dir = os.listdir(database_path)

speaker_sessions = {}
for speaker in database_dir:
    #print(speaker)
    path = database_path + "/" + speaker
    files_dir = os.listdir(path)
    speaker_sessions[speaker] = files_dir
    #print(files_dir)
    #print("*****************************")
    #print(" ")

print(database_path+"/911/"+speaker_sessions['911'][0])