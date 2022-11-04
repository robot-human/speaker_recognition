import os

database_path = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
database_dir = os.listdir(database_path)

speaker_files = {}
for speaker in database_dir:
    speaker_path = database_path + "/" + speaker
    sessions_dir = os.listdir(speaker_path)
    for session in sessions_dir:
        audio_files_path = speaker_path + "/" + session
        audio_files_dir = os.listdir(audio_files_path)
        for audio_file in audio_files_dir:
            if(audio_file.find(".flac") > 0):
                print(audio_file)


    #speaker_sessions[speaker] = files_dir
    #print(files_dir)
    #print("*****************************")
    #print(" ")

#new_path = database_path+"/911/"+speaker_sessions['911'][0]
#new_path_dir = os.listdir(new_path)
#for files in new_path_dir:
#    print(files)