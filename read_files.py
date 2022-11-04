import os

database_path = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
database_dir = os.listdir(database_path)

speaker_paths = []
for speaker in database_dir:
    print(speaker)
    path = database_path + "/" + speaker
    files_dir = os.listdir(path)
    print(files_dir)
    print("*****************************")
    print(" ")