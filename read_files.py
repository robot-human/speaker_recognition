import os

database_path = "/media/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
database_dir = os.listdir(database_path)

speaker_paths = []
for file in database_dir:
    path = database_path + "/" + file
    speaker_paths.append(path)

for paths in speaker_paths:
    print(paths)
