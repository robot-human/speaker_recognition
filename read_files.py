import os

database_path = "~/Data/databases/LibriSpeech/train-clean-100/train-clean-100"
path = "/Users/humanrobot/Documents/data/simpsons"

dir = os.listdir(database_path)

for file in dir:
    print(file)