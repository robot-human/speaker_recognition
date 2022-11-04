import os

database_path = "/media/Data/data_bases/Librispeech/train-clean-100/train-clean-100/"
path = "/Users/humanrobot/Documents/data/simpsons"

dir = os.listdir(database_path)

for file in dir:
    print(file)