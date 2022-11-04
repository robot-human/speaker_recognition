import os

database_path = "/home/ar/Data/ivette_thesis/data_bases/Librispeech/train-clean-100/train-clean-100/"
path = "/Users/humanrobot/Documents/data/simpsons"

dir = os.listdir(database_path)

for file in dir:
    print(file)