import os

print(os.getcwd())
database_path = "/media/Data/databases/LibriSpeech"
#path = "/Users/humanrobot/Documents/data/simpsons"

dir = os.listdir(database_path)

for file in dir:
    print(file)