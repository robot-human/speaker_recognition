import os

database_path = "~/"
path = "/Users/humanrobot/Documents/data/simpsons"

dir = os.listdir(database_path)

for file in dir:
    print(file)