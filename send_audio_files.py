import os
import math
import random
import functions
import time
from sys import getsizeof
import paho.mqtt.client as paho
from paho import mqtt
import paho.mqtt.publish as publish
from time import sleep
from env_variables import DATABASE_PATH

SERVER = 0
topic = "speaker_recognition_server"
clientID = "clientId-xMODDl314VwR-speaker-recognition"
file_path = f'./config.ini'
QOS = 2
KEEPALIVE=60

if(SERVER == 0):
    host ="broker.mqttdashboard.com"
    port=1883

database_dir = os.listdir(DATABASE_PATH)

for speaker in database_dir:
    speaker_path = database_path + "/" + speaker
    sessions_dir = os.listdir(speaker_path)
    for session in sessions_dir:
        print(session)