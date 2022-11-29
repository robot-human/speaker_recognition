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

speakers_id = []
speaker_sessions = []
for speaker in database_dir:
    speakers_id.append(speaker)
    speaker_path = DATABASE_PATH + "/" + speaker
    sessions_dir = os.listdir(speaker_path)
    sessions = []
    for session in sessions_dir:
        sessions.append(session)
    speaker_sessions.append(sessions)

print(len(speakers_id))
print(len(speaker_sessions))