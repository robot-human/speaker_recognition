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

audio_files_list = []
for idx in [7,11,25,40]:
    audio_path = DATABASE_PATH + "/" + speakers_id[idx] + "/" + speaker_sessions[idx][0]
    audio_dir = os.listdir(audio_path)
    for enum, audio in enumerate(audio_dir):
        if(audio.find(".flac") > 0):
            if(enum < 1):
                audio_files_list.append(audio_path+"/"+audio)

for audio in audio_files_list:
    print(audio)