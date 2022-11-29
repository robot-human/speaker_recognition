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
import librosa
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

audio_files_name = []
audio_files_path = []
for idx in [7,11,25,40]:
    audio_path = DATABASE_PATH + "/" + speakers_id[idx] + "/" + speaker_sessions[idx][0]
    audio_dir = os.listdir(audio_path)
    for enum, audio in enumerate(audio_dir):
        if(audio.find(".flac") > 0):
            if(enum < 1):
                audio_files_name.append(audio)
                audio_files_path.append(audio_path+"/")

for audio in audio_files_name:
    print(audio)


def on_connect(client, userdata, flags, rc, properties=None):
    print("on connect %s." % rc)
def on_disconnect(client, userdata, flags, rc):
    print("client disconnected")
def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))
def on_publish(client, userdata, mid, properties=None):
    print("mid: " + str(mid))


if __name__ == '__main__':
    print("start sending files")
    client = paho.Client(client_id=clientID, userdata=None, protocol=paho.MQTTv5)
    client.on_connect = on_connect
    client.connect(host, port, keepalive=KEEPALIVE)
    client.on_subscribe = on_subscribe
    client.subscribe(topic, qos=QOS)
    client.on_publish = on_publish
    
    n_files = 0
    for file_path in audio_files_name:
        n_files += 1
    
    print(f"Number of files {n_files}")
    #client.publish(topic, payload=f"Hello GUI am sending {n_files} files", qos=QOS)
    publish.single(topic, payload=f"Hello GUI am sending {n_files} files", qos=QOS, retain=False, hostname=host,port=port, client_id=clientID, keepalive=KEEPALIVE, will=None, auth=None, tls=None,protocol=paho.MQTTv5, transport="tcp")
    for enum, name in enumerate(audio_files_name):
        print(name)
        file_name_path = audio_files_path[enum] + name
        #client.publish(topic, payload=f"{name}", qos=QOS)
        publish.single(topic, payload=f"{name}", qos=QOS, retain=False, hostname=host,port=port, client_id=clientID, keepalive=KEEPALIVE, will=None, auth=None, tls=None,protocol=paho.MQTTv5, transport="tcp")
        sleep(0.2)
        content, _ = librosa.load(file_name_path, mono=True)
        print(getsizeof(content)/1000, " kbts")
        #client.publish(topic, payload=content, qos=QOS)
        publish.single(topic, payload=content, qos=QOS, retain=False, hostname=host,port=port, client_id=clientID, keepalive=KEEPALIVE, will=None, auth=None, tls=None,protocol=paho.MQTTv5, transport="tcp")
        sleep(0.2)     
    
    client.on_disconnect = on_disconnect
    client.disconnect()
    