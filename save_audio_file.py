import time
import paho.mqtt.client as paho
from paho import mqtt
import soundfile as sf
import numpy as np
import librosa

SERVER = 0
topic = "speaker_recognition_server"
clientID = "clientId-vqrRlEJH0314J-speaker-recognition"

if(SERVER == 0):
    host ="broker.mqttdashboard.com"
    port=1883



file_path = f'./log_files/'
file_name = ""
out_file = None
num = 0
count = 0

def on_connect(client, userdata, flags, rc, properties=None):
    print("on connect %s." % rc)
def on_disconnect(client, userdata, flags, rc):
    print("client disconnected")
def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print("Subscribed: " + str(mid) + " " + str(granted_qos))
def on_message(client, userdata, msg):
    global file_name
    global out_file
    global count
    global num
    
    print(count,num)
    #print(message)
    if(count==0):
        print("Got 1.0")
        message = msg.payload.decode('utf-8')
        num=int(message.split(" ")[4])
        print("Number of files",num)
        count+=1
        print("success")
    elif((count > 0) and (num > 0) and (count < num*2)):
        if(count%2==1):
            print("Got 2.1")
            message = msg.payload.decode('utf-8')
            print(message)
            file_name = file_path+message
            out_file = open(file_name, 'w')
            count+=1
            print("success")
        elif(count%2==0):
            print("Got 2.2")
            message = msg.payload.decode('utf-8')
            #print(message)
            message = np.frombuffer(message, dtype=np.array)
            print(message)
            #sf.write(file_name, message, 10000, subtype='PCM_24')
            #librosa.output.write_wav(file_name, msg, 10000)
            count+=1
            print("success")
    elif(count >= num*2):
        print("Got 3.0")
        message = msg.payload.decode('utf-8')
        message = np.frombuffer(message, dtype=np.array)
        print(message)
        #sf.write(file_name, message, 10000, subtype='PCM_24')
        #librosa.output.write_wav(file_name, msg, 10000)
        count=0
        client.disconnect()
        print("success")
    return None

def main():
    client = paho.Client(client_id=clientID, userdata=None, protocol=paho.MQTTv5)
    client.connect(host, port)
    client.on_subscribe = on_subscribe
    client.subscribe(topic, qos=2)
    client.on_message = on_message
    client.loop_forever()

main()