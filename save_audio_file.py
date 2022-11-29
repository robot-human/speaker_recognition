import time
import paho.mqtt.client as paho
from paho import mqtt
import soundfile as sf
import numpy as np

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
    
    message = msg.payload.decode('utf-8')
    print(count,num)
    #print(message)
    if(count==0):
        print("Got 1.0")
        num=int(message.split(" ")[4])
        print("Number of files",num)
        count+=1
        print("success")
    elif((count > 0) and (num > 0) and (count < num*2)):
        if(count%2==1):
            print("Got 2.1")
            print(message)
            file_name = file_path+message
            count+=1
            print("success")
        elif(count%2==0):
            print("Got 2.2")
            content = np.frombuffer(message)
            sf.write('stereo_file.flac', content, 10000, format='flac', subtype='PCM_24')
            count+=1
            print("success")
    elif(count >= num*2):
        print("Got 3.0")
        content = np.frombuffer(message)
        sf.write('stereo_file.flac', content, 10000, format='flac', subtype='PCM_24')
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