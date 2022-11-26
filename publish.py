import os
import time
from sys import getsizeof
import paho.mqtt.client as paho
from paho import mqtt
import paho.mqtt.publish as publish
from time import sleep

SERVER = 0
topic = "speaker_recognition_server"
clientID = "clientId-xMODDl314VwR-speaker-recognition"
file_path = f'./config.ini'
QOS = 2
KEEPALIVE=60

if(SERVER == 0):
    host ="broker.mqttdashboard.com"
    port=1883


output_files_path = f'./log_files/'
output_files = os.listdir(output_files_path)

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
    for file_path in output_files:
        n_files += 1
    
    print(f"Number of files {n_files}")
    #client.publish(topic, payload=f"Hello GUI am sending {n_files} files", qos=QOS)
    publish.single(topic, payload=f"Hello GUI am sending {n_files} files", qos=QOS, retain=False, hostname=host,port=port, client_id=clientID, keepalive=KEEPALIVE, will=None, auth=None, tls=None,protocol=paho.MQTTv5, transport="tcp")
    for name in output_files:
        print(name)
        file_name_path = output_files_path+name
        #client.publish(topic, payload=f"{name}", qos=QOS)
        publish.single(topic, payload=f"{name}", qos=QOS, retain=False, hostname=host,port=port, client_id=clientID, keepalive=KEEPALIVE, will=None, auth=None, tls=None,protocol=paho.MQTTv5, transport="tcp")
        sleep(0.2)
        f = open(file_name_path, "r")
        content = f.read()
        print(getsizeof(content)/1000, " kbts")
        #client.publish(topic, payload=content, qos=QOS)
        publish.single(topic, payload=content, qos=QOS, retain=False, hostname=host,port=port, client_id=clientID, keepalive=KEEPALIVE, will=None, auth=None, tls=None,protocol=paho.MQTTv5, transport="tcp")
        sleep(0.2)
        f.close()
        print(f"{name} closed")     
    
    client.on_disconnect = on_disconnect
    client.disconnect()
    