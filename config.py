from os.path import isfile
import os
from configparser import ConfigParser

cfg = None
                         
if not isfile(os.path.join(os.getcwd(), 'config.ini')):
    print("config file doesn't exist")
else:
    cfgParser = ConfigParser()
    cfgParser.read(os.path.join(os.getcwd(), 'config.ini'))

    print(cfgParser)