import base64
import urllib.parse
import requests
import json
import timeit
import sys
import io
import cv2
import numpy as np
import time
start_time = time.time()
url = 'http://0.0.0.0:80/predict_binary'
#url = 'https://aiclub.uit.edu.vn/gpu/service/craft_ocr_fastapi/predict_binary'
####################################
file_path = "test.jpg"
####################################
f = {'binary_file': open(file_path, 'rb')}
####################################
response = requests.post(url, files = f)
response = response.json()
print(response)
print('time', time.time()-start_time)
