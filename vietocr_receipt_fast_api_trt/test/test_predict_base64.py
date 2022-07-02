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
url = 'http://0.0.0.0:2341/predict'
#url = 'https://aiclub.uit.edu.vn/gpu/service/craft_ocr_fastapi/predict'
####################################
file_path = "test.jpg"
####################################
#image = open(file_path, 'rb')
#image_read = image.read()
#encoded = base64.encodestring(image_read)
#image_encoded = encoded.decode('utf-8')
####
img = cv2.imread(file_path)
is_success, buffer = cv2.imencode('.png', img)
f = io.BytesIO(buffer)
image_encoded = base64.encodebytes(f.getvalue()).decode('utf-8')
####################################
data ={"images": [image_encoded]}
headers = {'Content-type': 'application/json'}
data_json = json.dumps(data)
response = requests.post(url, data = data_json, headers=headers)
response = response.json()
print(response)
print('time', time.time()-start_time)
