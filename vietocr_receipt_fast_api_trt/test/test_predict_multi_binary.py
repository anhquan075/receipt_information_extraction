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
url = 'http://0.0.0.0:2363/predict_multi_binary'
url = 'http://192.168.20.150/service/vietocr_trt/predict_multi_binary'
####################################
file_path = "test.png"
####################################
f = ('binary_files', open(file_path, 'rb'))
f1 = ('binary_files', open(file_path, 'rb'))
####################################
response = requests.post(url, files = [f, f1])
response = response.json()
print(response)
print('time', time.time()-start_time)

