import requests
import json
import numpy as np
import cv2

TASK1_URL = 'http://service.aiclub.cs.uit.edu.vn/receipt/receipt_detect/predict'

img_path = '/home/huy/TEST_SUBMIT/end_to_end_submit/raw_data_img/test/mcocr_val_145114ixmyt.jpg'

img = cv2.imread(img_path)

detect_task1 = requests.post(TASK1_URL, files={"file": (
    "filename", open(img_path, "rb"), "image/jpeg")}).json()

print(detect_task1)