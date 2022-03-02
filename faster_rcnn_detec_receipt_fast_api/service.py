####default lib
import os
import base64
import logging
import time
import timeit
import datetime
import pydantic
####need install lib
import uvicorn
import cv2
import traceback
import asyncio
import numpy as np
####custom modules
import rcode
####default lib
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
from typing import Optional, List
from pydantic import BaseModel
from configparser import ConfigParser
####


import io
import requests
import random
import json

from src.utils.utils import load_class_names
from src.utils.parser import get_config
from src.utils.draw import draw_bbox

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg as config_detectron

from src.predict import predict_model

now = datetime.datetime.now()
#######################################
#####LOAD CONFIG####
config = ConfigParser()
config.read("config/service.cfg")

SERVICE_IP = str(config.get('main', 'SERVICE_IP'))
SERVICE_PORT = int(config.get('main', 'SERVICE_PORT'))
LOG_PATH = str(config.get('main', 'LOG_PATH'))

CLASSES = str(config.get('model', 'CLASSES'))
DETECT_WEIGHT = str(config.get('model', 'DETECT_WEIGHT'))
DETECT_CONFIG = str(config.get('model', 'DETECT_CONFIG'))
THRESHOLD = float(config.get('model', 'THRESHOLD'))
NUMBER_CLASS = int(config.get('model', 'NUMBER_CLASS'))
DEVICE = str(config.get('model', 'DEVICE'))



#######################################
app = FastAPI()
#######################################
#####CREATE LOGGER#####
logging.basicConfig(filename=os.path.join(LOG_PATH, now.strftime("%d%m%y_%H%M%S")+".log"), filemode="w",
                level=logging.DEBUG,
                format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)
#######################################
class Images(BaseModel):
    data: Optional[List[str]] = pydantic.Field(default=None,
                    example=None, description='List of base64 encoded images')
class PredictData(BaseModel):
#    images: Images
    images: Optional[List[str]] = pydantic.Field(default=None,
                    example=None, description='List of base64 encoded images')
#######################################
####LOAD MODEL HERE
# set up detectron
detectron = config_detectron()
detectron.MODEL.DEVICE = DEVICE
detectron.merge_from_file(DETECT_CONFIG)
detectron.MODEL.WEIGHTS = DETECT_WEIGHT

detectron.MODEL.RETINANET.SCORE_THRESH_TEST = THRESHOLD
#detectron.MODEL.ROI_HEADS.NUM_CLASSES = NUMBER_CLASS
detectron.MODEL.RETINANET.NUM_CLASSES = NUMBER_CLASS


PREDICTOR = DefaultPredictor(detectron)

# create labels
CLASSES = load_class_names(CLASSES)

def load_predict(image):
    height, width, channels = image.shape
    center_image = (width//2, height//2)
    print("shape image: ", (width, height))
    list_boxes, list_scores, list_classes = predict_model(
        image, PREDICTOR, CLASSES)
    print('list_boxes', list_boxes)
    print('list_classes', list_classes)

    # draw
    # image = draw_bbox(image, list_boxes, list_scores, list_classes)
    # cv2.imwrite("image.jpg", image)

    i = 0
    len_boxes = len(list_boxes)
    # point_tl = None
    # point_tr = None
    # point_bl = None
    # point_br = None
    receipt = None
    while i < len_boxes:
        bbox = list_boxes[i]
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        w = x2 - x1
        h = y2 - y1
        center_x = x1 + w//2
        center_y = y1 + h//2
        center = (center_x, center_y)
        # print("max: ", (x1, y1))
        # print("min: ", (x2, y2))
        if list_classes[i] == 'receipt':
            receipt = bbox

        i += 1

    result = {'receipt': receipt}

    return result
#######################################
print("SERVICE_IP", SERVICE_IP)
print("SERVICE_PORT", SERVICE_PORT)
print("LOG_PATH", LOG_PATH)
print("API READY")
#######################################
@app.post('/predict')
async def predict(data: PredictData):
    ###################
    #####
    logger.info("predict")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            images = jsonable_encoder(data.images)
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        for image in images:
            image_decoded = base64.b64decode(image)
            jpg_as_np = np.frombuffer(image_decoded, dtype=np.uint8)
            process_image = cv2.imdecode(jpg_as_np, flags=1)
            predicts = load_predict(process_image)
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

@app.post('/predict_binary')
async def predict_binary(binary_file: UploadFile = File(...)):
    ###################
    #####
    logger.info("predict_binary")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file = await binary_file.read()
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        nparr = np.fromstring(bytes_file, np.uint8)
        process_image = cv2.imdecode(nparr, flags=1)
        predicts = load_predict(process_image)
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result
        
@app.post('/predict_multi_binary')
async def predict_binary(binary_files: Optional[List[UploadFile]] = File(None)):
    ###################
    #####
    logger.info("predict_multi_binary")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file_list = []
            for binary_file in binary_files:
                bytes_file_list.append(await binary_file.read())
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        process_image_list = []
        for bytes_file in bytes_file_list:
            nparr = np.fromstring(bytes_file, np.uint8)
            process_image = cv2.imdecode(nparr, flags=1)
            predicts = load_predict(process_image)
            process_image_list.append(process_image)
        
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

@app.post('/predict_multipart')
async def predict_multipart(argument: Optional[float] = Form(...),
                binary_file: UploadFile = File(...)):
    ###################
    #####
    logger.info("predict_multipart")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    ###################
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file = await binary_file.read()
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        ###########################
        nparr = np.fromstring(bytes_file, np.uint8)
        process_image = cv2.imdecode(nparr, flags=1)
        predicts = load_predict(process_image)
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

if __name__ == '__main__':
    uvicorn.run(app, port=SERVICE_PORT, host=SERVICE_IP)

