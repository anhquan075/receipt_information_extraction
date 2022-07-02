####default lib
import os
import base64
import logging
import time
import timeit
import datetime
import pydantic
from more_itertools import chunked
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
from PIL import Image
####custom modules
from src.vietocr_trt import TrtOCR
from vietocr.tool.config import Cfg
####
now = datetime.datetime.now()
#######################################
#####LOAD CONFIG####
config = ConfigParser()
config.read("config/service.cfg")

SERVICE_IP = str(config.get('main', 'SERVICE_IP'))
SERVICE_PORT = int(config.get('main', 'SERVICE_PORT'))
LOG_PATH = str(config.get('main', 'LOG_PATH'))
WORKER_NUM = str(config.get('main', 'WORKER_NUM'))
ENCODER_PATH = str(config.get('main', 'ENCODER_PATH'))
DECODER_PATH = str(config.get('main', 'DECODER_PATH'))
BATCH_SIZE = str(config.get('main', 'BATCH_SIZE'))
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
config = Cfg.load_config_from_name('vgg_transformer')
dataset_params = {
    'name':'hw',
    'data_root':'./my_data/',
    'train_annotation':'train_line_annotation.txt',
    'valid_annotation':'test_line_annotation.txt'
}

params = {
         'print_every':200,
         'valid_every':15*200,
          'iters':20000,
          'checkpoint':'./checkpoint/transformerocr_checkpoint.pth',    
          'export':'./weights/transformerocr.pth',
          'metrics': 10000
         }

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'

ocr_model = TrtOCR(ENCODER_PATH, DECODER_PATH, config)
#######################################
print("SERVICE_IP", SERVICE_IP)
print("SERVICE_PORT", SERVICE_PORT)
print("LOG_PATH", LOG_PATH)
print("WORKER_NUM", WORKER_NUM)
print("API READY")
#######################################


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


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
            
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'WORKER_NUM': WORKER_NUM, 'return': 'base64 encoded file'}
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
        ####resize image
        h, w,_ = process_image.shape
        new_h = 32
        new_w = int((new_h * w) / h)
        new_w = max(min(new_w, 700), 128)
        process_image = cv2.resize(process_image, (new_w, new_h))
        ####
        process_image = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
        process_image = Image.fromarray(process_image)
        ####
        prd_label, score = ocr_model.predict(process_image)
#        print(prd_label, score)
        predicts.append((prd_label, score))
        
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'WORKER_NUM': WORKER_NUM, 'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

@app.post('/predict_multi_binary')
async def predict_binary(binary_files: Optional[List[UploadFile]] = File(None)):
    logger.info("predict_multi_binary")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    try:
        start_time = timeit.default_timer()
        predicts = []
        try:
            bytes_file_list = []
            for binary_file in binary_files:
                bytes_file_list.append((binary_file.filename, await binary_file.read()))
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        
        process_image_list = []
        for img_name, bytes_file in bytes_file_list:
            nparr = np.fromstring(bytes_file, np.uint8)
            process_image = cv2.imdecode(nparr, flags=1)
            # resize image
            h, w,_ = process_image.shape
            new_h = 32
            new_w = int((new_h * w) / h)
            new_w = max(min(new_w, 700), 128)
            process_image = cv2.resize(process_image, (new_w, new_h))
            process_image = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
            process_image = Image.fromarray(process_image)
            process_image_list.append((img_name, process_image))
            
        batch_image_lst = list(divide_chunks(process_image_list, int(BATCH_SIZE)))
        for batch in batch_image_lst:
            predict_img = ocr_model.predict_batch([bt[-1] for bt in batch])
            batch_imgname = [bt[0] for bt in batch]
            predicts.extend(zip(batch_imgname, predict_img))
        
        print(predicts)
        
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'WORKER_NUM': WORKER_NUM, 'return': 'base64 encoded file'}
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
        
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'process_time': timeit.default_timer()-start_time,
                        'WORKER_NUM': WORKER_NUM, 'return': 'base64 encoded file'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

if __name__ == '__main__':
    uvicorn.run(app, port=SERVICE_PORT, host=SERVICE_IP)

