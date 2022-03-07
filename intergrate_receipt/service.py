# default lib
import os
import base64
import logging
import datetime
import pydantic
# need install lib
import uvicorn
import cv2
import json
import asyncio
import numpy as np
# custom modules
import rcode

# default lib
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
from typing import Optional, List
from pydantic import BaseModel
from pymongo import MongoClient

from urllib.parse import quote_plus

from configparser import ConfigParser
from PIL import Image
# custom modules
from predict import intergrate

now = datetime.datetime.now()

# LOAD CONFIG
config = ConfigParser()
config.read("config/service_intergrate.cfg")

SERVICE_IP = str(config.get('main', 'SERVICE_IP'))
SERVICE_PORT = int(config.get('main', 'SERVICE_PORT'))
LOG_PATH = str(config.get('main', 'LOG_PATH'))


app = FastAPI()

# CREATE LOGGER
os.makedirs("logfile_intergrate", exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_PATH, now.strftime("%d%m%y_%H%M%S")+".log"), filemode="w",
                level=logging.DEBUG,
                format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

class Images(BaseModel):
    data: Optional[List[str]] = pydantic.Field(default=None,
                    example=None, description='List of base64 encoded images')
class PredictData(BaseModel):
#    images: Images
    images: Optional[List[str]] = pydantic.Field(default=None,
                    example=None, description='List of base64 encoded images')

def get_client():
    """
    Setup a mongo client for the site
    :return:
    """
    host = os.getenv('MONGODB_HOST', '')
    username = os.getenv('MONGODB_USER', '')
    password = os.getenv('MONGODB_PASSWORD', '')
    port = int(os.getenv('MONGODB_PORT', 27017))
    endpoint = 'mongodb://{0}:{1}@{2}'.format(quote_plus(username), quote_plus(password), host)
    mongo_client = MongoClient(endpoint, port)
    return mongo_client

client = get_client()
db = client.new_db
posts = db.posts

print("SERVICE_IP", SERVICE_IP)
print("SERVICE_PORT", SERVICE_PORT)
print("LOG_PATH", LOG_PATH)
print("API READY")

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    logger.info("predict")
    return_result = {'code': '1001', 'status': rcode.code_1001}
    try:
        predicts = []
        try:
            contents = await image.read()
            nparr = np.fromstring(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '609', 'status': rcode.code_609}
            return; 
        
        predicts = dict(intergrate(img))
        predicts.update({"image": image.filename})
        
        posts.insert_one(dict(predicts))
        logger.info("Insert successfully into database!")
            
        return_result = {'code': '1000', 'status': rcode.code_1000, 'predicts': predicts,
                        'return': 'list of predicted forms'}
    except Exception as e:
            logger.error(e, exc_info=True)
            return_result = {'code': '1001', 'status': rcode.code_1001}
    finally:
        return return_result

if __name__ == '__main__':
    uvicorn.run(app, port=SERVICE_PORT, host=SERVICE_IP)