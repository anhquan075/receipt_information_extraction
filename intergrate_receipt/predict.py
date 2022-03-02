import base64
from genericpath import exists
import requests
import json
import cv2
import numpy as np
import os
from datetime import datetime
import re

from configparser import SafeConfigParser
import pytesseract
from PIL import Image

from post_processing.submit import extract_info
from pre_processing.rotated_receipt_khanh import load_bbox, crop_text
import Levenshtein

config = SafeConfigParser()
config.read("config/service.cfg")
LOG_PATH = str(config.get('main', 'LOG_PATH'))
UPLOAD_FOLDER = str(config.get('main', 'UPLOAD_FOLDER'))
RESULT_FOLDER = str(config.get('main', 'RESULT_FOLDER'))
RECOG_FOLDER = str(config.get('main', 'RECOG_FOLDER'))
RESULT_TXT_FOLDER = str(config.get('main', 'RESULT_TXT_FOLDER'))
TASK1_URL = str(config.get('main', 'TASK1_URL'))
TASK2_URL = str(config.get('main', 'TASK2_URL'))
DETECT_RECEIPT_URL = str(config.get('main', 'DETECT_RECEIPT_URL'))
# ROTATED_RECEIPT_URL = str(config.get('main', 'ROTATED_RECEIPT_URL'))
DETECT_RECEIPT_FASTER_RCNN = config.get('main', 'DETECT_RECEIPT_FASTER_RCNN')
KIE_URL = str(config.get('main', 'KIE_URL'))

ROTATED_RECEIPT = config.get('main', 'ROTATED_RECEIPT')
OUTPUT_PATH_PRE_DETEC_RECEIPT = str(
    config.get('main', 'OUTPUT_PATH_PRE_DETEC_RECEIPT'))
OUTPUT_PATH_PRE_ROTATED_RECEIPT = str(
    config.get('main', 'OUTPUT_PATH_PRE_ROTATED_RECEIPT'))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(RECOG_FOLDER, exist_ok=True)
os.makedirs(RESULT_TXT_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_PATH_PRE_DETEC_RECEIPT, exist_ok=True)
os.makedirs(OUTPUT_PATH_PRE_ROTATED_RECEIPT, exist_ok=True)

def merge_values(values):
    try:
        if len(values) > 1:
            return ' '.join(sorted(values, reverse=True))
        else:
            return values[0]
    except:
        return ''


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

def lev(data_task1, data_task2, output_field):
    print('--------------Levenshtein----------------------')
    bboxs = data_task1['predicts']
    words = data_task2['result']
    seller, address, time_, total = output_field
    thre = 0.5
    # seller_bb, address_bb, time_bb, total_bb = []
    bb_field = []
    list_word_tmp = []
    for box, word in zip(bboxs, words):
        # if word in list_word_tmp:
        #     continue
        box = box['bbox']
        word = word['words']
        for field in output_field:
            # print(field)
            if len(word) <= 2:
                break
            if '|||' in field:
                sub_field = field.split('|||')
                # print(sub_field)

                for sub in sub_field:
                    # print('sub',sub)
                    if '.000' in sub or ',000' in sub:
                        print(sub, word)
                        thre = 0.80
                    if Levenshtein.ratio(sub, word) > thre:
                        if sub not in list_word_tmp:
                            bb_field.append(box)
                            list_word_tmp.append(word)
                    if word == '30.03.2019':
                        print(word, sub)
                        print(Levenshtein.ratio(sub, word))
            elif Levenshtein.ratio(field, word) > thre:
                if field not in list_word_tmp:
                    bb_field.append(box)
                    list_word_tmp.append(word)

    print(list_word_tmp)
    return bb_field


def output_csv(data_task1, data_task2, output_path):
    bboxs = data_task1['results'][0]
    words = data_task2['predicts']
    # print(words)
    result_txt_list = ''
    for idx, box in enumerate(bboxs):
        for word in words:
            if idx == int(word[0].split(".")[0]):
                result_txt_line = str(idx) + ','
                box = sum(box['text_region'], [])
                word = word[-1][0]
                # print(box)
                # print(word)
                for b in box:
                    result_txt_line += str(b) + ','
                    # print(result_txt_line)
                result_txt_line += str(word)
                result_txt_list += result_txt_line + '\n'
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(result_txt_list)
        print('output_path: ', output_path)


def intergrate(image):
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    img_name = str(date_time) + '.jpg'
    image_path = os.path.join(UPLOAD_FOLDER, date_time + '.jpg')
    print("image_path: ", image_path)
    cv2.imwrite(image_path, image)

    img = cv2.imread(image_path)

    start_time = datetime.now()
    if DETECT_RECEIPT_FASTER_RCNN:
        print('--------------CROP_IMG_DETECT_RECEIPT----------------')
        print(DETECT_RECEIPT_URL)
        # detect_receipt = requests.post(DETECT_RECEIPT_URL, files={"file": (
        #     "filename", open(image_path, "rb"), "image/jpeg")}).json()
        f = {'binary_file': open(image_path, 'rb')}
        detect_receipt = requests.post(DETECT_RECEIPT_URL, files=f).json()
        receipt_box = detect_receipt['predicts']['receipt']
        print('receipt_box: {}'.format(receipt_box))
        img_out_path = os.path.join(OUTPUT_PATH_PRE_DETEC_RECEIPT, img_name)
        if receipt_box is not None:
            crop = img[receipt_box[1]:receipt_box[3], receipt_box[0]:receipt_box[2]]
            
            # Find angle to rotate         
            rot_data = pytesseract.image_to_osd(crop)
            angle = int(re.search('(?<=Rotate: )\d+', rot_data).group(0))
            
            crop = Image.fromarray(crop)
            crop.rotate(angle, expand=True).save(img_out_path)
        else:
            cv2.imwrite(img_out_path, img)
        image_path = img_out_path
        img_new = cv2.imread(image_path)
        
        # f = {'binary_file': open(img_out_path, 'rb')}
        # rotated_func = requests.post(ROTATED_RECEIPT_URL, files=f).json()
        # print('rotated_func',
        # 	  rotated_func['predicts'], rotated_func['score'])
        # if rotated_func['predicts'] != 'None' and float(rotated_func['score']) > 0.6:
        # 	dic_rotate_fuc = {'ROTATE_90_CLOCKWISE': cv2.ROTATE_90_CLOCKWISE,
        # 					  'ROTATE_90_COUNTERCLOCKWISE': cv2.ROTATE_90_COUNTERCLOCKWISE, 'ROTATE_180': cv2.ROTATE_180}
        # 	crop = cv2.rotate(
        # 		crop, dic_rotate_fuc[rotated_func['predicts']])
        print("Time process:", datetime.now() - start_time)

    print('--------------DETECT_TEXT_PANNET----------------')
    headers = {"Content-type": "application/json"}
    img = open(image_path, 'rb').read()
    data = {'images': [cv2_to_base64(img)]}
    detect_task1 = requests.post(TASK1_URL, headers=headers, data=json.dumps(data)).json()
    print('detect_task1:', detect_task1)


    # Create folder to save cropped images
    cropped_folder = image_path.split(".")[0] + "_cropped"
    os.makedirs(cropped_folder, exist_ok=True)
    # Crop image into per batch
    bounding_box_detected = load_bbox(detect_task1)
    for i, bb in enumerate(bounding_box_detected):
        rect = cv2.boundingRect(np.array(bb))
        x, y, w, h = rect
        cropped = img_new[y:y+h, x:x+w].copy()
        cv2.imwrite(os.path.join(cropped_folder, str(i) + '.png'), cropped)

    print('Time process:', datetime.now() - start_time)

    print('--------------REG_TEXT_VIETOCR----------------')
    files = [('binary_files', (str(img), open(os.path.join(
        cropped_folder, img), 'rb'), 'image/png')) for img in os.listdir(cropped_folder)]

    detect_task2 = requests.post(TASK2_URL, files=files).json()
    print(detect_task2)
    # print('detect_task2:',detect_task2)
    # if detect_task2['code'] == 1000:
    #     detect_task2 = detect_task2.text
    # else:
    #     detect_task2 = requests.post(TASK2_URL, files=files)
    #     print('detect_task2:',detect_task2)

    # image_path_recog = os.path.join(RECOG_FOLDER, date_time+'.jpg')
    # image_path_recog = visualize(rotated_img, image_path , detect_task1, detect_task2)
    # cv2.imwrite(image_path_recog, image)
    output_tsv_path = os.path.join(RESULT_TXT_FOLDER, date_time+'.tsv')
    # print(image_path, output_tsv_path)
    output_csv(detect_task1, detect_task2, output_tsv_path)
    print('Time process:', datetime.now() - start_time)

    print('--------------KEY_INFORMATION_EXTRACTION_PICK----------------')
    data_json = {
        'output_tsv_path': output_tsv_path,
        'image_path': image_path
    }
    
    files = [
        ("image", ("filename", open(image_path, "rb"), "image/jpeg")),
        ('tsv', ('tsv', open(output_tsv_path, 'rb'), 'text/csv')),
        ('data', ('data', json.dumps(data_json), 'application/json'))
    ]

    kie_api = requests.post(KIE_URL, files=files, json=data_json)
    print('kie_api:', kie_api)
    if kie_api.status_code == 200:
        kie_api = kie_api.json()
    else:
        kie_api = requests.post(KIE_URL, files=files)
        print('kie_api:', kie_api)

    print('Time process:', datetime.now() - start_time)

    #seller, address, time, total = extract_info(image_path, output_txt_path)
    seller = kie_api['result'].get('SELLER')
    address = kie_api['result'].get('ADDRESS')
    time = kie_api['result'].get('TIMESTAMP')
    total = kie_api['result'].get('TOTAL_COST')
    # bb_field = lev(detect_task1, detect_task2, [seller, address, time, total])
    end_time = datetime.now()
    print('Total time process: {}'.format(end_time - start_time))
    return_result = {
        'seller': seller,
        'address': address,
        'time': time,
        'total': total,
        'time_execution': str(end_time - start_time)
    }

    return return_result
