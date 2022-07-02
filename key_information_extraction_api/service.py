import cv2
import numpy as np
import time
import logging
import traceback
import os
import io
import requests
import random
import json
import shutil
import csv
from time import gmtime, strftime
from collections import defaultdict

from flask import Flask, render_template, Response, request, jsonify, send_file

from utils.parser import get_config
# from utils.utils import load_class_names, get_image

import cv2
import numpy as np
import pandas as pd


# from model.predict import load_images_to_predict

from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import InvalidTagSequence

import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str
from utils.visualize import viz_output_of_pick
import time

from typing import List, Tuple
import torch
from pathlib import Path

TypedStringSpan = Tuple[str, Tuple[int, int]]

# create backup dir
if not os.path.exists('backup'):
    os.mkdir('backup')

# create json dir
if not os.path.exists('json_dir'):
    os.mkdir('json_dir')

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')

# create log_file, rcode
LOG_PATH = cfg.SERVICE.LOG_PATH
HOST = cfg.SERVICE.SERVICE_IP
PORT = cfg.SERVICE.SERVICE_PORT
MODEL_PATH = cfg.SERVICE.MODEL_PATH
OUTPUT_DIR = cfg.SERVICE.OUTPUT_DIR
KIE_VISUALIZE = cfg.SERVICE.KIE_VISUALIZE

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time()) + ".log"), filemode="w", level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device)

config = checkpoint['config']
state_dict = checkpoint['state_dict']
monitor_best = checkpoint['monitor_best']
print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(MODEL_PATH, monitor_best))

# prepare model for testing
pick_model = config.init_obj('model_arch', pick_arch_module)
pick_model = pick_model.to(device)
pick_model.load_state_dict(state_dict)
pick_model.eval()


def bio_tags_to_spans2(
        tag_sequence: List[str], text_length: List[int] = None
) -> List[TypedStringSpan]:
    list_idx_to_split = [0]
    init_idx = 0
    for text_len in text_length[0]:
        init_idx += text_len
        list_idx_to_split.append(init_idx)

    spans = []
    line_pos_from_bottom = []
    for index, string_tag in enumerate(tag_sequence):
        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "O"]:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]

        if bio_tag == "B":
            if index in list_idx_to_split:
                idx_start = list_idx_to_split.index(index)
                idx_end = list_idx_to_split[idx_start + 1] - 1
                spans.append((conll_tag, (index, idx_end)))
                line_pos_from_bottom.append(idx_start)
    return spans, line_pos_from_bottom


def get_list_coors_from_line_pos_from_bottom(img_dir, file_name, boxes_coors, list_line, resize=[560, 784]):
    list_coor = []

    img = cv2.imread(os.path.join(img_dir, file_name))
    h, w, _ = img.shape
    res_x = w / resize[0]
    res_y = h / resize[1]

    for line_idx in list_line:
        coors = boxes_coors[line_idx]
        ori_coors = ','.join([str(int(coors[2] * res_x)), str(int(coors[3] * res_y)),
                              str(int(coors[4] * res_x)), str(int(coors[5] * res_y)),
                              str(int(coors[6] * res_x)), str(int(coors[7] * res_y)),
                              str(int(coors[0] * res_x)), str(int(coors[1] * res_y))])
        list_coor.append(ori_coors)
    return list_coor


def extract_data(bboxes_transcripts, image_folder_path, output_path):
    test_dataset = PICKDataset(boxes_and_transcripts_folder=bboxes_transcripts,
                               images_folder=image_folder_path,
                               resized_image_size=(560, 784),
                               ignore_error=False,
                               training=False,
                               max_boxes_num=130,
                               max_transcript_len=70)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                                  num_workers=1, collate_fn=BatchCollateFn(training=False))

    # setup output path
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # predict and save to file
    now_start = time.time()
    with torch.no_grad():
        for step_idx, input_data_item in enumerate(test_data_loader):
            now = time.time()
            for key, input_value in input_data_item.items():
                if input_value is not None:
                    input_data_item[key] = input_value.to(device)
            output = pick_model(**input_data_item)
            logits = output['logits']
            new_mask = output['new_mask']
            print(output.keys(), input_data_item.keys())
            image_indexs = input_data_item['image_indexs']  # (B,)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
            mask = input_data_item['mask']
            text_length = input_data_item['text_length']
            # boxes_coors = input_data_item['boxes_coordinate'].cpu().numpy()[0]
            # List[(List[int], torch.Tensor)]
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)

            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)
            for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                # List[ Tuple[str, Tuple[int, int]] ]
                # spans = bio_tags_to_spans(decoded_tags, [])
                spans, _ = bio_tags_to_spans2(decoded_tags, text_length.cpu().numpy())
                # spans = sorted(spans, key=lambda x: x[1][0])

                entities = []  # exists one to many case
                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)

                # result_file_dir = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem)
                # result_file_dir.mkdir(exist_ok=True)
                # result_file = result_file_dir.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                # base_filename = os.path.basename(result_file)

                entities_lst = []
                for item in entities:
                    entities_lst.append([item['entity_name'], item['text']])

                default_entities_dict = defaultdict(list)
                for ent, text in entities_lst:
                    default_entities_dict[ent].append(text)

    logger.info('time run program:'.format(time.time() - now_start))

    print(default_entities_dict)
    return default_entities_dict


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']

        image_file = file.read()
        image = cv2.imdecode(np.frombuffer(image_file, dtype=np.uint8), -1)

        data = json.load(request.files['data'])
        image_path_dir = data.get('image_path')

        # Create folder contain txt file
        output_image_dir_output = Path(image_path_dir.split(".")[0])
        output_image_dir_output.mkdir(parents=True, exist_ok=True)
        base_name = os.path.basename(image_path_dir)
        cv2.imwrite(os.path.join(output_image_dir_output, base_name), image)

        output_txt_dir = data.get('output_tsv_path')

        # Create folder contain txt file
        output_txt_dir_output = Path(output_txt_dir.split(".")[0])
        output_txt_dir_output.mkdir(parents=True, exist_ok=True)

        tsv_file_path = os.path.join(output_txt_dir_output,
                                     output_txt_dir.split("/")[-1].split(".")[0]) + '.tsv'

        data = request.files['tsv']
        data.save(tsv_file_path)

        location_tsv = output_txt_dir_output
        location_image = output_image_dir_output
        print(location_tsv, location_image)
        result = extract_data(location_tsv, location_image, OUTPUT_DIR)

    with open(OUTPUT_DIR + '/result.json', 'w', encoding='utf-8') as outfile:
        json.dump(result, outfile)
    return jsonify(result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
