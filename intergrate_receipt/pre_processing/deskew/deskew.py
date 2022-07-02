import math
from typing import Tuple, Union

import cv2
import numpy as np

from deskew import determine_skew


def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def deskew(image, output_folder_path):
    image = cv2.imread(input_img_path)

    output_img_name = input_img_path.split('/')[-1]
    output_path = os.path.join(output_folder_path, output_img_name)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    print('deskew:',output_img_name, angle)
    if angle < 20 and angle > -20:      
        rotated = rotate(image, angle, (0, 0, 0))
        cv2.imwrite(output_path, rotated)
        return rotated
    else:
        cv2.imwrite(output_path, image)
        return image
    
    

import os
INPUT_FOLDER_IMAGES = 'tests'

path_img, dirs_img, files_img = next(os.walk(INPUT_FOLDER_IMAGES))
# path_txt, dirs_txt, files_txt = next(os.walk(INPUT_FOLDER_TXT))
print(len(files_img))
# print(len(files_txt))

OUTPUT_FOLDER_IMAGES = 'output'
if not os.path.exists(OUTPUT_FOLDER_IMAGES):
        os.mkdir(OUTPUT_FOLDER_IMAGES)


if __name__ == '__main__':
    for fn_img in files_img:
        image_path = os.path.join(path_img, fn_img)
        img = deskew(image_path, OUTPUT_FOLDER_IMAGES)
