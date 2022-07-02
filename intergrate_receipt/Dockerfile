FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y python3-pip wget llvm && \
    apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    apt-get install ffmpeg libsm6 libxext6  -y 

RUN apt-get install tesseract-ocr -y

RUN pip3 install -U pip && pip3 install pillow more_itertools sklearn pandas unidecode tqdm \
    recognizers_text recognizers_date_time Levenshtein pytesseract shapely numpy opencv-python \
    configparser==5.0.2 six==1.16.0 future==0.18.2 python-multipart uvicorn==0.14.0 fastapi==0.65.2 requests pymongo

WORKDIR /base

COPY . /base

CMD sh start_service.sh