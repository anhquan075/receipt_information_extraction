FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ARG DEBIAN_FRONTEND=noninteractive 

RUN apt-get update \
    && apt-get install ffmpeg libsm6 libxext6 wget  -y

WORKDIR /workingspace

COPY requirements.txt /workingspace

RUN pip install -U pip gdown && pip install -r requirements.txt

COPY . /workingspace/

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dsA2HKRplzjM5XJuy1kV9md9IHSfWdso' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dsA2HKRplzjM5XJuy1kV9md9IHSfWdso" -O saved_models/model_best.pth && rm -rf /tmp/cookies.txt

CMD ["python", "service.py"]