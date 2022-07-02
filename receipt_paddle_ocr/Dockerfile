FROM nvidia/cuda:11.2.1-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN apt-get update && apt-get install python3-pip git wget ffmpeg libsm6 libxext6  -y 
RUN git clone https://github.com/PaddlePaddle/PaddleOCR --depth 1
WORKDIR /PaddleOCR

RUN pip3 install paddlepaddle-gpu==2.2.2 && \
    pip3 install paddlehub==2.1.0 onnxruntime==1.9.0 paddle2onnx && pip3 install -r requirements.txt 

RUN pip3 install onnx protobuf==3.20

RUN wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar && \
    cd ./inference && tar xf ch_PP-OCRv2_det_infer.tar && cd ..

RUN paddle2onnx --model_dir ./inference/ch_PP-OCRv2_det_infer \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_file ./inference/det_onnx/model.onnx \
    --opset_version 10 \
    --input_shape_dict="{'x':[-1,3,-1,-1]}" \
    --enable_onnx_checker True

RUN hub install deploy/hubserving/ocr_det/

COPY params.py /PaddleOCR/deploy/hubserving/ocr_system/

CMD ["hub", "serving", "start", "-m", "ocr_det"]
