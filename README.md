# Information Extraction for Vietnamese Receipts in the wild

This repository aims to build a API for extracting Vietnamese Receipts in the wild in real time.
____
## 1) Our processing pipeline:  

TBD

## 2) How to run: 
Using ```docker-compose``` to easily run the project. You can change the local database environment in ```config/db.env```.
```
docker-compose --env-file=config/db.env up --build
```

**Note:**
- The VietOCR TensorRT model was build on Tesla T40. You need at least a GPU with 16GB VRAM to run. To run on GPU, please follow [this repository](https://github.com/NNDam/vietocr-tensorrt) to rebuild.