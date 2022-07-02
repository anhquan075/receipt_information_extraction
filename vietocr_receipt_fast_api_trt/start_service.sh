export PYTHONIOENCODING=UTF-8
export LANG=C.UTF-8
export PYTHONPATH=${PYTHONPATH}:/base/src
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/base
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/base/TensorRT-7.2.2.3/targets/x86_64-linux-gnu/lib/
uvicorn service:app --port 5000 --host 0.0.0.0 --workers 1
