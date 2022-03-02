export PYTHONIOENCODING=UTF-8
export LANG=C.UTF-8
export PYTHONPATH=${PYTHONPATH}:/base/post_processing
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/base
uvicorn service:app --port 5004 --host 0.0.0.0 --workers 1