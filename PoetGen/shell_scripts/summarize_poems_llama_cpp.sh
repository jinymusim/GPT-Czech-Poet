#!/bin/bash
module add python/3.10.4-gcc-8.3.0-ovkjwzd
export TMPDIR=$SCRATCHDIR
python -m pip install --target=$SCRATCHDIR  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/requirements.txt
CMAKE_ARGS="-DLLAMA_CUBLAS=on" python -m pip install --target=$SCRATCHDIR  llama-cpp-python --no-cache-dir
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/utils
python test_sumarization_llama_cpp.py