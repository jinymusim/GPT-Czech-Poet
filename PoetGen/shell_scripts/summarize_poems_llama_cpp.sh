#!/bin/bash
module add python/3.10.4-gcc-8.3.0-ovkjwzd
python3 -m ensurepip --upgrade
export TMPDIR=$SCRATCHDIR
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
export FORCE_CMAKE=1
cd $SCRATCHDIR
python3 -m pip install --upgrade pip
pip install --target=$SCRATCHDIR --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --target=$SCRATCHDIR --upgrade --force-reinstall llama-cpp-python --no-cache-dir
pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/requirements.txt
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/utils
python3 test_sumarization_llama_cpp.py