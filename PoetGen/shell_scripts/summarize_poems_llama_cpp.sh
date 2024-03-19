#!/bin/bash
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
export TMPDIR=$SCRATCHDIR
pip install --target=$SCRATCHDIR  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/requirements.txt
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --target=$SCRATCHDIR llama-cpp-python==0.2.55 --no-cache-dir --force-reinstall
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/utils
python test_sumarization_llama_cpp.py