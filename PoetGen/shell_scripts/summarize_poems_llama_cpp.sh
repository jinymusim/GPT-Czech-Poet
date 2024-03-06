#!/bin/bash
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
export TMPDIR=$SCRATCHDIR
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
export FORCE_CMAKE=1
cd $SCRATCHDIR
pip install --target=$SCRATCHDIR  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --target=$SCRATCHDIR  llama-cpp-python
pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/requirements.txt
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/utils
sudo python3 test_sumarization_llama_cpp.py