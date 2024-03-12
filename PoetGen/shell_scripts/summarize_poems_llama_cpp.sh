#!/bin/bash
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
export TMPDIR=$SCRATCHDIR
pip install --target=$SCRATCHDIR torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install --target=$SCRATCHDIR  llama-cpp-python --upgrade 
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/utils
python test_sumarization_llama_cpp.py