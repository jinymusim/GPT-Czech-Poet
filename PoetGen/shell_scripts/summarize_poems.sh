#!/bin/bash
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
python3 -m pip install --upgrade pip
export TMPDIR=$SCRATCHDIR
cd $SCRATCHDIR
pip install --target=$SCRATCHDIR torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/requirements.txt
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/utils
python3 test_sumarization.py