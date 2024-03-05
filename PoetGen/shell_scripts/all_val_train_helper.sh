#!/bin/bash
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
/cvmfs/software.metacentrum.cz/spack18/software/linux-debian11-x86_64_v2/gcc-10.2.1/python-3.9.12-rg2lpmkxpcq423gx5gmedbyam7eibwtc/bin/python3.9 -m pip install --upgrade pip
export TMPDIR=$SCRATCHDIR
cd $SCRATCHDIR
pip install --target=$SCRATCHDIR torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/requirements.txt
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen
python3 train_validator.py --epochs_metre=$EPMETER --epochs_rhyme=$EPRHYME --epochs_year=$EPYEAR --tokenizer="$TOK" --pretrained_model="$MODEL" --input_type="$INPUT" --train_with_context=$CONTEXT