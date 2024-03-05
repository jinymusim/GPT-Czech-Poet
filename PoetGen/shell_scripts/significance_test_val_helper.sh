#!/bin/bash
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
/cvmfs/software.metacentrum.cz/spack18/software/linux-debian11-x86_64_v2/gcc-10.2.1/python-3.9.12-rg2lpmkxpcq423gx5gmedbyam7eibwtc/bin/python3.9 -m pip install --upgrade pip
export TMPDIR=$SCRATCHDIR
cd $SCRATCHDIR
pip install --target=$SCRATCHDIR torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/requirements.txt
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen
python3 test_validator_significance.py --base_val_model_path_full=$BASE --improved_val_model_path_full=$IMPROVED --base_validator_tokenizer_model=$BASETOK --improved_validator_tokenizer_model=$IMPROVEDTOK --base_val_syllables=$BASESYL --improved_val_syllables=$IMPROVEDSYL --validator_type=$TYPE --base_meter_context=$BASECONTEXT --improved_meter_context=$IMPROVEDCONTEXT