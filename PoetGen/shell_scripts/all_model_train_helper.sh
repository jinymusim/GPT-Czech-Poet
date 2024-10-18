#!/bin/bash
module add python/3.10.4-gcc-8.3.0-ovkjwzd
export TMPDIR=$SCRATCHDIR 
cd $SCRATCHDIR
pip install --target=$SCRATCHDIR torch torchvision torchaudio
pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/requirements.txt
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen
python3 lm_finetune_torch_trainer_api.py  --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL"