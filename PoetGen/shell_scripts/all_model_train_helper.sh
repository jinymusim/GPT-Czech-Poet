#!/bin/bash
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
export TMPDIR=$SCRATCHDIR 
cd $SCRATCHDIR
pip install --target=$SCRATCHDIR torch torchvision torchaudio -U
pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/requirements.txt -U
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
git clone https://github.com/huggingface/trl.git
cd trl/
pip install --target=$SCRATCHDIR -e -U .
pip install --target=$SCRATCHDIR flash-attn --no-build-isolation -U
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen
python3 lm_finetune_torch_trainer_api.py  --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL"