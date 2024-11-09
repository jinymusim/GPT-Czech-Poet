#!/bin/bash
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
export TMPDIR=$SCRATCHDIR 
cd $SCRATCHDIR
python3 -m venv TRAIN_VENV
source TRAIN_VENV/bin/activate
pip install  torch torchvision torchaudio -U
pip install -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/requirements.txt -U
pip install git+https://github.com/huggingface/transformers -U
git clone https://github.com/huggingface/trl.git
cd trl/
pip install -e .
pip install flash-attn --no-build-isolation -U
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen
python3 lm_finetune_torch_trainer_api.py  --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL"