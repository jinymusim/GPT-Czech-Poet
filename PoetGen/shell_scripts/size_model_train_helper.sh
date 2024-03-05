#!/bin/bash
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
python3 -m pip install --upgrade pip
export TMPDIR=$SCRATCHDIR
cd $SCRATCHDIR
pip install --target=$SCRATCHDIR torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --target=$SCRATCHDIR -r /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/requirements.txt
export PYTHONPATH="${PYTHONPATH}:${TMPDIR}"
cd /storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen
python3 lm_finetune_torch_trainer_api.py --epochs_LM=$EPOCHSLM --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL" --size_test=true --sizes_to_test=1
python3 lm_finetune_torch_trainer_api.py --epochs_LM=$EPOCHSLM --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL" --size_test=true --sizes_to_test=0.5
python3 lm_finetune_torch_trainer_api.py --epochs_LM=$EPOCHSLM --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL" --size_test=true --sizes_to_test=0.25
python3 lm_finetune_torch_trainer_api.py --epochs_LM=$EPOCHSLM --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL" --size_test=true --sizes_to_test=0.1
python3 lm_finetune_torch_trainer_api.py --epochs_LM=$EPOCHSLM --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL" --size_test=true --sizes_to_test=0.05
python3 lm_finetune_torch_trainer_api.py --epochs_LM=$EPOCHSLM --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL" --size_test=true --sizes_to_test=0.025
python3 lm_finetune_torch_trainer_api.py --epochs_LM=$EPOCHSLM --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL" --size_test=true --sizes_to_test=0.001
python3 lm_finetune_torch_trainer_api.py --epochs_LM=$EPOCHSLM --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL" --size_test=true --sizes_to_test=0.0005
python3 lm_finetune_torch_trainer_api.py --epochs_LM=$EPOCHSLM --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL" --size_test=true --sizes_to_test=0.00025
python3 lm_finetune_torch_trainer_api.py --epochs_LM=$EPOCHSLM --epochs_poet=$EPOCHSPOET --tokenizer="$TOKENIZER" --model_type="$MODELTYPE" --model_path="$MODEL" --default_hf_model="$HFMODEL" --size_test=true --sizes_to_test=0.00001