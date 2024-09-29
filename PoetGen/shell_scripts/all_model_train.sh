#!/bin/bash

# TEST MODEL

#qsub -N TestModelE2E2 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=2,  EPOCHSPOET=2,  TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=base,  MODEL=./Test-Model-e2e2, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 

# 2 SYLLABLE TEST

#qsub -N CZBaseTokenizerNormalTextGPTBaseTasksE16 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=base,  MODEL=./CZ-Base-Tokenizer-VERSE-MARK-END-gpt-cz-poetry-base-e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 

# EPOCH 4, EPOCH 8 Effective batch 64, 48

#qsub -N CZBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=base,  MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 
#qsub -N CZBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=all,  MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 

#qsub -N CZUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar' shell_scripts/all_model_train_helper.sh 
#qsub -N CZUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 

#qsub -N CZSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 
#qsub -N CZSyllableBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 

#qsub -N CZProcessedBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh
#qsub -N CZProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 


# EPOCH 16

qsub -N CZBaseTokenizerNormalTextTinyLlamaBaseTasksE32 -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=256:00:00 -v 'EPOCHSPOET=32,  TOKENIZER=jinymusim/TinyLlama-Czech-Poet,  MODELTYPE=base,  MODEL=./CZ-Base-Tokenizer-NormalText-TinyLama-cz-poetry-base-e32, HFMODEL=jinymusim/TinyLlama-Czech-Poet'  shell_scripts/all_model_train_helper.sh 
#qsub -N CZBaseTokenizerNormalTextGPTAllTasksE16 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16, TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=all,  MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 

qsub -N CZUnicodeTokenizerNormalTextTinyLlamaBaseTasksE8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=96:00:00 -v 'EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./CZ-Unicode-Tokenizer-NormalText-TinyLama-cz-poetry-base-e8, HFMODEL=BUT-FIT/CSTinyLlama-1.2B' shell_scripts/all_model_train_helper.sh 
#qsub -N CZUnicodeTokenizerNormalTextGPTAllTasksE16 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 

qsub -N CZSyllableBPENormalTextTinyLlamaBaseTasksE8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=96:00:00 -v 'EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./CZ-New-Syllable-BPE-NormalText-TinyLama-cz-poetry-base-e8, HFMODEL=BUT-FIT/CSTinyLlama-1.2B'  shell_scripts/all_model_train_helper.sh 
#qsub -N CZSyllableBPENormalTextGPTAllTasksE16 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 

qsub -N CZProcessedBPENormalTextTinyLlamaBaseTasksE8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=96:00:00 -v 'EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./CZ-New-Processed-BPE-NormalText-TinyLama-cz-poetry-base-e8, HFMODEL=BUT-FIT/CSTinyLlama-1.2B'  shell_scripts/all_model_train_helper.sh 
#qsub -N CZProcessedBPENormalTextGPTAllTasksE16 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh


# Distil Models

#qsub -N DistilBaseE16E32 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=200:00:00 -v 'EPOCHSLM=16,  EPOCHSPOET=32,  TOKENIZER=/storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e16_LM,  MODELTYPE=distil,  MODEL=./Distil-Base-gpt-cz-poetry-e16e32, HFMODEL=/storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e16_LM'  shell_scripts/all_model_train_helper.sh
#qsub -N DistilUnicodeE16E32 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=200:00:00 -v 'EPOCHSLM=16,  EPOCHSPOET=32,  TOKENIZER=/storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e16_LM,  MODELTYPE=distil,  MODEL=./Distil-Unicode-gpt-cz-poetry-e16e32, HFMODEL=/storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e16_LM'  shell_scripts/all_model_train_helper.sh 
#qsub -N DistilSyllableE16E32 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=200:00:00 -v 'EPOCHSLM=16,  EPOCHSPOET=32,  TOKENIZER=/storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e16_LM,  MODELTYPE=distil,  MODEL=./Distil-Syllable-gpt-cz-poetry-e16e32, HFMODEL=/storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e16_LM'  shell_scripts/all_model_train_helper.sh 
#qsub -N DistilProcessedE16E32 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=200:00:00 -v 'EPOCHSLM=16,  EPOCHSPOET=32,  TOKENIZER=/storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e16_LM,  MODELTYPE=distil,  MODEL=./Distil-Processed-gpt-cz-poetry-e16e32, HFMODEL=/storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e16_LM'  shell_scripts/all_model_train_helper.sh


# LARGE MODELS

#qsub -N LargeBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=stabilityai/StableBeluga-7B,  MODELTYPE=base,  MODEL=./Large-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  shell_scripts/all_model_train_helper.sh 
#qsub -N LargeBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=stabilityai/StableBeluga-7B,  MODELTYPE=all,  MODEL=./Large-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  shell_scripts/all_model_train_helper.sh 
#
#qsub -N LargeUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./Large-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=stabilityai/StableBeluga-7B' shell_scripts/all_model_train_helper.sh 
#qsub -N LargeUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./Large-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  shell_scripts/all_model_train_helper.sh 
#
#qsub -N LargeSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./Large-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  shell_scripts/all_model_train_helper.sh 
#qsub -N LargeSyllableBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./Large-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  shell_scripts/all_model_train_helper.sh 
#
#qsub -N LargeProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./Large-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  shell_scripts/all_model_train_helper.sh 
#qsub -N LargeProcessedBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./Large-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  shell_scripts/all_model_train_helper.sh

# CUSTOM EMBED SIZE

#qsub -N EmbedBaseE128E256 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=200:00:00 -v 'EPOCHSLM=128,  EPOCHSPOET=256,  TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=small,  MODEL=./Embed-Base-gpt-cz-poetry-e128e256, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh
#qsub -N EmbedUnicodeE128E256 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=200:00:00 -v 'EPOCHSLM=128,  EPOCHSPOET=256,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=small,  MODEL=./Embed-Unicode-gpt-cz-poetry-e128e256, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 
#qsub -N EmbedSyllableE128E256 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=200:00:00 -v 'EPOCHSLM=128,  EPOCHSPOET=256,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=small,  MODEL=./Embed-Syllable-gpt-cz-poetry-e128e256, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh 
#qsub -N EmbedProcessedE128E256 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=200:00:00 -v 'EPOCHSLM=128,  EPOCHSPOET=256,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=small,  MODEL=./Embed-Processed-gpt-cz-poetry-e128e256, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/all_model_train_helper.sh
