#!/bin/bash

# EPOCH 4, EPOCH 8

#qsub -N MirrorEmbeddingsCZBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=base,  MODEL=./MIRROREMBED-CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsCZBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=all,  MODEL=./MIRROREMBED-CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsCZUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar' all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsCZUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsCZSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsCZSyllableBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsCZProcessedBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh
#qsub -N MirrorEmbeddingsCZProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 

#qsub -N MirrorEmbeddingsALTBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=spital/gpt2-small-czech-cs,  MODELTYPE=base,  MODEL=./MIRROREMBED-ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsALTBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=spital/gpt2-small-czech-cs,  MODELTYPE=all,  MODEL=./MIRROREMBED-ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsALTUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=spital/gpt2-small-czech-cs' all_model_train_helper.sh  
#qsub -N MirrorEmbeddingsALTUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh
#
#qsub -N MirrorEmbeddingsALTSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsALTSyllableBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsALTProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsALTProcessedBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh
#
#qsub -N MirrorEmbeddingsENBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=gpt2,  MODELTYPE=base,  MODEL=./MIRROREMBED-EN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsENBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=gpt2,  MODELTYPE=all,  MODEL=./MIRROREMBED-EN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsENUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=gpt2' all_model_train_helper.sh
#qsub -N MirrorEmbeddingsENUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh
#
#qsub -N MirrorEmbeddingsENSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsENSyllableBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsENProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-EN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsENProcessedBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-EN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsENALTBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=distilgpt2,  MODELTYPE=base,  MODEL=./MIRROREMBED-ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsENALTBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=distilgpt2,  MODELTYPE=all,  MODEL=./MIRROREMBED-ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsENALTUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=distilgpt2' all_model_train_helper.sh
#qsub -N MirrorEmbeddingsENALTUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh
#
#qsub -N MirrorEmbeddingsENALTSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsENALTSyllableBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsENALTProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsENALTProcessedBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh 

qsub -N RNNBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=RWKV/rwkv-4-169m-pile,  MODELTYPE=base,  MODEL=./RNN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 
qsub -N RNNBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=RWKV/rwkv-4-169m-pile,  MODELTYPE=all,  MODEL=./RNN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 

qsub -N RNNUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./RNN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile' all_model_train_helper.sh 
qsub -N RNNUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./RNN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 

qsub -N RNNSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./RNN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 
qsub -N RNNSyllableBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./RNN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 

qsub -N RNNProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./RNN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 
qsub -N RNNProcessedBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./RNN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh


# EPOCH 4, EPOCH 16

#qsub -N MirrorEmbeddingsCZBaseTokenizerNormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=base,  MODEL=./MIRROREMBED-CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsCZBaseTokenizerNormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16, TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=all,  MODEL=./MIRROREMBED-CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsCZUnicodeTokenizerNormalTextGPTBaseTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar' all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsCZUnicodeTokenizerNormalTextGPTAllTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsCZSyllableBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsCZSyllableBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsCZProcessedBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsCZProcessedBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh
#
#qsub -N MirrorEmbeddingsALTBaseTokenizerNormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=spital/gpt2-small-czech-cs,  MODELTYPE=base,  MODEL=./MIRROREMBED-ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsALTBaseTokenizerNormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16, TOKENIZER=spital/gpt2-small-czech-cs,  MODELTYPE=all,  MODEL=./MIRROREMBED-ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsALTUnicodeTokenizerNormalTextGPTBaseTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=spital/gpt2-small-czech-cs' all_model_train_helper.sh  
#qsub -N MirrorEmbeddingsALTUnicodeTokenizerNormalTextGPTAllTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh
#
#qsub -N MirrorEmbeddingsALTSyllableBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsALTSyllableBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsALTProcessedBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsALTProcessedBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh
#
#qsub -N MirrorEmbeddingsENBaseTokenizerNormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=gpt2,  MODELTYPE=base,  MODEL=./MIRROREMBED-EN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsENBaseTokenizerNormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16, TOKENIZER=gpt2,  MODELTYPE=all,  MODEL=./MIRROREMBED-EN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsENUnicodeTokenizerNormalTextGPTBaseTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=gpt2' all_model_train_helper.sh
#qsub -N MirrorEmbeddingsENUnicodeTokenizerNormalTextGPTAllTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh
#
#qsub -N MirrorEmbeddingsENSyllableBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsENSyllableBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsENProcessedBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-EN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsENProcessedBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-EN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsENALTBaseTokenizerNormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=distilgpt2,  MODELTYPE=base,  MODEL=./MIRROREMBED-ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsENALTBaseTokenizerNormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16, TOKENIZER=distilgpt2,  MODELTYPE=all,  MODEL=./MIRROREMBED-ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsENALTUnicodeTokenizerNormalTextGPTBaseTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=distilgpt2' all_model_train_helper.sh
#qsub -N MirrorEmbeddingsENALTUnicodeTokenizerNormalTextGPTAllTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh
#
#qsub -N MirrorEmbeddingsENALTSyllableBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsENALTSyllableBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsENALTProcessedBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsENALTProcessedBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh 


qsub -N RNNBaseTokenizerNormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=RWKV/rwkv-4-169m-pile,  MODELTYPE=base,  MODEL=./RNN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 
qsub -N RNNBaseTokenizerNormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16, TOKENIZER=RWKV/rwkv-4-169m-pile,  MODELTYPE=all,  MODEL=./RNN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 

qsub -N RNNUnicodeTokenizerNormalTextGPTBaseTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./RNN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile' all_model_train_helper.sh 
qsub -N RNNUnicodeTokenizerNormalTextGPTAllTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./RNN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 

qsub -N RNNSyllableBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./RNN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 
qsub -N RNNSyllableBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./RNN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 

qsub -N RNNProcessedBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./RNN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 
qsub -N RNNProcessedBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./RNN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh

# LARGE MODELS


#qsub -N MirrorEmbeddingsLargeBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=stabilityai/StableBeluga-7B,  MODELTYPE=base,  MODEL=./MIRROREMBED-Large-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsLargeBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=stabilityai/StableBeluga-7B,  MODELTYPE=all,  MODEL=./MIRROREMBED-Large-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsLargeUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-Large-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=stabilityai/StableBeluga-7B' all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsLargeUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-Large-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsLargeSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-Large-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsLargeSyllableBPENormalTextGPTAllTasksE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-Large-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh 
#
#qsub -N MirrorEmbeddingsLargeProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./MIRROREMBED-Large-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh 
#qsub -N MirrorEmbeddingsLargeProcessedBPENormalTextGPTAllTasksE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./MIRROREMBED-Large-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh