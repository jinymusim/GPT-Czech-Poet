#!/bin/bash

# EPOCH 4, EPOCH 8 Effective batch 64, 48

#qsub -N CZBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=base,  MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#qsub -N CZBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=all,  MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 

#qsub -N CZUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar' all_model_train_helper.sh 
#qsub -N CZUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 

#qsub -N CZSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#qsub -N CZSyllableBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 

#qsub -N CZProcessedBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh
#qsub -N CZProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 

#qsub -N ALTBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=spital/gpt2-small-czech-cs,  MODELTYPE=base,  MODEL=./ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#qsub -N ALTBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=spital/gpt2-small-czech-cs,  MODELTYPE=all,  MODEL=./ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 

#qsub -N ALTUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=spital/gpt2-small-czech-cs' all_model_train_helper.sh  
#qsub -N ALTUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh

#qsub -N ALTSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#qsub -N ALTSyllableBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 

#qsub -N ALTProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#qsub -N ALTProcessedBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh

#qsub -N ENBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=gpt2,  MODELTYPE=base,  MODEL=./EN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh 
#qsub -N ENBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=gpt2,  MODELTYPE=all,  MODEL=./EN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh 

#qsub -N ENUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=gpt2' all_model_train_helper.sh
#qsub -N ENUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh

#qsub -N ENSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh 
#qsub -N ENSyllableBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh 

#qsub -N ENProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./EN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh 
#qsub -N ENProcessedBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./EN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=gpt2'  all_model_train_helper.sh 

#qsub -N ENALTBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=distilgpt2,  MODELTYPE=base,  MODEL=./ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#qsub -N ENALTBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=distilgpt2,  MODELTYPE=all,  MODEL=./ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh 

#qsub -N ENALTUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=distilgpt2' all_model_train_helper.sh
#qsub -N ENALTUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh

#qsub -N ENALTSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#qsub -N ENALTSyllableBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh 

#qsub -N ENALTProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#qsub -N ENALTProcessedBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=distilgpt2'  all_model_train_helper.sh 

#qsub -N RNNBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=RWKV/rwkv-4-169m-pile,  MODELTYPE=base,  MODEL=./RNN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 
#qsub -N RNNBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=RWKV/rwkv-4-169m-pile,  MODELTYPE=all,  MODEL=./RNN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 

#qsub -N RNNUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./RNN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile' all_model_train_helper.sh 
#qsub -N RNNUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./RNN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 

qsub -N RNNSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./RNN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 
qsub -N RNNSyllableBPENormalTextGPTAllTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./RNN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 

qsub -N RNNProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./RNN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 
qsub -N RNNProcessedBPENormalTextGPTAllTasksE4E8 -q gpu_long -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./RNN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh


# EPOCH 4, EPOCH 16

#qsub -N CZBaseTokenizerNormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=base,  MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#qsub -N CZBaseTokenizerNormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16, TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=all,  MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 

#qsub -N CZUnicodeTokenizerNormalTextGPTBaseTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar' all_model_train_helper.sh 
#qsub -N CZUnicodeTokenizerNormalTextGPTAllTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 

#qsub -N CZSyllableBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#qsub -N CZSyllableBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 

#qsub -N CZProcessedBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh 
#qsub -N CZProcessedBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=lchaloupsky/czech-gpt2-oscar'  all_model_train_helper.sh

#qsub -N ALTBaseTokenizerNormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=spital/gpt2-small-czech-cs,  MODELTYPE=base,  MODEL=./ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#qsub -N ALTBaseTokenizerNormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16, TOKENIZER=spital/gpt2-small-czech-cs,  MODELTYPE=all,  MODEL=./ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 

#qsub -N ALTUnicodeTokenizerNormalTextGPTBaseTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=spital/gpt2-small-czech-cs' all_model_train_helper.sh  
#qsub -N ALTUnicodeTokenizerNormalTextGPTAllTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh

#qsub -N ALTSyllableBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#qsub -N ALTSyllableBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 

#qsub -N ALTProcessedBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh 
#qsub -N ALTProcessedBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=spital/gpt2-small-czech-cs'  all_model_train_helper.sh

#qsub -N ENBaseTokenizerNormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=gpt2,  MODELTYPE=base,  MODEL=./EN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh 
#qsub -N ENBaseTokenizerNormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16, TOKENIZER=gpt2,  MODELTYPE=all,  MODEL=./EN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh 

#qsub -N ENUnicodeTokenizerNormalTextGPTBaseTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=gpt2' all_model_train_helper.sh
#qsub -N ENUnicodeTokenizerNormalTextGPTAllTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh

#qsub -N ENSyllableBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh 
#qsub -N ENSyllableBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh 

#qsub -N ENProcessedBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./EN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh 
#qsub -N ENProcessedBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./EN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=gpt2'  all_model_train_helper.sh 

#qsub -N ENALTBaseTokenizerNormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=distilgpt2,  MODELTYPE=base,  MODEL=./ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#qsub -N ENALTBaseTokenizerNormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16, TOKENIZER=distilgpt2,  MODELTYPE=all,  MODEL=./ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh 

#qsub -N ENALTUnicodeTokenizerNormalTextGPTBaseTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=distilgpt2' all_model_train_helper.sh
#qsub -N ENALTUnicodeTokenizerNormalTextGPTAllTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh

#qsub -N ENALTSyllableBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#qsub -N ENALTSyllableBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh 

#qsub -N ENALTProcessedBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh 
#qsub -N ENALTProcessedBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=distilgpt2'  all_model_train_helper.sh 

#qsub -N RNNBaseTokenizerNormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=RWKV/rwkv-4-169m-pile,  MODELTYPE=base,  MODEL=./RNN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 
#qsub -N RNNBaseTokenizerNormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16, TOKENIZER=RWKV/rwkv-4-169m-pile,  MODELTYPE=all,  MODEL=./RNN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 

#qsub -N RNNUnicodeTokenizerNormalTextGPTBaseTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./RNN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile' all_model_train_helper.sh 
#qsub -N RNNUnicodeTokenizerNormalTextGPTAllTasksE4E16 -q gpu_long -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./RNN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 

#qsub -N RNNSyllableBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./RNN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 
#qsub -N RNNSyllableBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./RNN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 

#qsub -N RNNProcessedBPENormalTextGPTBaseTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./RNN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh 
#qsub -N RNNProcessedBPENormalTextGPTAllTasksE4E16 -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=16,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./RNN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16, HFMODEL=RWKV/rwkv-4-169m-pile'  all_model_train_helper.sh


# LARGE MODELS

#qsub -N LargeBaseTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=stabilityai/StableBeluga-7B,  MODELTYPE=base,  MODEL=./Large-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh 
#qsub -N LargeBaseTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8, TOKENIZER=stabilityai/StableBeluga-7B,  MODELTYPE=all,  MODEL=./Large-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh 
#
#qsub -N LargeUnicodeTokenizerNormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./Large-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=stabilityai/StableBeluga-7B' all_model_train_helper.sh 
#qsub -N LargeUnicodeTokenizerNormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=all,  MODEL=./Large-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh 
#
#qsub -N LargeSyllableBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./Large-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh 
#qsub -N LargeSyllableBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./Large-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh 
#
#qsub -N LargeProcessedBPENormalTextGPTBaseTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./Large-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh 
#qsub -N LargeProcessedBPENormalTextGPTAllTasksE4E8 -q gpu -l select=1:ncpus=4:ngpus=2:mem=80gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v 'EPOCHSLM=4,  EPOCHSPOET=8,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=all,  MODEL=./Large-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8, HFMODEL=stabilityai/StableBeluga-7B'  all_model_train_helper.sh