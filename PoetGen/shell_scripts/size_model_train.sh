#!/bin/bash

# Size Models

qsub -N SizeTestBaseE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=200:00:00 -v 'EPOCHSLM=2,  EPOCHSPOET=4,  TOKENIZER=lchaloupsky/czech-gpt2-oscar,  MODELTYPE=base,  MODEL=./SizeTest-Base-gpt-cz-poetry-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/size_model_train_helper.sh
qsub -N SizeTestUnicodeE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=200:00:00 -v 'EPOCHSLM=2,  EPOCHSPOET=4,  TOKENIZER=./utils/tokenizers/Unicode/unicode_tokenizer.json,  MODELTYPE=base,  MODEL=./SizeTest-Unicode-gpt-cz-poetry-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/size_model_train_helper.sh 
qsub -N SizeTestSyllableE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=200:00:00 -v 'EPOCHSLM=2,  EPOCHSPOET=4,  TOKENIZER=./utils/tokenizers/BPE/new_syllabs_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./SizeTest-Syllable-gpt-cz-poetry-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/size_model_train_helper.sh 
qsub -N SizeTestProcessedE4E8 -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=200:00:00 -v 'EPOCHSLM=2,  EPOCHSPOET=4,  TOKENIZER=./utils/tokenizers/BPE/new_processed_tokenizer.json,  MODELTYPE=base,  MODEL=./SizeTest-Processed-gpt-cz-poetry-e4e8, HFMODEL=lchaloupsky/czech-gpt2-oscar'  shell_scripts/size_model_train_helper.sh
