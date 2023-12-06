qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  