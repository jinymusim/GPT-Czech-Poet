qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM all_model_val_helper.sh  
qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=24gb:scratch_local=20gb -l walltime=72:00:00 -v MODEL=./backup_LMS/EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM all_model_val_helper.sh  