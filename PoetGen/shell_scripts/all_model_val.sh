# E4E16

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh 
qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh 

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh 
qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh  

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh 
qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh  

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh  
qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-VerseMarks-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM  shell_scripts/all_model_val_helper.sh  
qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-VerseMarks-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM  shell_scripts/all_model_val_helper.sh


#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh  

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh  

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh 
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh 

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh 


#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh 

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh 

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh 

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh 


#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh 

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh  

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh  

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh  


# E4E8

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh 
 
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh  
 
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh  

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh    
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh 


#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh 

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh 

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh 

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh  


#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh 

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh  

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh 

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./ENALT-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh  


#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh  

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh  

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh  

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh  
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./EN-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh  

# E0E24

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e0e24_LM shell_scripts/all_model_val_helper.sh

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e0e24_LM shell_scripts/all_model_val_helper.sh

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e0e24_LM shell_scripts/all_model_val_helper.sh

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e0e24_LM shell_scripts/all_model_val_helper.sh

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./CZ-VerseMarks-BPE-NormalText-gpt-cz-poetry-all-e0e24_LM shell_scripts/all_model_val_helper.sh

# Format + Pretarin

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./gpt-cz-poetry-basic-format-e4e16_LM shell_scripts/all_model_val_helper.sh
qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./Base-Tokenizer-gpt-cz-poetry-verse-param-format-e4e16_LM shell_scripts/all_model_val_helper.sh

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./Base-Tokenizer-gpt-cz-poetry-verse-param-format-e0e4-pretrained_LM shell_scripts/all_model_val_helper.sh
qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v MODEL=./gpt-cz-poetry-basic-format-e0e4-pretrained_LM shell_scripts/all_model_val_helper.sh

# E8E32

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v MODEL=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e8e32_LM shell_scripts/all_model_val_helper.sh
qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v MODEL=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e8e32_LM shell_scripts/all_model_val_helper.sh
qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v MODEL=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e8e32_LM shell_scripts/all_model_val_helper.sh
qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v MODEL=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e8e32_LM shell_scripts/all_model_val_helper.sh


# Diff ENDS

qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v MODEL=./CZ-Base-Tokenizer-2-SYLLABLE-END-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh
qsub -N ModelValidation -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=48:00:00 -v MODEL=./CZ-Base-Tokenizer-VERSE-MARK-END-gpt-cz-poetry-base-e4e16_LM shell_scripts/all_model_val_helper.sh


# RNN

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./RNN-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM shell_scripts/all_model_val_helper.sh
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./RNN-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8_LM shell_scripts/all_model_val_helper.sh

# Distil Models

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./Distil-Base-gpt-cz-poetry-e4e16_LM shell_scripts/all_model_val_helper.sh
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./Distil-Processed-gpt-cz-poetry-e4e16_LM shell_scripts/all_model_val_helper.sh
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./Distil-Syllable-gpt-cz-poetry-e4e16_LM shell_scripts/all_model_val_helper.sh
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./Distil-Unicode-gpt-cz-poetry-e4e16_LM shell_scripts/all_model_val_helper.sh

#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./Distil-Base-gpt-cz-poetry-e16e32_LM shell_scripts/all_model_val_helper.sh
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./Distil-Processed-gpt-cz-poetry-e16e32_LM shell_scripts/all_model_val_helper.sh
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./Distil-Syllable-gpt-cz-poetry-e16e32_LM shell_scripts/all_model_val_helper.sh
#qsub -N ModelValidation -q default -l select=1:ncpus=1:mem=36gb:scratch_local=20gb -l walltime=128:00:00 -v MODEL=./Distil-Unicode-gpt-cz-poetry-e16e32_LM shell_scripts/all_model_val_helper.sh