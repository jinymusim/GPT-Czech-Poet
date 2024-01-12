# Testing for SAM

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=128,  EPRHYME=128, EPYEAR=128, TOK=./utils/tokenizers/BPE/new_class_processed_tokenizer.json, MODEL=distilroberta-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=128,  EPRHYME=128, EPYEAR=128, TOK=./utils/tokenizers/BPE/new_class_processed_tokenizer.json, MODEL=ufal/robeczech-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=128,  EPRHYME=128, EPYEAR=128, TOK=./utils/tokenizers/BPE/new_class_processed_tokenizer.json, MODEL=roberta-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=128,  EPRHYME=128, EPYEAR=128, TOK=./utils/tokenizers/BPE/new_class_syllabs_processed_tokenizer.json, MODEL=distilroberta-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=128,  EPRHYME=128, EPYEAR=128, TOK=./utils/tokenizers/BPE/new_class_syllabs_processed_tokenizer.json, MODEL=ufal/robeczech-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=128,  EPRHYME=128, EPYEAR=128, TOK=./utils/tokenizers/BPE/new_class_syllabs_processed_tokenizer.json, MODEL=roberta-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=16, EPYEAR=16, TOK=distilroberta-base, MODEL=distilroberta-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=16, EPYEAR=16, TOK=ufal/robeczech-base,  MODEL=ufal/robeczech-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=16, EPYEAR=16, TOK=roberta-base,  MODEL=roberta-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=16, EPYEAR=16, TOK=distilroberta-base, MODEL=distilroberta-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=16, EPYEAR=16, TOK=ufal/robeczech-base,  MODEL=ufal/robeczech-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=16, EPYEAR=16, TOK=roberta-base,  MODEL=roberta-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh

# Test of Multi GPU for bigger batch (256)

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=distilroberta-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=ufal/robeczech-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=roberta-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=distilroberta-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=ufal/robeczech-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=roberta-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh 