
# Traing

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=distilbert/distilroberta-base, MODEL=distilbert/distilroberta-base, INPUT=BASE, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=distilbert/distilroberta-base, MODEL=distilbert/distilroberta-base, INPUT=SYLLABLE, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=distilbert/distilroberta-base, MODEL=distilbert/distilroberta-base, INPUT=VERSEMARK, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=FacebookAI/roberta-base, MODEL=FacebookAI/roberta-base, INPUT=BASE, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=FacebookAI/roberta-base, MODEL=FacebookAI/roberta-base, INPUT=SYLLABLE, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=FacebookAI/roberta-base, MODEL=FacebookAI/roberta-base, INPUT=VERSEMARK, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=ufal/robeczech-base, MODEL=ufal/robeczech-base, INPUT=BASE, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=ufal/robeczech-base, MODEL=ufal/robeczech-base, INPUT=SYLLABLE, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=ufal/robeczech-base, MODEL=ufal/robeczech-base, INPUT=VERSEMARK, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=microsoft/deberta-v3-base, MODEL=microsoft/deberta-v3-base, INPUT=BASE, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=microsoft/deberta-v3-base, MODEL=microsoft/deberta-v3-base, INPUT=SYLLABLE, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=microsoft/deberta-v3-base, MODEL=microsoft/deberta-v3-base, INPUT=VERSEMARK, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=albert/albert-base-v2, MODEL=albert/albert-base-v2, INPUT=BASE, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=albert/albert-base-v2, MODEL=albert/albert-base-v2, INPUT=SYLLABLE, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=albert/albert-base-v2, MODEL=albert/albert-base-v2, INPUT=VERSEMARK, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=google-bert/bert-base-uncased, MODEL=google-bert/bert-base-uncased, INPUT=BASE, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=google-bert/bert-base-uncased, MODEL=google-bert/bert-base-uncased, INPUT=SYLLABLE, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=16, EPYEAR=16, TOK=google-bert/bert-base-uncased, MODEL=google-bert/bert-base-uncased, INPUT=VERSEMARK, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 


# RHYME LONG TRAINING DONE 
# METER LONG TRAINING DONE 
#TODO: YEAR LONG TRAINING
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0, EPRHYME=64, EPYEAR=0, TOK=roberta-base,  MODEL=roberta-base, SYLAB=false, SAM=false, CONTEXT=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0, EPRHYME=64, EPYEAR=0, TOK=distilroberta-base, MODEL=distilroberta-base, SYLAB=true, SAM=false, CONTEXT=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0, EPRHYME=0, EPYEAR=64, TOK=ufal/robeczech-base,  MODEL=ufal/robeczech-base, SYLAB=false, SAM=false, CONTEXT=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64, EPRHYME=0, EPYEAR=0, TOK=roberta-base,  MODEL=roberta-base, SYLAB=true, SAM=false, CONTEXT=true'  shell_scripts/all_val_train_helper.sh

# Context Train

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=0, EPYEAR=0, TOK=distilroberta-base, MODEL=distilroberta-base, SYLAB=true, SAM=false, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=0, EPYEAR=0, TOK=ufal/robeczech-base,  MODEL=ufal/robeczech-base, SYLAB=true, SAM=false, CONTEXT=true'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16, EPRHYME=0, EPYEAR=0, TOK=roberta-base,  MODEL=roberta-base, SYLAB=true, SAM=false, CONTEXT=true'  shell_scripts/all_val_train_helper.sh


# Test of Multi GPU for bigger batch (256)

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=distilroberta-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=ufal/robeczech-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=roberta-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=distilroberta-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=ufal/robeczech-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=roberta-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh 

# Test XLM-Roberta

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=128,  EPRHYME=128, EPYEAR=128, TOK=xlm-roberta-base,  MODEL=xlm-roberta-base, SYLAB=false, SAM=false'  shell_scripts/all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=128,  EPRHYME=128, EPYEAR=128, TOK=xlm-roberta-base,  MODEL=xlm-roberta-base, SYLAB=true, SAM=false'  shell_scripts/all_val_train_helper.sh