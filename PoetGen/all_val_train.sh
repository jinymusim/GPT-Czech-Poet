# Testing for SAM

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=0, EPYEAR=128, TOK=distilroberta-base, SYLAB=false, SAM=false'  all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=0, EPYEAR=128, TOK=ufal/robeczech-base, SYLAB=false, SAM=false'  all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=0, EPYEAR=128, TOK=roberta-base, SYLAB=false, SAM=false'  all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=128,  EPRHYME=0, EPYEAR=0, TOK=distilroberta-base, SYLAB=true, SAM=false'  all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=128,  EPRHYME=0, EPYEAR=0, TOK=ufal/robeczech-base, SYLAB=true, SAM=false'  all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=128,  EPRHYME=0, EPYEAR=0, TOK=roberta-base, SYLAB=true, SAM=false'  all_val_train_helper.sh

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=32,  EPRHYME=32, EPYEAR=32, TOK=distilroberta-base, SYLAB=false, SAM=true'  all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=32,  EPRHYME=32, EPYEAR=32, TOK=ufal/robeczech-base, SYLAB=false, SAM=true'  all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=32,  EPRHYME=32, EPYEAR=32, TOK=roberta-base, SYLAB=false, SAM=true'  all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=32,  EPRHYME=32, EPYEAR=32, TOK=distilroberta-base, SYLAB=true, SAM=true'  all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=32,  EPRHYME=32, EPYEAR=32, TOK=ufal/robeczech-base, SYLAB=true, SAM=true'  all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=32,  EPRHYME=32, EPYEAR=32, TOK=roberta-base, SYLAB=true, SAM=true'  all_val_train_helper.sh

# Test of Multi GPU for bigger batch (256)

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=distilroberta-base, SYLAB=false, SAM=false'  all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=ufal/robeczech-base, SYLAB=false, SAM=false'  all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=roberta-base, SYLAB=false, SAM=false'  all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=distilroberta-base, SYLAB=true, SAM=false'  all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=ufal/robeczech-base, SYLAB=true, SAM=false'  all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=64,  EPRHYME=64, EPYEAR=64, TOK=roberta-base, SYLAB=true, SAM=false'  all_val_train_helper.sh 