qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=16, EPYEAR=16, TOK=distilroberta-base, SYLAB=false, SAM=false'  all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=16, EPYEAR=16, TOK=ufal/robeczech-base, SYLAB=false, SAM=false'  all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=16, EPYEAR=16, TOK=roberta-base, SYLAB=false, SAM=false'  all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=16, EPYEAR=16, TOK=distilroberta-base, SYLAB=true, SAM=false'  all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=16, EPYEAR=16, TOK=ufal/robeczech-base, SYLAB=true, SAM=false'  all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=4:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=16, EPYEAR=16, TOK=roberta-base, SYLAB=true, SAM=false'  all_val_train_helper.sh 