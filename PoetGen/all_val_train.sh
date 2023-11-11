#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=16, EPYEAR=0, TOK=xlm-roberta-base, SYLAB=true'  all_val_train_helper.sh 
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=0, EPYEAR=0, TOK=xlm-roberta-base, SYLAB=true'  all_val_train_helper.sh 
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=0, EPYEAR=16, TOK=xlm-roberta-base, SYLAB=true'  all_val_train_helper.sh 
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=0, EPYEAR=16, TOK=xlm-roberta-base, SYLAB=False'  all_val_train_helper.sh 
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=16, EPYEAR=0, TOK=xlm-roberta-base, SYLAB=False'  all_val_train_helper.sh 
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=0, EPYEAR=0, TOK=xlm-roberta-base, SYLAB=False'  all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=16, EPYEAR=0, TOK=distilroberta-base, SYLAB=true'  all_val_train_helper.sh 
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=0, EPYEAR=0, TOK=distilroberta-base, SYLAB=true'  all_val_train_helper.sh 
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=0, EPYEAR=16, TOK=distilroberta-base, SYLAB=true'  all_val_train_helper.sh 
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=0, EPYEAR=16, TOK=distilroberta-base, SYLAB=False'  all_val_train_helper.sh 
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=16, EPYEAR=0, TOK=distilroberta-base, SYLAB=False'  all_val_train_helper.sh 
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=0, EPYEAR=0, TOK=distilroberta-base, SYLAB=False'  all_val_train_helper.sh 

#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=0, EPYEAR=16, TOK=ufal/robeczech-base, SYLAB=False'  all_val_train_helper.sh 
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=16, EPYEAR=0, TOK=ufal/robeczech-base, SYLAB=False'  all_val_train_helper.sh 
#qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=0, EPYEAR=0, TOK=ufal/robeczech-base, SYLAB=False'  all_val_train_helper.sh 

qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=0, EPYEAR=16, TOK=roberta-base, SYLAB=False'  all_val_train_helper.sh 
qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=0,  EPRHYME=16, EPYEAR=0, TOK=roberta-base, SYLAB=False'  all_val_train_helper.sh 
qsub -N TrainVal -q gpu_dgx -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=300:00:00 -v 'EPMETER=16,  EPRHYME=0, EPYEAR=0, TOK=roberta-base, SYLAB=False'  all_val_train_helper.sh 