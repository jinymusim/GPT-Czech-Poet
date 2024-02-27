qsub -N MeasurePerplexity -q gpu -l select=1:ncpus=1:ngpus=1:mem=40gb:gpu_mem=12gb:scratch_local=32gb -l walltime=24:00:00 -v MODEL=./backup_LMS/CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/perplexity_measure_helper.sh
qsub -N MeasurePerplexity -q gpu -l select=1:ncpus=1:ngpus=1:mem=40gb:gpu_mem=12gb:scratch_local=32gb -l walltime=24:00:00 -v MODEL=./backup_LMS/CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/perplexity_measure_helper.sh 

qsub -N MeasurePerplexity -q gpu -l select=1:ncpus=1:ngpus=1:mem=40gb:gpu_mem=12gb:scratch_local=32gb -l walltime=24:00:00 -v MODEL=./backup_LMS/CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/perplexity_measure_helper.sh 
qsub -N MeasurePerplexity -q gpu -l select=1:ncpus=1:ngpus=1:mem=40gb:gpu_mem=12gb:scratch_local=32gb -l walltime=24:00:00 -v MODEL=./backup_LMS/CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/perplexity_measure_helper.sh  

qsub -N MeasurePerplexity -q gpu -l select=1:ncpus=1:ngpus=1:mem=40gb:gpu_mem=12gb:scratch_local=32gb -l walltime=24:00:00 -v MODEL=./backup_LMS/CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/perplexity_measure_helper.sh 
qsub -N MeasurePerplexity -q gpu -l select=1:ncpus=1:ngpus=1:mem=40gb:gpu_mem=12gb:scratch_local=32gb -l walltime=24:00:00 -v MODEL=./backup_LMS/CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/perplexity_measure_helper.sh  

qsub -N MeasurePerplexity -q gpu -l select=1:ncpus=1:ngpus=1:mem=40gb:gpu_mem=12gb:scratch_local=32gb -l walltime=24:00:00 -v MODEL=./backup_LMS/CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM shell_scripts/perplexity_measure_helper.sh  
qsub -N MeasurePerplexity -q gpu -l select=1:ncpus=1:ngpus=1:mem=40gb:gpu_mem=12gb:scratch_local=32gb -l walltime=24:00:00 -v MODEL=./backup_LMS/CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM shell_scripts/perplexity_measure_helper.sh

qsub -N MeasurePerplexity -q gpu -l select=1:ncpus=1:ngpus=1:mem=40gb:gpu_mem=12gb:scratch_local=32gb -l walltime=24:00:00 -v MODEL=./backup_LMS/CZ-VerseMarks-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM  shell_scripts/perplexity_measure_helper.sh  
qsub -N MeasurePerplexity -q gpu -l select=1:ncpus=1:ngpus=1:mem=40gb:gpu_mem=12gb:scratch_local=32gb -l walltime=24:00:00 -v MODEL=./backup_LMS/CZ-VerseMarks-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM  shell_scripts/perplexity_measure_helper.sh