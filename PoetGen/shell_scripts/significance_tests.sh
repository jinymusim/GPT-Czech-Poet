# Signif Test for Input Format

#qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./gpt-cz-poetry-basic-format-e4e8_LM, IMPROVED=./gpt-cz-poetry-basic-format-e0e4-pretrained_LM, BASEGEN=BASIC, IMPROVEDGEN=BASIC, BASEINPUT=BASIC, IMPROVEDINPUT=BASIC'  shell_scripts/significance_test_helper.sh
#qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./Base-Tokenizer-gpt-cz-poetry-verse-param-format-e4e8_LM, IMPROVED=./Base-Tokenizer-gpt-cz-poetry-verse-param-format-e0e4-pretrained_LM, BASEGEN=BASIC, IMPROVEDGEN=BASIC, BASEINPUT=VERSE_PAR, IMPROVEDINPUT=VERSE_PAR'  shell_scripts/significance_test_helper.sh
#qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./gpt-cz-poetry-basic-format-e4e8_LM, IMPROVED=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=BASIC, BASEINPUT=BASIC, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
#qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./Base-Tokenizer-gpt-cz-poetry-verse-param-format-e4e8_LM, IMPROVED=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=BASIC, BASEINPUT=VERSE_PAR, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh

# Signif Test for FORCED

qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, IMPROVED=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=FORCED, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM, IMPROVED=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=FORCED, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM, IMPROVED=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=FORCED, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, IMPROVED=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=FORCED, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh

# Signif Test for Tokenizer

qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, IMPROVED=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=FORCED, IMPROVEDGEN=FORCED, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, IMPROVED=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=FORCED, IMPROVEDGEN=FORCED, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, IMPROVED=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=FORCED, IMPROVEDGEN=FORCED, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh

qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM, IMPROVED=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=FORCED, IMPROVEDGEN=FORCED, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM, IMPROVED=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=FORCED, IMPROVEDGEN=FORCED, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh

qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM, IMPROVED=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=FORCED, IMPROVEDGEN=FORCED, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh

# Signif Test Secondary

qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM, IMPROVED=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=BASIC, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM, IMPROVED=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=BASIC, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM, IMPROVED=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=BASIC, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM, IMPROVED=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=BASIC, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh

# Signif Test Language Learning 

qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e0e24_LM, IMPROVED=./CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=BASIC, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e0e24_LM, IMPROVED=./CZ-New-Processed-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=BASIC, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e0e24_LM, IMPROVED=./CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=BASIC, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh
qsub -N TestSignif -q gpu -l select=1:ncpus=4:ngpus=1:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'BASE=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e0e24_LM, IMPROVED=./CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM, BASEGEN=BASIC, IMPROVEDGEN=BASIC, BASEINPUT=METER_VERSE, IMPROVEDINPUT=METER_VERSE'  shell_scripts/significance_test_helper.sh

### Validators Test ###

# Signif Test Difference RHYME

# Signif Test Difference METER

# Signif Test Difference YEAR

# Signif Test Syllable RHYME

# Signif Test Syllable METER

# Signif Test Syllable YEAR