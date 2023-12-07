qsub -N GPTE0E4BasicFormat -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=0,  EPOCHSPOET=4,  TOKENIZER=/auto/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM,  MODELTYPE=base,  MODEL=./gpt-cz-poetry-basic-format-e0e4, HFMODEL=/storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM, FORMAT=BASIC'  format_model_train_helper.sh
qsub -N BASEE0E4VerseParFormat -q gpu -l select=1:ncpus=4:ngpus=2:mem=40gb:gpu_mem=30gb:scratch_local=64gb -l walltime=24:00:00 -v 'EPOCHSLM=0,  EPOCHSPOET=4, TOKENIZER=/auto/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM,  MODELTYPE=base,  MODEL=./Base-Tokenizer-gpt-cz-poetry-verse-param-format-e0e4, HFMODEL=/storage/brno2/home/chudobm/tf_shorts/Tensorflow-Shorts/PoetGen/CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-base-e4e8_LM, FORMAT=VERSE_PAR'  format_model_train_helper.sh