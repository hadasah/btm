# Path to preprocessed data
DATA_PATH=$1
# Folder of model for evaluation
MODEL_FOLDER=$2
# model file name
CHECKPOINT_FILE_NAME=$3
# split you'd like to evaluate on ("valid" or "test")
DATA_SPLIT=$4
# target domain to evaluate on
DATA_DOMAIN_NAME=$5

model_file=${MODEL_FOLDER}/${CHECKPOINT_FILE_NAME};
results_folder=${MODEL_FOLDER}/evals/${DATA_DOMAIN_NAME}
results_path=${results_folder}/test_results.txt
mkdir -p results_folder;

python -u fairseq/fairseq_cli/eval_lm.py \
${DATA_PATH} \
--path ${model_file} \
--gen-subset ${DATA_SPLIT}_${DATA_DOMAIN_NAME} \
--task multidomain_language_modeling \
--sample-break-mode none \
--tokens-per-sample 1024     \
--batch-size 2  \
--eval-domains ${DATA_DOMAIN_NAME} \
--results-path ${results_path} \
--partial-load;