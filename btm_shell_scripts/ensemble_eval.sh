# Number of GPUs you'd like to evaluate on. Set this equal to number of experts you'd like to mix.
NUM_GPUS=$1
# Path to preprocessed data
DATA_PATH=$2
# Semi-colon separated list of model files
MODELS=$3
# Target data domain to evaluate on
DOMAIN=$4
# Ensemble type, one of "simple_average","cached_prior", "updating_prior", "uniform_prior"
ENSEMBLE_TYPE=$5
# Folder for saving results
RESULTS_OUTPUT_FOLDER=$6

mkdir -p ${RESULTS_OUTPUT_FOLDER}
prior_results_path=${RESULTS_OUTPUT_FOLDER}/dev_posteriors.jsonl;
results_path=${RESULTS_OUTPUT_FOLDER}/test_results.txt;

target_eval_split=valid_${DOMAIN};

python -u fairseq/btm_fairseq_cli/ensemble_eval_lm.py $DATA_PATH \
--path $MODELS \
--gen-subset $target_eval_split \
--target-domain train_${DOMAIN} \
--target-eval ${target_eval_split} \
--task multidomain_language_modeling \
--sample-break-mode none \
--tokens-per-sample 1024      \
--batch-size 2  \
--sample-break-mode none     \
--log-format simple     \
--log-interval 50     \
--skip-invalid-size-inputs-valid-test               \
--no-save           \
--bucket-cap-mb 200                       \
--ddp-backend no_c10d      \
--arch transformer_lm                 \
--train-domains ${DOMAIN} \
--eval-domains ${DOMAIN} \
--log-format tqdm \
--train-subset train_${DOMAIN} \
--partial-load \
--ensemble-type "updating_prior" \
--results-path ${prior_results_path} \
--max-samples 100 \
--distributed-world-size $num_gpus \
--distributed-port 12345;

precomputed_prior=$(tail -n 1 ${prior_results_path} | jq -rc '.exp_avg_posterior | join(",")');
target_eval_split=test_${DOMAIN};

 python -u btm_fairseq_cli/ensemble_eval_lm.py $DATA_PATH \
--path $MODELS \
--gen-subset $target_eval_split \
--target-domain train_${DOMAIN} \
--target-eval ${target_eval_split} \
--task multidomain_language_modeling \
--sample-break-mode none \
--tokens-per-sample 1024      \
--batch-size 2  \
--sample-break-mode none     \
--log-format simple     \
--log-interval 50     \
--skip-invalid-size-inputs-valid-test               \
--no-save           \
--bucket-cap-mb 200                       \
--ddp-backend no_c10d      \
--arch transformer_lm                 \
--train-domains ${DOMAIN} \
--eval-domains ${DOMAIN} \
--log-format tqdm \
--train-subset train_${DOMAIN} \
--partial-load \
--results-path ${results_path} \
--ensemble-type ${ENSEMBLE_TYPE} \
--precomputed-prior ${precomputed_prior} \
--eval-topk ${eval_top_k} \
--distributed-world-size $num_gpus \
--distributed-port 12345 ;
