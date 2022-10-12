SWEEP_NAME=$1
# Number of GPUs you'd like to train on
NUM_GPUS=$2
# Number of nodes you'd like to train on (assuming 8 GPUs per node)
NUM_NODES=$((${NUM_GPUS}/8))
# Fairseq model name (e.g. transformer_lm; see https://github.com/kernelmachine/demix/blob/main/fairseq/models/transformer_lm.py for other options)
ARCH=$3
# Baseline type: choice between demix, dense, unbalanced_dense, and domain_token
EXPERIMENT=$4
# Path to data-bins
DATA_PATH=$5
# Name of domain to train on
DOMAIN=$6;
# Old directory to copy checkpoints from -- can be "None" if training from scratch
OLD_DIR=$8
# path to top-level directory to where you'd like to output the model
SERIALIZATION_DIR=$9
# Name of subdirectory containing checkpoint to copy
SUBFOLDER_NAME=${10}
# Ratio of updates to spend in first phase training - "None" or a float, e.g. 0.5
SEED_PHASE_RATIO=${12}
# Number of updates to spend in first phase training - "None" or an int, e.g. 36000
SEED_PHASE_UPDATE_NUM=${13}
# comma separated list of items to reset in checkpoint (dataloader,meters,lr-scheduler,optimizer), or "None"
RESET_DATALOADER=${14};
# total number of steps in training -- determines lr schedule
NUM_STEPS=${15};
# update frequency
UPDATE_FREQ=${16};
# learning rate
LR=${17};
# 
SAVE_INTERVAL_UPDATES=${18}
# port for distributed comms
DISTRIBUTED_PORT=${19}
# name of wandb project to track model output (at wandb.ai)
WANDB_PROJECT=${20};
# path to mod code folder
MOD_FOLDER=${22};
# random seed
SEED=${23}
# Unique identifer of this run
RUN_ID=${24}

export WANDB_NAME=$SWEEP_NAME/$RUN_ID;

domains=${DOMAIN};
train_subset=train;
valid_subset=valid_${DOMAIN};

# Set training hyperparams
TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
KEEP_INTERVAL_UPDATES=1;
CLIP_NORM=0.1;
NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));

if [[ $ARCH == *"gpt3_small"* ]]; then
     VALIDATION_INTERVAL=4000;
elif [[ $ARCH == *"gpt3_medium"* ]]; then
     VALIDATION_INTERVAL=2000;
elif [[ $ARCH == *"gpt3_large"* ]]; then
     VALIDATION_INTERVAL=1000;
elif [[ $ARCH == *"gpt3_xl"* ]]; then
     VALIDATION_INTERVAL=500;
elif [[ $ARCH == *"transformer_lm"* ]]; then
     VALIDATION_INTERVAL=6000;
fi;

RESET_PHRASE='';
if [[ $RESET_DATALOADER != "False" ]]; then
     RESET_PHRASE="--reset-dataloader "
fi;

# Add distributed training args if necessary
DISTRIBUTED_ARGS_PHRASE='';
if [ $NUM_GPUS \> 1 ]; then
     DISTRIBUTED_ARGS_PHRASE="--ddp-backend no_c10d --distributed-world-size $NUM_GPUS --distributed-port $DISTRIBUTED_PORT";
fi;

# Only use memory efficient fp16 for larger models
FP_16_PHRASE='--fp16 ';
if [[ $ARCH == *"gpt3_large"* ]] || [[ $ARCH == *"gpt3_xl"* ]]; then
     FP_16_PHRASE='--memory-efficient-fp16 ';
fi;

# Copying over the checkpoint
if [[ $OLD_DIR != "None" ]]; then
     NEW_SUBFOLDER_PHRASE='';
     if [[ $RUN_ID != "" ]]; then
          NEW_SUBFOLDER_PHRASE="--new-subfolder $RUN_ID ";
     fi;
     PHASE_PHRASE='';
     if [[ $PHASE_ONE_RATIO != "None" ]] && [[ $PHASE_ONE_UPDATE_NUM != "None" ]]; then
          printf '%s\n' "Cannot set both PHASE_ONE_RATIO and PHASE_ONE_UPDATE_NUM" >&2
          exit 1
     elif [[ $PHASE_ONE_RATIO != "None" ]]; then
          PHASE_PHRASE="--phase-one-ratio $PHASE_ONE_RATIO"
     elif [[ $PHASE_ONE_UPDATE_NUM != "None" ]]; then 
          PHASE_PHRASE="--phase-one-update-num $PHASE_ONE_UPDATE_NUM"
     else
          printf '%s\n' "If copying checkpoints, must set one of PHASE_ONE_RATIO or PHASE_ONE_UPDATE_NUM" >&2
          exit 1
     fi;
     read -r did_copy < <(python -u $MOD_FOLDER/mod_utils/mod_checkpoint_utils.py \
          --old-folder $OLD_DIR \
          --new-folder $SERIALIZATION_DIR \
          --subfolder $SUBFOLDER_NAME \
          $NEW_SUBFOLDER_PHRASE \
          $PHASE_PHRASE);
     if [[ $did_copy == "False" ]]; then
          RESET_PHRASE='';
     fi;
fi;

python -u $MOD_FOLDER/fairseq_cli/train.py  $DATA_PATH \
     --task multidomain_language_modeling \
     --train-subset $train_subset \
     --valid-subset $valid_subset \
     --train-domains $domains  \
     --eval-domains $domains \
     --arch $ARCH    \
     --sample-break-mode none \
     --criterion cross_entropy     \
     --lr-scheduler polynomial_decay     \
     --lr $LR              \
     --log-format simple  \
     --log-interval $LOG_INTERVAL    \
     --skip-invalid-size-inputs-valid-test     \
     --validate-interval-updates $VALIDATION_INTERVAL     \
     --save-interval-updates $SAVE_INTERVAL_UPDATES     \
     --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
     --no-epoch-checkpoints \
     --num-workers 2 \
     --max-sentences $BATCH_SIZE \
     --max-sentences-valid $BATCH_SIZE \
     --tokens-per-sample $TOKENS_PER_SAMPLE          \
     --optimizer adam \
     --adam-betas '(0.9, 0.95)'  \
     --adam-eps 10e-8 \
     --weight-decay 0.1 \
     --clip-norm $CLIP_NORM      \
     --max-update $NUM_STEPS     \
     --total-num-update $NUM_STEPS     \
     --update-freq $UPDATE_FREQ     \
     --save-dir $SERIALIZATION_DIR/$RUN_ID/   \
     --batch-size-valid 2            \
     --required-batch-size-multiple 1 \
     $DISTRIBUTED_ARGS_PHRASE \
     $FP_16_PHRASE \
     $RESET_PHRASE \
     --all-gather-list-size 32000 \
     --seed $SEED \
     --unbalanced \
     --wandb-project $WANDB_PROJECT;
