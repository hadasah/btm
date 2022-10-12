SWEEP_NAME=$1
# Number of GPUs you'd like to train on
NUM_GPUS=$2
# Number of nodes you'd like to train on (assuming 8 GPUs per node)
NUM_NODES=$((${NUM_GPUS}/8))
# Fairseq model name (e.g. transformer_lm; see fairseq/models/transformer_lm.py for other options)
ARCH=$3
# Baseline type: choice between dense, unbalanced_dense, and demix
EXPERIMENT=$4
# Path to data-bins
DATA_PATH=$5
# Comma separated list of domains in the data path, e.g. '1b,reddit' "
DOMAINS=$6;
# Old directory to copy checkpoints from -- can be "None" if training from scratch
INIT_CHECKPOINT_DIR=$7
# path to top-level directory to where you'd like to output the model
SERIALIZATION_DIR=$8
# Name of subdirectory containing checkpoint to copy
SUBFOLDER_NAME=$9
# Ratio of updates to spend in first phase training - "None" or a float, e.g. 0.5
SEED_PHASE_RATIO=${10}
# Number of updates to spend in first phase training - "None" or an int, e.g. 36000
SEED_PHASE_UPDATE_NUM=${11}
# comma separated list of items to reset in checkpoint (dataloader,meters,lr-scheduler,optimizer), or "None"
RESET_DATALOADER=${12};
# total number of steps in training -- determines lr schedule
NUM_STEPS=${13};
# update frequency
UPDATE_FREQ=${14};
# learning rate
LR=${15};
# save internal updates
SAVE_INTERVAL_UPDATES=${16}
# port for distributed comms
DISTRIBUTED_PORT=${17}
# name of wandb project to track model output (at wandb.ai)
WANDB_PROJECT=${18};
# path to fairseq code folder
BTM_FOLDER=${19};
# random seed
SEED=${20};
# Unique identifer of this run
RUN_ID=${21}

export WANDB_NAME=$SWEEP_NAME/$RUN_ID;
FAIRSEQ_FOLDER=$BTM_FOLDER/fairseq;
DATA_PHRASE="";
OIFS=$IFS;
IFS=','
read -a domain_names <<< "$DOMAINS";
IFS=$OIFS;

if [[ ${#domain_names[@]} > 1 ]]; then
     domains="";
     valid_domains="";
     for id in "${domain_names[@]}"; do
          domains="${domains},$id"
          valid_domains="${valid_domains},valid_$id"
     done;
     
     DATA_PHRASE="$DATA_PATH \
          --task multidomain_language_modeling \
          --valid-subset ${valid_domains#?} \
          --train-domains ${domains#?}  \
          --eval-domains ${domains#?} \
          --criterion desynchronized_cross_entropy     \
          "
else
     id=${domain_names[0]}
     valid_domains="valid_$id";
     DATA_PHRASE="${DATA_PATH}/${id} \
          --task language_modeling \
          --valid-subset ${valid_domains} \
          --criterion cross_entropy     \
          ";
fi;
echo $DATA_PHRASE;

TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
KEEP_INTERVAL_UPDATES=-1;
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
if [[ $INIT_CHECKPOINT_DIR != "None" ]]; then
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
     read -r did_copy < <(python -u $BTM_FOLDER/btm_utils/branching_checkpoint_utils.py \
          --old-folder $INIT_CHECKPOINT_DIR \
          --new-folder $SERIALIZATION_DIR \
          --subfolder $SUBFOLDER_NAME \
          $NEW_SUBFOLDER_PHRASE \
          $PHASE_PHRASE);
     if [[ $did_copy == "False" ]]; then
          RESET_PHRASE='';
     fi;
else 
     SERIALIZATION_DIR=$SERIALIZATION_DIR/$SWEEP_NAME
fi;

EXP_PHRASE="";
if [[ $EXPERIMENT == *"unbalanced"* ]]; then
     EXP_PHRASE=" --unbalanced ";
elif [[ $EXPEIRMENT == *"demix"* ]]; then
     # $DATA_PARALLEL_GROUPS identifies which ranks we will synchronize over. "A,B C,D" means we will synchronize ranks A,B and synchronize ranks C,D.
     if [[ $NUM_GPUS == "8" ]]; then
          if  [[ $EXPERIMENT == *"demix"* ]]; then
               DATA_PARALLEL_GROUPS="0 1 2 3 4 5 6 7";
          fi;
     elif [[ $NUM_GPUS == "16" ]]; then
          if  [[ $EXPERIMENT == *"demix"* ]]; then
               DATA_PARALLEL_GROUPS="0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15";
          fi;
     elif [[ $NUM_GPUS == "32" ]]; then
          if [[ $EXPERIMENT == *"demix"* ]]; then
               DATA_PARALLEL_GROUPS="0,1,2,3 4,5,6,7 8,9,10,11 12,13,14,15 16,17,18,19 20,21,22,23 24,25,26,27 28,29,30,31";
          fi;
     elif [[ $NUM_GPUS == "64" ]]; then
          if [[ $EXPERIMENT == *"demix"* ]]; then
               DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7 8,9,10,11,12,13,14,15 16,17,18,19,20,21,22,23 24,25,26,27,28,29,30,31 32,33,34,35,36,37,38,39 40,41,42,43,44,45,46,47 48,49,50,51,52,53,54,55 56,57,58,59,60,61,62,63";
          fi;
     elif [[ $NUM_GPUS == "128" ]]; then
          if [[ $EXPERIMENT == *"demix"* ]]; then
               DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7 8,9,10,11,12,13,14,15 16,17,18,19,20,21,22,23 24,25,26,27,28,29,30,31 32,33,34,35,36,37,38,39 40,41,42,43,44,45,46,47 48,49,50,51,52,53,54,55 56,57,58,59,60,61,62,63 64,65,66,67,68,69,70,71 72,73,74,75,76,77,78,79 80,81,82,83,84,85,86,87 88,89,90,91,92,93,94,95 96,97,98,99,100,101,102,103 104,105,106,107,108,109,110,111 112,113,114,115,116,117,118,119 120,121,122,123,124,125,126,127";
          fi;
     fi;
     EXP_PHRASE=" --desynchronize --domain-parallel \
          --sync-type manual \
          --untie-parameters feedforward \
          --data-parallel-groups "${DATA_PARALLEL_GROUPS}" \
          --pad-to-fixed-length ";
fi;

python -u $FAIRSEQ_FOLDER/fairseq_cli/train.py $DATA_PHRASE \
     --arch $ARCH    \
     --sample-break-mode none \
     --lr-scheduler polynomial_decay     \
     --lr $LR              \
     --max-update $NUM_STEPS     \
     --total-num-update $NUM_STEPS     \
     --warmup-updates $NUM_WARMUP_STEPS     \
     --update-freq $UPDATE_FREQ     \
     --skip-invalid-size-inputs-valid-test     \
     --validate-interval-updates $VALIDATION_INTERVAL     \
     --save-interval-updates $SAVE_INTERVAL_UPDATES     \
     --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
     --no-epoch-checkpoints \
     --log-format simple  \
     --log-interval $LOG_INTERVAL    \
     --num-workers 2 \
     --max-sentences $BATCH_SIZE \
     --max-sentences-valid $BATCH_SIZE \
     --tokens-per-sample $TOKENS_PER_SAMPLE          \
     --optimizer adam \
     --adam-betas '(0.9, 0.95)'  \
     --adam-eps 10e-8 \
     --weight-decay 0.1 \
     --clip-norm $CLIP_NORM      \
     --save-dir $SERIALIZATION_DIR/$RUN_ID/     \
     --batch-size-valid 2                        \
     --required-batch-size-multiple 1 \
     $DISTRIBUTED_ARGS_PHRASE \
     $FP_16_PHRASE \
     $RESET_PHRASE \
     --all-gather-list-size 32000 \
     --seed $SEED \
     $EXP_PHRASE \
     --wandb-project $WANDB_PROJECT;
