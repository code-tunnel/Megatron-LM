#!/bin/bash
# Copied from examples/gpt3/train_gpt3_175b_distributed.sh
# - DISTRIBUTED_ARGS are excluded to adapt to different launchers
# - GPT_MODEL_ARGS and MODEL_PARALLEL_ARGS are excluded to adapt to different experiment configurations

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# GPUS_PER_NODE=8
# Change for multinode config
# MASTER_ADDR=localhost
# MASTER_PORT=6000
# NUM_NODES=1
# NODE_RANK=0
# WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# CHECKPOINT_PATH=$1 #<Specify path>
# TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=examples/gpt3/gpt2tokenizer/gpt2-vocab.json #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=examples/gpt3/gpt2tokenizer/gpt2-merges.txt #<Specify path to file>/gpt2-merges.txt
# DATA_PATH=$5 #<Specify path and file prefix>_text_document

# DISTRIBUTED_ARGS=(
#     --nproc_per_node $GPUS_PER_NODE 
#     --nnodes $NUM_NODES 
#     --master_addr $MASTER_ADDR 
#     --master_port $MASTER_PORT
# )

# GPT_MODEL_ARGS=(
#     --num-layers 96 
#     --hidden-size 12288 
#     --num-attention-heads 96 
#     --seq-length 2048 
#     --max-position-embeddings 2048 
# )

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BS 
    --global-batch-size $GLOBAL_BS 
    # --rampup-batch-size 16 16 5859375 
    --train-iters 30 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

# MODEL_PARALLEL_ARGS=(
# 	--tensor-model-parallel-size 8 
# 	--pipeline-model-parallel-size 16 
# )

DATA_ARGS=(
    # --data-path $DATA_PATH 
    --mock-data
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    # --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    # --save-interval 10000 
    # --eval-interval 1000 
    # --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 0
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
#     ${GPT_MODEL_ARGS[@]} \
#     ${TRAINING_ARGS[@]} \
#     ${MODEL_PARALLEL_ARGS[@]} \
#     ${DATA_ARGS[@]} \
#     ${EVAL_AND_LOGGING_ARGS[@]}
