#!/bin/bash

# =============== Note ======================
# this script sets some common env variables and model hyperparameters

# =============== env params ================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =============== model params ================
export GPT_ARGS="
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --sequence-parallel \
    --num-layers $NUM_LAYER \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_HEAD \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size ${MICRO_BS} \
    --global-batch-size ${GLOBAL_BS} \
    --lr 0.00015 \
    --train-iters 30 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --use-flash-attn \
    --no-attention-mask \
    --parallel-position-embedding "

VOCAB_FILE="scripts/gpt2tokenizer/gpt2-vocab.json"
MERGE_FILE="scripts/gpt2tokenizer/gpt2-merges.txt"

export DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --synthesize-dataloader "

export OUTPUT_ARGS="
    --log-interval 10 "
    
    # --dump-memory-snapshot \
    # --memory-snapshot-prefix offload_snapshot

    # --profile \
    # --profile-prefix tb_log

    # --tensorboard-dir ./tb_log \
    # --tensorboard-log-interval 1\
    # --log-timers-to-tensorboard \
    # --log-memory-to-tensorboard

export LAUNCH_1F1B_CMD="
    pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --recompute-activations "
    
export LAUNCH_HELIX_CMD="
    pretrain_gpt.py \
        $GPT_ARGS \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        --attention-pipeline \
        --num-fold $NUM_FOLD \
        --num-layers-per-virtual-pipeline-stage 1 \
        --transfer-weight \
        --checkpoint-without-attn "
