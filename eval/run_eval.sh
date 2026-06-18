#!/bin/bash

BASE_MODEL=${BASE_MODEL:-"PATH_TO_BASE_MODEL"}
CHECKPOINT=${CHECKPOINT:-"PATH_TO_CHECKPOINT"}
DATASET=${DATASET:-"aime24"}
VALN=${VALN:-1}
CUVIS=${CUVIS:-0}
NGPU=${NGPU:-4}
GPU_MEM=${GPU_MEM:-0.9}

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$CUVIS" python eval/evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "$DATASET" \
    --val_n "$VALN" \
    --temperature 1.0 \
    --tensor_parallel_size "$NGPU" \
    --checkpoint_dir "$CHECKPOINT" \
    --gpu_memory_utilization "$GPU_MEM"
wait
