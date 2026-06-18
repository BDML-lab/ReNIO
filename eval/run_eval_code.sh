#!/bin/bash

BASE_MODEL=${BASE_MODEL:-"PATH_TO_BASE_MODEL"}
CHECKPOINT=${CHECKPOINT:-""}
DATASET=${DATASET:-"humaneval"}
VALN=${VALN:-1}
CUVIS=${CUVIS:-0}
NGPU=${NGPU:-4}

CMD="NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=\"$CUVIS\" python eval/evaluate_code.py \
    --base_model \"$BASE_MODEL\" \
    --dataset \"$DATASET\" \
    --val_n $VALN \
    --temperature 1.0 \
    --tensor_parallel_size $NGPU"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint_dir \"$CHECKPOINT\""
fi

echo "Running: $CMD"
eval $CMD
wait
