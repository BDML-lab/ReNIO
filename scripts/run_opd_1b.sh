GRAD=${GRAD:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-5e-6}
PORT=${PORT:-12949}
CLIP=${CLIP:-2.0}
SGO_TEM=${SGO_TEM:-1.0}
IMP=${IMP:-1.0}
NGPU=${NGPU:-4}
BETA=${BETA:-0}
DATA=${DATA:-"data/openthoughts_math_30k"}
TASK=${TASK:-"math"}
TEACHER=${TEACHER:-PATH_TO_TEACHER_MODEL}
MAX_LENGTH=${MAX_LENGTH:-1024}
WANDB_PRO=${WANDB_PRO:-"OPD"}
CUVIS=${CUVIS:-0,1,2,3}
WEIGHT_NORM=${WEIGHT_NORM:-"batch_mean"}
RENIO=${RENIO:-False}

export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES="$CUVIS" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NGPU" \
    --gradient_accumulation_steps "$GRAD" \
    --main_process_port "$PORT" \
    opsd_train.py \
    --model_name_or_path PATH_TO_BASE_MODEL \
    --teacher_path "$TEACHER" \
    --learning_rate "$LR" \
    --dataset_path "$DATA" \
    --task_type "$TASK" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRAD" \
    --output_dir  results/opd \
    --run_config qwen31b_gen"$MAX_LENGTH"_fixteacher_temp11_lr"$LR"_beta"$BETA"_norm"$WEIGHT_NORM" \
    --num_train_epochs 1 \
    --max_completion_length "$MAX_LENGTH" \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta "$BETA" \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --jsd_token_clip 0.05 \
    --wandb_project "$WANDB_PRO" \
    --ratio_type "$RATIO_TYPE" \
    --kd_type "OPD" \
    --kd_clamp "$CLIP" \
    --kd_sgo_tem "$SGO_TEM" \
    --imp_token_threshold "$IMP" \
    --weight_norm_type "$WEIGHT_NORM" \
    --enable_signal_analysis \
    --use_renio "$RENIO" \
    --max_steps 126 \
    ${TEACHER_LORA:+--teacher_lora_path "$TEACHER_LORA"}
