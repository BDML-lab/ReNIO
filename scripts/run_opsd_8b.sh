GRAD=${GRAD:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-5e-6}
PORT=${PORT:-12949}
CLIP=${CLIP:-2.0}
SGO_TEM=${SGO_TEM:-1.0}
IMP=${IMP:-0.01}
NGPU=${NGPU:-4}
BETA=${BETA:-0}
DATA=${DATA:-"data/openthoughts_math_30k"}
TASK=${TASK:-"math"}
MAX_LENGTH=${MAX_LENGTH:-1024}
CUVIS=${CUVIS:-0,1,2,3}
WANDB_PRO=${WANDB_PRO:-"OPSD"}
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
    --dataset_path "$DATA" \
    --task_type "$TASK" \
    --learning_rate "$LR" \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GRAD" \
    --output_dir  results/opsd \
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
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.06 \
    --wandb_project OPSD \
    --kd_type "OPSD" \
    --kd_clamp "$CLIP" \
    --kd_sgo_tem "$SGO_TEM" \
    --imp_token_threshold "$IMP" \
    --weight_norm_type "$WEIGHT_NORM" \
    --use_renio "$RENIO" \
    --max_steps 126
