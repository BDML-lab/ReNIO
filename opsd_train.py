import os
import subprocess
import wandb

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opsd_trainer import OPSDTrainer
from dataclasses import dataclass, field

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    teacher_path: str = field(
        default=None,
        metadata={
            "help": "Path or name of the teacher model. If specified, will use a separate teacher model "
            "instead of self-distillation. The teacher model is kept frozen during training."
        },
    )
    teacher_lora_path: str = field(
        default=None,
        metadata={
            "help": "Path to a LoRA adapter for the teacher model. When specified, loads the base model "
            "from teacher_path and applies the LoRA adapter on top. The merged model is then frozen. "
            "This is more memory-efficient than loading a full separate teacher model. "
            "Requires teacher_path to be set (as the base model path)."
        },
    )
    kd_type: str = field(
        default="SFT",
        metadata={
            "help": "model training type: SFT, Distillm, KD, GKD, OPSD, etc."
        }
    )
    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates. "
            "Not compatible with teacher_path."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    dataset_path: str = field(
        default="data/openthoughts_math_30k",
        metadata={
            "help": "Path to local dataset. Can be a JSON file, JSONL file, or directory containing Arrow/Parquet files. "
        },
    )
    task_type: str = field(
        default="math",
        metadata={
            "help": "Task type for prompt template selection. "
            "Options: 'math' (math reasoning with boxed answers), 'coding' (code generation). "
            "Default: 'math'"
        },
    )
    use_renio: bool = field(
        default=False,
        metadata={
            "help": "Enable ReNIO sample weighting (fixed-threshold S/T log-ratio filtering). "
            "When disabled, uniform sample weights are used."
        },
    )
    imp_token_threshold: float = field(
        default=0.2,
        metadata={
            "help": "Important token threshold (percentage). "
            "Used to select top-k% tokens for computing sample weights. "
            "Default: 0.3 (top 30% tokens)"
        },
    )
    kd_clamp: float = field(
        default=1.0,
        metadata={
            "help": "Clamp value for log-ratio. "
            "Log-ratio values will be clamped to this maximum value. "
            "Default: 2.0"
        },
    )
    weight_norm_type: str = field(
        default="batch_mean",
        metadata={
            "help": "Weight normalization type. "
            "Options: 'batch_mean' (normalize per batch), "
            "'ema' (normalize using global EMA statistics), "
            "'none' (no normalization), "
            "'clamp' (clamp then normalize). "
            "Default: 'batch_mean'"
        },
    )
    kd_sgo_tem: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for SGO weight computation. "
            "Used to soften the log-ratio before exp. "
            "Default: 1.0"
        },
    )


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    
    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )
    
    
    model_name = model_args.model_name_or_path.split("/")[-1]

    # Create concise run name
    full_wandb_run_config = (
        script_args.dataset_path.split("/")[-1] + "_"
        f"{model_name}_"
        f"renio{script_args.use_renio}"
        f"type{script_args.kd_type}_"
        f"lr{lr_str}_"
        f"bs{effective_batch_size}_"
        f"tok{training_args.max_completion_length}"
        f"clip{script_args.kd_clamp}_"
        f"tem{script_args.kd_sgo_tem}_"
        f"thresh{script_args.imp_token_threshold}"
        f"beta{training_args.beta}"
    )

    # Add fixed_teacher to wandb name if enabled
    if script_args.fixed_teacher:
        full_wandb_run_config += "_fixteach"
    
    from pathlib import Path

    training_args.output_dir = str(Path(training_args.output_dir) / model_name / script_args.dataset_path.split("/")[-1] / script_args.kd_type / script_args.run_config / f"renio_{script_args.use_renio}_clip_{script_args.kd_clamp}_tem_{script_args.kd_sgo_tem}_thr_{script_args.imp_token_threshold}")


    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Validate fixed_teacher and teacher_path are mutually exclusive
    if script_args.fixed_teacher and script_args.teacher_path is not None:
        raise ValueError(
            "fixed_teacher=True and teacher_path are mutually exclusive. "
            "Use either a separate teacher model or fixed teacher (self-distillation), not both."
        )

    # Validate fixed_teacher argument
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    # Validate teacher_lora_path requires teacher_path
    if script_args.teacher_lora_path is not None and script_args.teacher_path is None:
        raise ValueError(
            "teacher_lora_path requires teacher_path to specify the base model."
        )

    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "student_model_name": model_args.model_name_or_path,
                "teacher_model_name": script_args.teacher_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
                "use_renio": script_args.use_renio,
                "imp_token_threshold": script_args.imp_token_threshold,
                "kd_clamp": script_args.kd_clamp,
                "weight_norm_type": script_args.weight_norm_type,
                "kd_sgo_tem": script_args.kd_sgo_tem,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Load Teacher Model (if specified)
    ################
    teacher_model = None
    if script_args.teacher_path is not None:
        print(f"\n{'='*80}")
        print(f"Loading teacher model from: {script_args.teacher_path}")
        if script_args.teacher_lora_path:
            print(f"  with LoRA adapter: {script_args.teacher_lora_path}")
        print(f"{'='*80}\n")

        from transformers import AutoModelForCausalLM

        # Use same dtype as student model
        teacher_dtype = model_dtype

        # Prepare teacher model kwargs
        teacher_model_kwargs = dict(
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation or "flash_attention_2",
            torch_dtype=teacher_dtype,
            use_cache=False,  # Disable cache for teacher during training
        )

        # Load base model
        teacher_model = AutoModelForCausalLM.from_pretrained(
            script_args.teacher_path,
            **teacher_model_kwargs,
        )

        # Apply LoRA adapter if specified
        if script_args.teacher_lora_path is not None:
            from peft import PeftModel

            print(f"Applying LoRA adapter from: {script_args.teacher_lora_path}")
            teacher_model = PeftModel.from_pretrained(
                teacher_model, script_args.teacher_lora_path
            )
            # Merge LoRA into base weights to avoid adapter overhead during forward
            teacher_model = teacher_model.merge_and_unload()
            print(f"LoRA adapter merged into base model")

        # Disable dropout in teacher model
        if training_args.disable_dropout:
            from trl.trainer.utils import disable_dropout_in_model
            disable_dropout_in_model(teacher_model)

        print(f"\n{'='*80}")
        print(f"Teacher model loaded successfully")
        print(f"  Base model: {script_args.teacher_path}")
        if script_args.teacher_lora_path:
            print(f"  LoRA adapter: {script_args.teacher_lora_path} (merged)")
        print(f"  Parameters: {teacher_model.num_parameters():,}")
        print(f"  Dtype: {teacher_dtype}")
        print(f"{'='*80}\n")

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    training_args.presence_penalty = script_args.presence_penalty

    # Skip SFTTrainer's default dataset tokenization — OPSD uses its own data collator
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset from local path or HuggingFace hub
    if script_args.dataset_path:
        print(f"\n{'='*80}")
        print(f"Loading dataset from local path: {script_args.dataset_path}")
        print(f"{'='*80}\n")
        if script_args.dataset_path.endswith(('.json', '.jsonl')):
            dataset = load_dataset(
                "json",
                data_files=script_args.dataset_path,
                split="train"
            )
        else:
            import pandas as pd
            
            if os.path.isdir(script_args.dataset_path):
                parquet_files = [os.path.join(script_args.dataset_path, f) 
                                for f in os.listdir(script_args.dataset_path) 
                                if f.endswith('.parquet')]
                print(f"Found {len(parquet_files)} parquet files")
                dfs = [pd.read_parquet(f) for f in parquet_files]
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = pd.read_parquet(script_args.dataset_path)
            
            dataset = Dataset.from_pandas(df)
            print(f"Loaded {len(dataset)} examples from parquet files")
        
    else:
        print(f"\n{'='*80}")
        print(f"{'='*80}\n")
        dataset = load_dataset("PATH_TO_DATASET_URL")
        dataset = dataset["train"]

    train_dataset = dataset if isinstance(dataset, Dataset) else dataset["train"]


    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        teacher_model=teacher_model,
        use_renio=script_args.use_renio,
        imp_token_threshold=script_args.imp_token_threshold,
        kd_clamp=script_args.kd_clamp,
        weight_norm_type=script_args.weight_norm_type,
        kd_sgo_tem=script_args.kd_sgo_tem,
        task_type=script_args.task_type,
        # dataset_kwargs={"skip_prepare_dataset": True},  # Skip dataset preparation
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)

