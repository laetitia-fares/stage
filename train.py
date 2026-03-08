import os
import torch
import torch.distributed as dist

# Megatron Core
from megatron.core import parallel_state

# Megatron Bridge
from megatron.bridge import AutoBridge
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.config import (
    ConfigContainer,
    TrainingConfig,
    CheckpointConfig,
    SchedulerConfig,
    FinetuningDatasetConfig,
)
from megatron.core.optimizer import OptimizerConfig

# HF utils
from transformers import AutoTokenizer
from megatron.bridge.models.hf_pretrained.utils import is_safe_repo
from megatron.bridge.utils.common_utils import print_rank_0


def main():

    # 1. Distributed initialization (torchrun compatible)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")

    print_rank_0("Distributed environment initialized")

    # 2. Base Hugging Face model (foundation model)
    HF_MODEL_ID = "meta-llama/Llama-3.2-1B"

    print_rank_0(f" Loading base model: {HF_MODEL_ID}")

    bridge = AutoBridge.from_hf_pretrained(
        HF_MODEL_ID,
        trust_remote_code=is_safe_repo(
            trust_remote_code=False,
            hf_path=HF_MODEL_ID,
        ),
    )

    # 3. Megatron model provider (Bridge owns lifecycle)
    model_provider = bridge.to_megatron_provider(load_weights=True)

    # Parallelism configuration (safe & scalable)
    model_provider.tensor_model_parallel_size = 1
    model_provider.pipeline_model_parallel_size = 1
    model_provider.pipeline_dtype = torch.bfloat16

    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=1234)

    print_rank_0(" Megatron model provider ready")

   
    # 4. Tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID,
        trust_remote_code=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. LoRA (PEFT) configuration 
    peft_lora = LoRA(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    print_rank_0(" LoRA PEFT configured")

    # 6. Dataset configuration (REAL, LARGE, TOPIC-SPECIFIC)
    dataset_cfg = FinetuningDatasetConfig(
        dataset_name="microsoft/orca-math-word-problems-200k",
        split="train",
        text_column="question",   # dataset already contains clean text fields
        max_seq_length=1024,
    )

    print_rank_0(" Dataset configured: Orca Math Word Problems")

    # 7. Training configuration (REAL RUN, not demo)
    training_cfg = TrainingConfig(
        micro_batch_size=1,
        global_batch_size=32,
        train_iters=20000,          #  fine-tuning run
        log_interval=20,
        eval_interval=0,
        save_interval=2000,
    )

    
    # 8. Optimizer & LR scheduler
    optimizer_cfg = OptimizerConfig(
        optimizer="adam",
        lr=2e-4,
        weight_decay=0.01,
    )

    scheduler_cfg = SchedulerConfig(
        lr_warmup_steps=500,
        lr_decay_steps=20000,
        lr_decay_style="cosine",
    )

    # 9. Checkpointing (enabled, professional)
    checkpoint_cfg = CheckpointConfig(
        save=True,
        save_interval=2000,
        checkpoint_dir="./checkpoints",
    )

    
    # 10. Global training configuration container
    config = ConfigContainer(
        training=training_cfg,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        checkpoint=checkpoint_cfg,
        finetuning_dataset=dataset_cfg,
        peft=peft_lora,
    )

    print_rank_0(" Starting LoRA fine-tuning with Megatron Bridge")

    # 11. Launch training (Bridge controls everything)
    bridge.train(
        tokenizer=tokenizer,
        config=config,
    )

    print_rank_0(" Training completed successfully")
    # 12. Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
