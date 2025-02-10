export DEBUG_MODE="false" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b_$(date +%Y-%m-%d-%H-%M).txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir "/mnt/jfs-test/checkpoints/par/duidian/r1v_acc-reward" \
    --model_name_or_path "/mnt/jfs-test/models/Qwen2-VL-2B-Instruct" \
    --dataset_name "/mnt/jfs-test/data/clevr_cogen_a_train" \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true \
    --temperature 1.0 \
    --trainable_parts "full" \
    --num_generations 8 \
    --max_steps 100 \
    --reward_funcs "acc" "format"