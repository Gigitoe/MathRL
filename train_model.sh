#!/bin/bash

MODEL_PATH=models/Qwen3-8B-Base

current_date=$(date +%Y-%m-%d)

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/math_train.parquet \
    data.val_files=[data/math500.parquet,data/aime24_avg10.parquet,data/aime25_avg10.parquet,data/amc23_avg10.parquet,data/minerva_math.parquet,data/olympiad_bench.parquet] \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='MathRL' \
    trainer.experiment_name=${MODEL_PATH}_${current_date} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=1000 $@
