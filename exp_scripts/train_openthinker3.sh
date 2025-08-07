set -x
export HF_ENDPOINT=https://hf-mirror.com

eval "$(conda shell.bash hook)"
conda activate offline-grpo

#### NOTE: change to your root dir ####
ROOT="/home/ubuntu/offline-grpo"
#######################################
ray stop

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS

export MODEL_PATH=open-thoughts/OpenThinker3-7B
export DATA_DIR=$ROOT/data

#### Hyperparameters ####
export SAMPLING_N=8
export LR=1e-7
export KL_COEF=0.2
export TRAIN_FILE=OpenThoughts.parquet
##### Reward designs #####
export GRPO_STD=False
export ADD_ALL_POSITIVE=True
export CLIP_RANGE=null
#########################

REMOVE_CLIP=True
if [[ "$CLIP_RANGE" != "null" ]]; then REMOVE_CLIP=False; fi

train_base_name="${TRAIN_FILE%.parquet}"
ADV_PARAMS=""
if [[ "$GRPO_STD" == "True" ]]; then
    ADV_PARAMS="${ADV_PARAMS}-STD"
fi
if [[ "$ADD_ALL_POSITIVE" == "True" ]]; then
    ADV_PARAMS="${ADV_PARAMS}-AAP"
fi
if [[ "$CLIP_RANGE" != "null" ]]; then
    ADV_PARAMS="${ADV_PARAMS}-CLIP${CLIP_RANGE}"
fi
export EXP_NAME="${train_base_name}-n${SAMPLING_N}-lr${LR}-kl${KL_COEF}${ADV_PARAMS}"

if [[ "$KL_COEF" == "0.0" ]]; then
    use_kl_loss="False"
else
    use_kl_loss="True"
fi

export TENSORBOARD_PROJECT="offline-grpo"

cd $ROOT/src/verl/

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.mix_src.main_mix_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/$TRAIN_FILE \
    data.val_files=$DATA_DIR/valid.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=31744 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=$SAMPLING_N \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_prefix_len=31744 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="$TENSORBOARD_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.use_sft_prefix_reward=False \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.n_prefix=$SAMPLING_N \
    actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=$use_kl_loss \
    actor_rollout_ref.actor.use_off_policy_loss=True \
    actor_rollout_ref.actor.off_policy_normalize=False \
    actor_rollout_ref.actor.off_policy_reshape="no_reshape" \
    actor_rollout_ref.actor.off_policy_loss_impl=token \
    actor_rollout_ref.actor.off_policy_cliprange=$CLIP_RANGE \
    algorithm.grpo_use_std=$GRPO_STD \
    +algorithm.add_all_positive=$ADD_ALL_POSITIVE \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=$REMOVE_CLIP \
    data.reward_impl_version=3 \
    trainer.max_optim_to_keep=1 \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=5 "${@:1}"
