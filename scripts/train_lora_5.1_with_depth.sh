export MODEL_NAME="models/Diffusion_Transformer/EasyAnimateV5.1-7b-zh-InP"
export DATASET_NAME="/mnt/chenyang_lei/Datasets/easyanimate_dataset"
export DATASET_META_NAME="/mnt/chenyang_lei/Datasets/easyanimate_dataset/realestate_dataset/metadata.json"
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# When train model with multi machines, use "--config_file accelerate.yaml" instead of "--mixed_precision='bf16'".
accelerate launch \
  --use_deepspeed \
  --deepspeed_config_file config/zero_stage2_config.json \
  --deepspeed_multinode_launcher standard \
  --main_process_port 29501 \
  scripts/train_lora_5.1_with_depth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --config_path "config/easyanimate_video_v5.1_magvit_qwen_with_depth.yaml" \
  --image_sample_size=1024 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_stride=4 \
  --video_sample_n_frames=49 \
  --train_batch_size=2 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=8 \
  --num_train_epochs=3 \
  --checkpointing_steps=4019 \
  --learning_rate=1e-04 \
  --seed=42 \
  --low_vram \
  --output_dir="output_dir_20250227_inpainting_with_depth_lora" \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=5e-3 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=0.05 \
  --loss_type="flow" \
  --use_deepspeed \
  --uniform_sampling \
  --train_mode="inpaint"
  --resume_from_checkpoint="latest"