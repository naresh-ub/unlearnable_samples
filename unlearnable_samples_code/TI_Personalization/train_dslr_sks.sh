
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/celeba_ids/3875"
export OUTPUT_DIR="./celeba_3875_dslr"

CUDA_VISIBLE_DEVICES=6 python TI_dslr_sks.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<dslr-sks>" \
  --initializer_token="human" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1500 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="A dslr portrait of <dslr-sks> person." \
  --num_validation_images=6 \
  --validation_steps=200