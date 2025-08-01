# export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/wikiart/post_impression"
export OUTPUT_DIR="./post_impressionism_sks-2-1"
export TYPE="art"
export PLACEHOLDER="<new-post_impressionism-token>"

CUDA_VISIBLE_DEVICES=7 python TI_photo_sks-2-1_wikiart.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="style" \
  --placeholder_token=$PLACEHOLDER \
  --initializer_token=$TYPE \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=400 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="a painting in the style of $PLACEHOLDER." \
  --num_validation_images=8 \
  --validation_steps=200