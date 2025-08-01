export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/db_objects/clock"
export OUTPUT_DIR="./db_clock_sks"
export TYPE="clock"
export PLACEHOLDER="<db-clock>"

CUDA_VISIBLE_DEVICES=7 python TI_photo_sks.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token=$PLACEHOLDER \
  --initializer_token=$TYPE \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1500 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="A photo of $PLACEHOLDER $TYPE." \
  --num_validation_images=8 \
  --validation_steps=100

export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/db_objects/duck_toy"
export OUTPUT_DIR="./db_duck_toy_sks"
export TYPE="duck"
export PLACEHOLDER="<db-duck-toy>"

CUDA_VISIBLE_DEVICES=7 python TI_photo_sks.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token=$PLACEHOLDER \
  --initializer_token=$TYPE \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1500 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="A photo of $PLACEHOLDER $TYPE." \
  --num_validation_images=8 \
  --validation_steps=100


export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/db_objects/monster_toy"
export OUTPUT_DIR="./db_monster_toy_sks"
export TYPE="monster"
export PLACEHOLDER="<db-monster-toy>"

CUDA_VISIBLE_DEVICES=7 python TI_photo_sks.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token=$PLACEHOLDER \
  --initializer_token=$TYPE \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1500 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="A photo of $PLACEHOLDER $TYPE." \
  --num_validation_images=8 \
  --validation_steps=100

export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/db_objects/backpack"
export OUTPUT_DIR="./db_backpack_sks"
export TYPE="backpack"
export PLACEHOLDER="<db-backpack>"

CUDA_VISIBLE_DEVICES=7 python TI_photo_sks.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token=$PLACEHOLDER \
  --initializer_token=$TYPE \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1500 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="A photo of $PLACEHOLDER $TYPE." \
  --num_validation_images=8 \
  --validation_steps=100

export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/db_objects/cat"
export OUTPUT_DIR="./db_cat_sks"
export TYPE="cat"
export PLACEHOLDER="<db-cat-sks>"

CUDA_VISIBLE_DEVICES=7 python TI_photo_sks.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token=$PLACEHOLDER \
  --initializer_token=$TYPE \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1500 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="A photo of $PLACEHOLDER $TYPE." \
  --num_validation_images=8 \
  --validation_steps=100