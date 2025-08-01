# export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/db_objects/dog8"
export OUTPUT_DIR="./db_dog_8_sks-2-1"
export TYPE="dog"
export PLACEHOLDER="<db-dog8>"

CUDA_VISIBLE_DEVICES=6 python TI_photo_sks-2-1.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
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
  --validation_prompt="A photo of $PLACEHOLDER $TYPE." \
  --num_validation_images=8 \
  --validation_steps=200

# export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/db_objects/robot_toy"
export OUTPUT_DIR="./db_robot_toy_sks-2-1"
export TYPE="robot"
export PLACEHOLDER="<db-robot-toy>"

CUDA_VISIBLE_DEVICES=6 python TI_photo_sks-2-1.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
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
  --validation_prompt="A photo of $PLACEHOLDER $TYPE." \
  --num_validation_images=8 \
  --validation_steps=200




