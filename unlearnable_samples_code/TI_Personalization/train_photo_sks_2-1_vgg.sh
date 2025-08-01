# # export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
# export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/vggface2_ids_new/n000080"
# export OUTPUT_DIR="./n000080-2-1"
# export TYPE="person"
# export PLACEHOLDER="<n000080-token>"

# CUDA_VISIBLE_DEVICES=7 python TI_photo_sks-2-1.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATA_DIR \
#   --learnable_property="object" \
#   --placeholder_token=$PLACEHOLDER \
#   --initializer_token=$TYPE \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=400 \tr
#   --learning_rate=5.0e-04 \
#   --scale_lr \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --output_dir=$OUTPUT_DIR \
#   --validation_prompt="a photo of $PLACEHOLDER person." \
#   --num_validation_images=12 \
#   --validation_steps=200

# # export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
# export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/vggface2_ids_new/n000098"
# export OUTPUT_DIR="./n000098-2-1"
# export TYPE="person"
# export PLACEHOLDER="<n000098-token>"

# CUDA_VISIBLE_DEVICES=7 python TI_photo_sks-2-1.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATA_DIR \
#   --learnable_property="object" \
#   --placeholder_token=$PLACEHOLDER \
#   --initializer_token=$TYPE \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=400 \
#   --learning_rate=5.0e-04 \
#   --scale_lr \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --output_dir=$OUTPUT_DIR \
#   --validation_prompt="a photo of $PLACEHOLDER person." \
#   --num_validation_images=12 \
#   --validation_steps=200

# # export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
# export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/vggface2_ids_new/n000176"
# export OUTPUT_DIR="./n000176-2-1"
# export TYPE="person"
# export PLACEHOLDER="<n000176-token>"

# CUDA_VISIBLE_DEVICES=7 python TI_photo_sks-2-1.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATA_DIR \
#   --learnable_property="object" \
#   --placeholder_token=$PLACEHOLDER \
#   --initializer_token=$TYPE \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=400 \
#   --learning_rate=5.0e-04 \
#   --scale_lr \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --output_dir=$OUTPUT_DIR \
#   --validation_prompt="a photo of $PLACEHOLDER person." \
#   --num_validation_images=12 \
#   --validation_steps=200

# # export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
# export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
# export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/vggface2_ids_new/n000076"
# export OUTPUT_DIR="./n000076-2-1"
# export TYPE="person"
# export PLACEHOLDER="<n000076-token>"

# CUDA_VISIBLE_DEVICES=7 python TI_photo_sks-2-1.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --train_data_dir=$DATA_DIR \
#   --learnable_property="object" \
#   --placeholder_token=$PLACEHOLDER \
#   --initializer_token=$TYPE \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=400 \
#   --learning_rate=5.0e-04 \
#   --scale_lr \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --output_dir=$OUTPUT_DIR \
#   --validation_prompt="a photo of $PLACEHOLDER person." \
#   --num_validation_images=12 \
#   --validation_steps=200

# export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/vggface2_ids_new/n000161"
export OUTPUT_DIR="./n000161-2-1"
export TYPE="person"
export PLACEHOLDER="<n000161-token>"

CUDA_VISIBLE_DEVICES=7 python TI_photo_sks-2-1.py \
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
  --validation_prompt="a photo of $PLACEHOLDER person." \
  --num_validation_images=12 \
  --validation_steps=200

# export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export DATA_DIR="/home/csgrad/devulapa/phd/acm_baselines/data/vggface2_ids_new/n000150"
export OUTPUT_DIR="./n000150-2-1"
export TYPE="person"
export PLACEHOLDER="<n000150-token>"

CUDA_VISIBLE_DEVICES=7 python TI_photo_sks-2-1.py \
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
  --validation_prompt="a photo of $PLACEHOLDER person." \
  --num_validation_images=12 \
  --validation_steps=200