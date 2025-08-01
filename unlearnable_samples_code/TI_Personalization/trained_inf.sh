#!/bin/bash

# Set variables
export concept_path="/home/csgrad/devulapa/phd/acm_baselines/TI_personalization/celeba_3875/learned_embeds-steps-1000.safetensors"
export token_id="<cat-toy>"
export prompt="A photo of a <cat-toy> person in front of eiffel tower"
export output="personalized_imgs/3875_eiffel.png"

# Run the inference script
CUDA_VISIBLE_DEVICES=0 python trained_inference.py \
  --model_id "stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --concept_path "$concept_path" \
  --token_id "$token_id" \
  --prompt "$prompt" \
  --output "$output"