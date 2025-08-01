# inference_unlearnable.py

import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from models.U_UNet import UnlearnableUnetModel

def parse_args():
    parser = argparse.ArgumentParser(description="Generate unlearnable samples with pretrained models")
    parser.add_argument("--input_image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated unlearnable image")
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion model ID or path for VAE & scheduler")
    parser.add_argument("--shortcut_weights", type=str, required=True,
                        help="Path to .pth file for shortcut diffusion UNet")
    parser.add_argument("--unlearnable_weights", type=str, required=True,
                        help="Path to .pth file for trained Unlearnable UNet")
    parser.add_argument("--denoise_steps", type=int, default=4,
                        help="Number of denoising steps for shortcut UNet")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Resize shorter side to this resolution")
    return parser.parse_args()

def load_models(args, device):
    # VAE
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae").to(device)
    vae.requires_grad_(False)
    vae.eval()

    # Scheduler
    scheduler = DDPMScheduler.from_pretrained(args.sd_model, subfolder="scheduler")
    scheduler.set_timesteps(args.denoise_steps)

    # Shortcut UNet
    base_unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet")
    config = base_unet.config
    del base_unet
    shortcut = UNet2DConditionModel.from_config(config).to(device)
    sd_w = torch.load(args.shortcut_weights, map_location=device)
    shortcut.load_state_dict(sd_w)
    shortcut.requires_grad_(False)
    shortcut.eval()

    # Unlearnable UNet
    unlearn = UnlearnableUnetModel(in_channels=config.in_channels).to(device)
    ul_w = torch.load(args.unlearnable_weights, map_location=device)
    unlearn.load_state_dict(ul_w)
    unlearn.requires_grad_(False)
    unlearn.eval()

    return vae, scheduler, shortcut, unlearn

def preprocess_image(path, resolution, device):
    img = Image.open(path).convert("RGB")
    # Resize keeping aspect ratio, shorter side -> resolution
    w, h = img.size
    if w < h:
        new_w = resolution
        new_h = int(h * resolution / w)
    else:
        new_h = resolution
        new_w = int(w * resolution / h)
    img = img.resize((new_w, new_h), resample=Image.LANCZOS)
    # Center crop to square
    left = (new_w - resolution) // 2
    top = (new_h - resolution) // 2
    img = img.crop((left, top, left + resolution, top + resolution))
    # To tensor and normalize
    t = transforms.ToTensor()(img)
    t = transforms.Normalize([0.5]*3, [0.5]*3)(t)
    return t.unsqueeze(0).to(device)

def save_image(tensor, path):
    # tensor: (1,3,H,W), values in [-1,1]
    img = tensor.detach().cpu().clamp(-1,1)
    img = (img + 1) / 2.0  # [0,1]
    img = img.mul(255).permute(0,2,3,1).byte().squeeze(0).numpy()
    Image.fromarray(img).save(path)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae, scheduler, shortcut_unet, unlearnable_unet = load_models(args, device)
    input_tensor = preprocess_image(args.input_image, args.resolution, device)

    # 1) Encode to latent
    with torch.no_grad():
        latent = vae.encode(input_tensor).latent_dist.sample() * vae.config.scaling_factor

    # 2) Forward diffuse to final timestep
    noise = torch.randn_like(latent)
    t_final = torch.full((latent.size(0),), scheduler.config.num_train_timesteps - 1,
                         dtype=torch.long, device=device)
    noisy = scheduler.add_noise(latent, noise, t_final)

    # 3) Perturb via Unlearnable UNet
    with torch.no_grad():
        perturbed = unlearnable_unet(noisy, t_final, None).sample

    # 4) Shortcut denoise back to latent0
    lat = perturbed
    for t in scheduler.timesteps:
        with torch.no_grad():
            pred = shortcut_unet(lat, t, None).sample
            lat = scheduler.step(pred, t, lat).prev_sample
    denoised_latent = lat

    # 5) Decode to image space
    with torch.no_grad():
        reconstructed = vae.decode(denoised_latent).sample

    # Save output
    out_path = os.path.join(args.output_dir, "unlearnable_sample.png")
    save_image(reconstructed, out_path)
    print(f"Saved unlearnable sample to {out_path}")

if __name__ == "__main__":
    main()