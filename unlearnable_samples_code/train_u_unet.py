import os
import argparse
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import (
    DDPMScheduler,
    AutoencoderKL,
    StableDiffusionPipeline
)
from accelerate import Accelerator, ProjectConfiguration
from accelerate.utils import set_seed
from tqdm import tqdm
from models.U_UNet import UnlearnableUnetModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train Unlearnable Sample Generator")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Stable Diffusion model ID or path for VAE & scheduler")
    parser.add_argument("--shortcut_weights", type=str, required=True,
                        help="Path to local .pth weights for the shortcut diffusion UNet")
    parser.add_argument("--text_inv_checkpoint", type=str, required=True,
                        help="Textual inversion checkpoint ID or path")
    parser.add_argument("--prompt_template", type=str, default="a photo of {} person",
                        help="Template for textual inversion prompt")
    parser.add_argument("--ti_token", type=str, default="<person>",
                        help="Textual inversion token")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_steps", type=int, default=100_000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, choices=["no","fp16","bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--checkpoint_steps", type=int, default=10_000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--denoise_steps", type=int, default=4,
                        help="Number of denoising steps in shortcut model")
    return parser.parse_args()

def setup_logger(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log")
    logger = logging.getLogger("train_unlearnable")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    return logger

def main():
    args = parse_args()
    logger = setup_logger(args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir)
    )
    set_seed(args.seed)
    device = accelerator.device

    # Load & freeze VAE
    logger.info("Loading VAE")
    vae = AutoencoderKL.from_pretrained(args.sd_model, subfolder="vae").to(device)
    vae.requires_grad_(False)

    # Load & prepare scheduler
    logger.info("Loading Scheduler")
    scheduler = DDPMScheduler.from_pretrained(args.sd_model, subfolder="scheduler").to(device)
    scheduler.set_timesteps(args.denoise_steps)

    # Load shortcut UNet weights
    logger.info("Loading shortcut UNet")
    # We only need its config to instantiate our UnlearnableUnetModel later
    base_unet = UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet")
    latent_channels = base_unet.config.in_channels
    del base_unet
    shortcut_unet = UNet2DConditionModel.from_config(
        UNet2DConditionModel.from_pretrained(args.sd_model, subfolder="unet").config
    ).to(device)
    sd_weights = torch.load(args.shortcut_weights, map_location=device)
    shortcut_unet.load_state_dict(sd_weights)
    shortcut_unet.requires_grad_(False)

    # Load Textual Inversion pipeline
    logger.info("Loading Textual Inversion pipeline")
    ti_pipe = StableDiffusionPipeline.from_pretrained(args.sd_model, safety_checker=None, feature_extractor=None).to(device)
    ti_pipe.load_textual_inversion(args.text_inv_checkpoint)
    ti_unet = ti_pipe.unet; ti_unet.requires_grad_(False)
    ti_text_encoder = ti_pipe.text_encoder; ti_text_encoder.requires_grad_(False)
    ti_tokenizer = ti_pipe.tokenizer

    # Instantiate trainable Unlearnable UNet from scratch
    logger.info("Initializing Unlearnable UNet")
    train_unet = UnlearnableUnetModel(in_channels=latent_channels).to(device)
    train_unet.train()

    # EMA
    ema = None
    if args.use_ema:
        from diffusers.utils import EMAModel
        logger.info("Initializing EMA")
        ema = EMAModel(train_unet.parameters(), model_cls=UnlearnableUnetModel, model_config=train_unet.config).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(train_unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # DataLoader
    logger.info("Preparing DataLoader")
    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    dataset = datasets.ImageFolder(args.train_data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    train_unet, optimizer, dataloader = accelerator.prepare(train_unet, optimizer, dataloader)

    logger.info("Starting training for %d steps", args.num_train_steps)
    step = 0
    progress = tqdm(total=args.num_train_steps, disable=not accelerator.is_local_main_process)
    while step < args.num_train_steps:
        for images, _ in dataloader:
            if step >= args.num_train_steps:
                break
            images = images.to(device)

            # 1) Encode to latent
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

            # 2) Add final-step noise
            noise = torch.randn_like(latents)
            t_final = torch.full((latents.size(0),), scheduler.config.num_train_timesteps - 1,
                                  dtype=torch.long, device=device)
            noisy = scheduler.add_noise(latents, noise, t_final)

            # 3) Perturb via trainable unlearnable UNet
            perturbed = train_unet(noisy, t_final, None).sample

            # 4) Shortcut denoise
            lat = perturbed
            for t in scheduler.timesteps:
                with torch.no_grad():
                    pred = shortcut_unet(lat, t, None).sample
                    lat = scheduler.step(pred, t, lat).prev_sample
            denoised = lat

            # Branch1: TI adversarial
            t_rand = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),), device=device)
            noise_rand = torch.randn_like(latents)
            noised_rand = scheduler.add_noise(denoised, noise_rand, t_rand)
            prompt = args.prompt_template.format(args.ti_token)
            inputs = ti_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            enc_states = ti_text_encoder(inputs).last_hidden_state
            pred_noise = ti_unet(noised_rand, t_rand, encoder_hidden_states=enc_states).sample
            loss_ti = -F.mse_loss(pred_noise, noise_rand)

            # Branch2: budget loss
            recon = vae.decode(denoised).sample
            loss_recon = F.mse_loss(recon, images)
            budget = (8/255.)**2
            loss_budget = F.relu(loss_recon - budget)

            # Combine & step
            loss = loss_ti + loss_budget
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if ema: ema.step(train_unet.parameters())

            step += 1
            progress.update(1)
            if step % args.log_interval == 0 and accelerator.is_local_main_process:
                logger.info(f"Step {step} | TI_loss {loss_ti.item():.4f} | Budget_loss {loss_budget.item():.4f}")
            if step % args.checkpoint_steps == 0 and accelerator.is_local_main_process:
                ckpt_dir = os.path.join(args.output_dir, f"ckpt-{step}")
                logger.info("Saving checkpoint %s", ckpt_dir)
                accelerator.save_state(ckpt_dir)

    # End training loop
    logger.info("Training complete. Saving unlearnable UNet and pipeline.")
    if accelerator.is_local_main_process:
        final_unet = accelerator.unwrap_model(train_unet)
        if ema: ema.copy_to(final_unet.parameters())
        # Save unlearnable UNet separately
        torch.save(final_unet.state_dict(), os.path.join(args.output_dir, "unlearnable_unet.pth"))
        
        logger.info("Saved unlearnable UNet weights to %s", args.output_dir)

if __name__ == "__main__":
    main()