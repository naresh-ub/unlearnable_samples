import os
import argparse
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from diffusers import (
    UNet2DConditionModel,
    DDPMScheduler,
    StableDiffusionPipeline,
    AutoencoderKL,
    CLIPTextModel,
    CLIPTokenizer
)
from diffusers.utils import EMAModel
from accelerate import Accelerator, ProjectConfiguration
from accelerate.utils import set_seed
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train shortcut diffusion model from scratch with Diffusers.")
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_train_steps", type=int, default=1_000_000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, choices=["no","fp16","bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--checkpointing_steps", type=int, default=100_000)
    parser.add_argument("--log_interval", type=int, default=1_000)
    parser.add_argument("--bootstrap_every", type=int, default=8)
    parser.add_argument("--bootstrap_dt_bias", type=int, default=0)
    parser.add_argument("--bootstrap_cfg", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--class_dropout_prob", type=float, default=0.1)
    parser.add_argument("--denoise_timesteps", type=int, default=128)
    return parser.parse_args()


def get_targets(args, latents, labels, unet, ema_unet, encoder_hidden_states, device):
    B = latents.shape[0]
    bs_boot = B // args.bootstrap_every
    log2_sec = int(math.log2(args.denoise_timesteps))
    # 1) Sample dt_base
    if args.bootstrap_dt_bias == 0:
        base = []
        for i in range(log2_sec):
            base += [log2_sec - 1 - i] * (bs_boot // log2_sec)
        dt_base = torch.tensor(base + [0] * (bs_boot - len(base)), device=device)
        num_dt_cfg = bs_boot // log2_sec
    else:
        base = []
        for i in range(log2_sec - 2):
            base += [log2_sec - 1 - i] * ((bs_boot // 2) // log2_sec)
        base += [1] * (bs_boot // 4) + [0] * (bs_boot // 4)
        dt_base = torch.tensor(base + [0] * (bs_boot - len(base)), device=device)
        num_dt_cfg = (bs_boot // 2) // log2_sec
    dt = 1.0 / (2 ** dt_base.float())
    dt_boot = dt / 2

    # 2) Sample bootstrap t
    dt_sections = 2 ** dt_base
    t_int = torch.randint(0, dt_sections, (bs_boot,), device=device)
    t_norm = t_int.float() / dt_sections.float()
    t_full = t_norm.view(-1, 1, 1, 1)

    # 3) Bootstrap targets
    x1 = latents[:bs_boot]
    x0 = torch.randn_like(x1)
    xt = (1 - (1 - 1e-5) * t_full) * x0 + t_full * x1
    bst_labels = labels[:bs_boot]

    model_fn = ema_unet if (args.use_ema and ema_unet is not None) else unet
    model_fn.eval()
    with torch.no_grad():
        if args.bootstrap_cfg == 0:
            v_b1 = model_fn(xt, t_int, encoder_hidden_states).sample
            x_t2 = xt + dt_boot.view(-1, 1, 1, 1) * v_b1
            x_t2 = x_t2.clamp(-4, 4)
            v_b2 = model_fn(x_t2, t_int, encoder_hidden_states).sample
            v_target = (v_b1 + v_b2) / 2
        else:
            x_extra = torch.cat([xt, xt[:num_dt_cfg]], dim=0)
            t_extra = torch.cat([t_int, t_int[:num_dt_cfg]], dim=0)
            labels_extra = torch.cat([
                bst_labels,
                torch.full((num_dt_cfg,), args.denoise_timesteps, dtype=torch.long, device=device)
            ], dim=0)
            v_raw = model_fn(x_extra, t_extra, encoder_hidden_states).sample
            v_cond = v_raw[:bs_boot]
            v_uncond = v_raw[bs_boot:]
            v_cfg = v_uncond + args.cfg_scale * (v_cond[:num_dt_cfg] - v_uncond[:num_dt_cfg])
            v_b1 = torch.cat([v_cfg, v_cond[num_dt_cfg:]], dim=0)

            x_t2 = xt + dt_boot.view(-1, 1, 1, 1) * v_b1
            x_t2 = x_t2.clamp(-4, 4)
            x_extra2 = torch.cat([x_t2, x_t2[:num_dt_cfg]], dim=0)
            t_extra2 = torch.cat([t_int, t_int[:num_dt_cfg]], dim=0)
            v2_raw = model_fn(x_extra2, t_extra2, encoder_hidden_states).sample
            v_cond2 = v2_raw[:bs_boot]
            v_uncond2 = v2_raw[bs_boot:]
            v_cfg2 = v_uncond2 + args.cfg_scale * (v_cond2[:num_dt_cfg] - v_uncond2[:num_dt_cfg])
            v_b2 = torch.cat([v_cfg2, v_cond2[num_dt_cfg:]], dim=0)
            v_target = (v_b1 + v_b2) / 2

    v_target = v_target.clamp(-4, 4)

    # 4) Flow-matching targets
    drop_mask = torch.rand(labels.shape, device=device) < args.class_dropout_prob
    labels_d = torch.where(
        drop_mask,
        torch.full_like(labels, args.denoise_timesteps),
        labels
    )
    info = {"dropped_ratio": drop_mask.float().mean().item()}

    t_flow_int = torch.randint(
        0, args.denoise_timesteps, (B,), device=device
    )
    t_flow_norm = t_flow_int.float() / args.denoise_timesteps
    t_flow_full = t_flow_norm.view(-1, 1, 1, 1)
    x1f = latents
    x0f = torch.randn_like(x1f)
    x_t_flow = (1 - (1 - 1e-5) * t_flow_full) * x0f + t_flow_full * x1f
    v_t_flow = x1f - (1 - 1e-5) * x0f

    # 5) Merge bootstrap + flow
    rest = B - bs_boot
    x_t_final = torch.cat([xt, x_t_flow[:rest]], dim=0)
    t_int_final = torch.cat([t_int, t_flow_int[:rest]], dim=0)
    v_t_final = torch.cat([v_target, v_t_flow[:rest]], dim=0)
    labels_final = torch.cat([bst_labels, labels_d[:rest]], dim=0)
    info.update({
        "bootstrap_ratio": (dt_base.ne(int(math.log2(args.denoise_timesteps))).float().mean().item()),
        "v_magnitude_bootstrap": v_target.square().mean().sqrt().item(),
        "v_magnitude_b1": v_b1.square().mean().sqrt().item(),
        "v_magnitude_b2": v_b2.square().mean().sqrt().item()
    })

    return x_t_final, v_t_final, t_int_final, labels_final, info


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir)
    )
    set_seed(args.seed)

    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(accelerator.device)
    text_encoder.requires_grad_(False)

    # Load VAE
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4").to(accelerator.device)
    vae.requires_grad_(False)

    # Initialize UNet from config (train from scratch)
    pretrained_unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet"
    )
    unet = UNet2DConditionModel.from_config(pretrained_unet.config)
    del pretrained_unet
    unet.train()

    # EMA
    ema_unet = None
    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config
        )
        ema_unet.to(accelerator.device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    # Data
    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.ImageFolder(args.train_data_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    # Training loop
    step = 0
    for epoch in range(1):  # single pass over DataLoader for num_train_steps
        for images, labels in dataloader:
            if step >= args.num_train_steps:
                break

            images = images.to(accelerator.device)
            labels = labels.to(accelerator.device)
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

            # Dummy text conditioning: use zero tokens
            input_ids = torch.zeros((latents.size(0), 1), dtype=torch.long, device=accelerator.device)
            encoder_hidden_states = text_encoder(input_ids).last_hidden_state

            # Get targets
            x_t, v_t, t_int, labels_masked, info = get_targets(
                args, latents, labels, unet, ema_unet, encoder_hidden_states, accelerator.device
            )

            # Forward
            output = unet(x_t, t_int, encoder_hidden_states).sample
            loss = F.mse_loss(output.float(), v_t.float(), reduction="mean")

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            if ema_unet:
                ema_unet.step(unet.parameters())

            step += 1
            if step % args.log_interval == 0 and accelerator.is_main_process:
                print(f"Step {step} Loss {loss.item():.4f}")

            if step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{step}"))

    # Final save
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if ema_unet:
            ema_unet.copy_to(unet.parameters())
        scheduler = DDPMScheduler(num_train_timesteps=args.denoise_timesteps)
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=scheduler
        )
        pipeline.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
