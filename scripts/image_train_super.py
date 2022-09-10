"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_text_super import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import torch
import random

from ldm.util import instantiate_from_config

from omegaconf import OmegaConf

import torchvision.transforms as T

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("loading vae...")

    kl_config = OmegaConf.load('kl.yaml')
    kl_sd = torch.load(args.kl_model, map_location="cpu")

    encoder = instantiate_from_config(kl_config.model)
    encoder.load_state_dict(kl_sd, strict=True)

    encoder.to(dist_util.dev())
    encoder.eval()
    encoder.requires_grad_(False)
    set_requires_grad(encoder, False)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())

    logger.log('total base parameters', sum(x.numel() for x in model.parameters()))

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_latent_data(
        encoder,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.actual_image_size,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        lr_warmup_steps=args.lr_warmup_steps,
    ).run_loop()

def load_latent_data(encoder, data_dir, batch_size, image_size):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=False,
    )
    for batch, batch_small in data:
        model_kwargs = {}

        batch_small = batch_small.to(dist_util.dev())
        emb = encoder.encode(batch_small).sample().half()
        emb *= 0.18215

        model_kwargs["super_res_embed"] = emb

        yield batch, model_kwargs

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        kl_model=None,
        actual_image_size=1024,
        lr_warmup_steps=0,
    )
    defaults.update(model_and_diffusion_defaults())

    defaults['image_condition'] = False
    defaults['super_res_condition'] = True
    defaults['context_dim'] = None
    defaults['use_new_attention_order'] = False
    defaults['use_spatial_transformer'] = False

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
