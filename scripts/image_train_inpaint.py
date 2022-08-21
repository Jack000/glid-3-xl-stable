"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_text_datasets import load_data
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

from encoders.modules import BERTEmbedder

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    from clip_custom import clip # make clip end up on the right device

    logger.log("loading clip...")
    clip_model, _ = clip.load('ViT-L/14', device=dist_util.dev(), jit=False)
    clip_model.eval().requires_grad_(False)
    set_requires_grad(clip_model, False)

    del clip_model.visual

    logger.log("loading vae...")

    encoder = torch.load(args.kl_model, map_location="cpu")
    encoder.half().to(dist_util.dev())
    encoder.eval()
    set_requires_grad(encoder, False)

    del encoder.decoder
    del encoder.loss

    logger.log("loading text encoder...")

    
    bert = BERTEmbedder(1280, 32)
    sd = torch.load(args.bert_model, map_location="cpu")
    bert.load_state_dict(sd)

    bert.half().to(dist_util.dev())
    bert.eval()
    set_requires_grad(bert, False)

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
        bert,
        clip_model,
        clip,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
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
    ).run_loop()

def load_latent_data(encoder, bert, clip_model, clip, data_dir, batch_size, image_size):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=256,
        class_cond=False,
    )

    blur = transforms.GaussianBlur(kernel_size=(15, 15), sigma=(0.1, 5))

    for batch, model_kwargs, text in data:

        text = list(text)
        for i in range(len(text)):
            if random.randint(0,100) < 20:
                text[i] = ''

        text_emb = bert.encode(text).to(dist_util.dev()).half()

        clip_text = clip.tokenize(text, truncate=True).to(dist_util.dev())
        clip_emb = clip_model.encode_text(clip_text)

        model_kwargs["context"] = text_emb
        model_kwargs["clip_embed"] = clip_emb

        batch = batch.to(dist_util.dev())
        emb = encoder.encode(batch.half()).sample().half()
        emb *= 0.18215

        emb_cond = emb.detach().clone()

        for i in range(batch.shape[0]):
            if random.randint(0,100) < 20:
                emb_cond[i,:,:,:] = 0 # unconditional
            else:
                if random.randint(0,100) < 50:
                    mask = torch.randn(1, 32, 32)
                    mask = blur(mask)
                    mask = (mask > 0)
                    mask = mask.repeat(4, 1, 1)
                    mask = mask.float()
                    emb_cond[i] *= mask
                else:
                    # mask out 4 random rectangles
                    for j in range(random.randint(1,4)):
                        max_area = 32*16
                        w = random.randint(1,32)
                        h = random.randint(1,32)
                        if w*h > max_area:
                            if random.randint(0,100) < 50:
                                w = max_area//h
                            else:
                                h = max_area//w
                        if w == 32:
                            offsetx = 0
                        else:
                            offsetx = random.randint(0, 32-w)
                        if h == 32:
                            offsety = 0
                        else:
                            offsety = random.randint(0, 32-h)
                        emb_cond[i,:, offsety:offsety+h, offsetx:offsetx+w] = 0

        model_kwargs["image_embed"] = emb_cond

        yield emb, model_kwargs

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
        bert_model=None,
    )
    defaults.update(model_and_diffusion_defaults())

    defaults['clip_embed_dim'] = 768
    defaults['image_condition'] = True

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
