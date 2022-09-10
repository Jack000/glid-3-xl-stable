import gc
import io
import math
import sys

from PIL import Image, ImageOps
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

import numpy as np

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

from einops import rearrange
from math import log2, sqrt

import argparse
import pickle

import os

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default = 'upscale.pt',
                   help='path to the diffusion model')

parser.add_argument('--kl_path', type=str, default = 'kl.pt',
                   help='path to the LDM first stage model')

parser.add_argument('--image', type = str, required = True, default = '',
                    help='path to image (npy containing latent embeddings or image file)')

parser.add_argument('--text', type = str, required = False, default = '',
                    help='your text prompt (not necessary). If provided, use clip guidance')

parser.add_argument('--cutn', type = int, default = 16, required = False,
                    help='Number of cuts for clip guidance')

parser.add_argument('--clip_guidance_scale', type = int, default = 10000, required = False,
                    help='Amount of clip guidance')

parser.add_argument('--tv_scale', type = int, default = 0, required = False,
                    help='Controls the smoothness of the final output')

parser.add_argument('--range_scale', type = int, default = 0, required = False,
                    help='Controls how far out of range RGB values are allowed to be')

parser.add_argument('--init_image', type=str, required = False, default = None,
                   help='init image to use')

parser.add_argument('--skip_timesteps', type=int, required = False, default = 0,
                   help='how many diffusion steps are gonna be skipped')

parser.add_argument('--prefix', type = str, required = False, default = 'super_',
                    help='prefix for output files')

parser.add_argument('--num_batches', type = int, default = 1, required = False,
                    help='number of batches')

parser.add_argument('--batch_size', type = int, default = 1, required = False,
                    help='batch size')

parser.add_argument('--width', type = int, default = 0, required = False,
                    help='image size of output (multiple of 8)')

parser.add_argument('--height', type = int, default = 0, required = False,
                    help='image size of output (multiple of 8)')

parser.add_argument('--seed', type = int, default=-1, required = False,
                    help='random seed')

parser.add_argument('--steps', type = int, default = 0, required = False,
                    help='number of diffusion steps')

parser.add_argument('--cpu', dest='cpu', action='store_true')

parser.add_argument('--guide', dest='guide', action='store_true')

parser.add_argument('--avg', dest='avg', action='store_true')

parser.add_argument('--clip_score', dest='clip_score', action='store_true')

parser.add_argument('--ddim', dest='ddim', action='store_true') # turn on to use 50 step ddim

parser.add_argument('--ddpm', dest='ddpm', action='store_true') # turn on to use 50 step ddim

args = parser.parse_args()

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()

        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])

device = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
print('Using device:', device)

model_state_dict = torch.load(args.model_path, map_location='cpu')

model_params = {
    'attention_resolutions': '64,32',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '15',
    'image_size': 1024,
    'learn_sigma': True,
    'noise_schedule': 'linear_openai',
    'num_channels': 256,
    'num_head_channels': 64,
    'num_res_blocks': 2,
    'resblock_updown': True,
    'use_fp16': True,
    'use_scale_shift_norm': True,
    'clip_embed_dim': None,
    'image_condition': False,
    'super_res_condition': True,
    'context_dim': None,
    'use_spatial_transformer': False,
}

if args.ddpm:
    model_params['timestep_respacing'] = '1000'
if args.ddim:
    if args.steps:
        model_params['timestep_respacing'] = 'ddim'+str(args.steps)
    else:
        model_params['timestep_respacing'] = 'ddim50'
elif args.steps:
    model_params['timestep_respacing'] = str(args.steps)

model_config = model_and_diffusion_defaults()
model_config.update(model_params)

if args.cpu:
    model_config['use_fp16'] = False

# Load models
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(model_state_dict, strict=True)
model.requires_grad_(True if args.text or args.guide else False).eval().to(device)

if model_config['use_fp16']:
    model.convert_to_fp16()
else:
    model.convert_to_fp32()

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

kl_config = OmegaConf.load('kl.yaml')
kl_sd = torch.load(args.kl_path, map_location="cpu")

ldm = instantiate_from_config(kl_config.model)
ldm.load_state_dict(kl_sd, strict=True)

ldm.to(device)
ldm.eval()
ldm.requires_grad_(False)
set_requires_grad(ldm, False)

if args.image.endswith('.npy'):
    image = np.load(args.image)
    im_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
    input_image = ldm.decode(im_tensor).clamp(-1,1)
    input_image_small = F.interpolate(input_image, size=(im_tensor.shape[3]*8, im_tensor.shape[2]*8), mode='bilinear')
    args.width = im_tensor.shape[3]*16
    args.height = im_tensor.shape[2]*16
else:
    image = Image.open(args.image).convert('RGB')

    if args.width == 0 and args.height == 0:
        crop_width = math.floor(image.width/64)*64
        crop_height = math.floor(image.height/64)*64

        left = (image.width-crop_width)//2
        top = (image.height-crop_height)//2

        image = image.crop((left, top, left+crop_width, top+crop_height))

        args.width = 2*crop_width
        args.height = 2*crop_height
    else:
        image = ImageOps.fit(image, (args.width//2, args.height//2))

    input_image = TF.to_tensor(image).to(device).unsqueeze(0)
    input_image = input_image * 2 - 1

    im_tensor = ldm.encode(input_image).sample().to(device)
    input_image_small = F.interpolate(input_image, size=(im_tensor.shape[3]*4, im_tensor.shape[2]*4), mode='bilinear')

if args.text:
    import clip

    # if text is given, use clip guidance
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()

    clip_model, clip_preprocess = clip.load('ViT-L/14', jit=False)
    clip_model.eval().requires_grad_(False).to(device)
    clip_size = clip_model.visual.input_resolution

    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    make_cutouts = MakeCutouts(clip_size, args.cutn)

    side_x = args.width
    side_y = args.height

    target_embeds, weights = [], []

    prompts = args.text.split('|')

    for prompt in prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(prompt).to(device)).float())
        weights.append(weight)

    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError('The weights must not sum to 0.')
    weights /= weights.sum().abs()


def do_run():

    if args.seed >= 0:
        torch.manual_seed(args.seed)

    kwargs = {
        "super_res_embed": im_tensor*0.18215,
    }

    if args.batch_size > 1:
        kwargs['super_res_embed'] = torch.cat(args.batch_size*[kwargs['super_res_embed']], dim=0)

    cur_t = None

    if args.ddpm:
        sample_fn = diffusion.ddpm_sample_loop_progressive
    elif args.ddim:
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    def save_sample(i, sample, clip_score=False):
        for k, image in enumerate(sample['pred_xstart']):
            out = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))

            filename = f'output/{args.prefix}{i * args.batch_size + k:05}.png'
            out.save(filename)

    def cond_fn_clip(x, t, super_res_embed=None):

        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]

            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t

            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'super_res_embed': super_res_embed})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)
            clip_in = normalize(make_cutouts(x_in.add(1).div(2)))
            clip_embeds = clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(clip_embeds.unsqueeze(1), target_embeds.unsqueeze(0))
            dists = dists.view([args.cutn, n, -1])
            losses = dists.mul(weights).sum(2).mean(0)
            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out['pred_xstart'])
            loss = losses.sum() * args.clip_guidance_scale + tv_losses.sum() * args.tv_scale + range_losses.sum() * args.range_scale
            return -torch.autograd.grad(loss, x)[0]

    def cond_fn_super(x, t, super_res_embed=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()

            n = x.shape[0]

            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t

            out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={'super_res_embed': super_res_embed})

            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out['pred_xstart'] * fac + x * (1 - fac)

            x_small = F.interpolate(x_in, size=(x.shape[3]//2, x.shape[2]//2), mode='bilinear')
            diff = x_small - input_image_small

            loss = 100*(diff**2).mean([1, 2, 3]).sum()

            return -torch.autograd.grad(loss, x)[0]

    if args.init_image:
        init = Image.open(args.init_image).convert('RGB')
        init = init.resize((int(args.width), int(args.height)), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0)
        init = 2*init - 1
    elif args.skip_timesteps > 0:
        init = F.interpolate(input_image, size=(args.height, args.width), mode='bicubic')
    else:
        init = None

    if args.text:
        cond_fn = cond_fn_clip
    elif args.guide:
        cond_fn = cond_fn_super
    else:
        cond_fn = None

    if args.avg:
        kwargs['super_res_embed'] = torch.cat([kwargs['super_res_embed'], kwargs['super_res_embed']], dim=0)
        kwargs['avg'] = True
        if init is not None:
            init = torch.cat([init, init], dim=0)

    for i in range(args.num_batches):
        cur_t = diffusion.num_timesteps - 1

        samples = sample_fn(
            model,
            (args.batch_size*2 if args.avg else args.batch_size, 3, args.height, args.width),
            clip_denoised=False,
            model_kwargs=kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=True,
            init_image=init,
            skip_timesteps=args.skip_timesteps,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            if j % 5 == 0 and j != diffusion.num_timesteps - 1:
                save_sample(i, sample)

        save_sample(i, sample, args.clip_score)

gc.collect()
do_run()
