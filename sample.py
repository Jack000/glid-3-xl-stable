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

from transformers import CLIPTokenizer, CLIPTextModel

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default = 'diffusion.pt',
                   help='path to the diffusion model')

parser.add_argument('--kl_path', type=str, default = 'kl.pt',
                   help='path to the LDM first stage model')

parser.add_argument('--text', type = str, required = False, default = '',
                    help='your text prompt')

parser.add_argument('--edit', type = str, required = False,
                    help='path to the image you want to edit (either an image file or .npy containing a numpy array of the image embeddings)')

parser.add_argument('--edit_x', type = int, required = False, default = 0,
                    help='x position of the edit image in the generation frame (need to be multiple of 8)')

parser.add_argument('--edit_y', type = int, required = False, default = 0,
                    help='y position of the edit image in the generation frame (need to be multiple of 8)')

parser.add_argument('--edit_width', type = int, required = False, default = 0,
                    help='width of the edit image in the generation frame (need to be multiple of 8)')

parser.add_argument('--edit_height', type = int, required = False, default = 0,
                    help='height of the edit image in the generation frame (need to be multiple of 8)')

parser.add_argument('--mask', type = str, required = False,
                    help='path to a mask image. white pixels = keep, black pixels = discard. width = image width/8, height = image height/8')

parser.add_argument('--negative', type = str, required = False, default = '',
                    help='negative text prompt')

parser.add_argument('--init_image', type=str, required = False, default = None,
                   help='init image to use')

parser.add_argument('--skip_timesteps', type=int, required = False, default = 0,
                   help='how many diffusion steps are gonna be skipped')

parser.add_argument('--prefix', type = str, required = False, default = '',
                    help='prefix for output files')

parser.add_argument('--num_batches', type = int, default = 1, required = False,
                    help='number of batches')

parser.add_argument('--batch_size', type = int, default = 1, required = False,
                    help='batch size')

parser.add_argument('--width', type = int, default = 512, required = False,
                    help='image size of output (multiple of 8)')

parser.add_argument('--height', type = int, default = 512, required = False,
                    help='image size of output (multiple of 8)')

parser.add_argument('--seed', type = int, default=-1, required = False,
                    help='random seed')

parser.add_argument('--guidance_scale', type = float, default = 7.0, required = False,
                    help='classifier-free guidance scale')

parser.add_argument('--steps', type = int, default = 0, required = False,
                    help='number of diffusion steps')

parser.add_argument('--cpu', dest='cpu', action='store_true')

parser.add_argument('--clip_score', dest='clip_score', action='store_true')

parser.add_argument('--ddim', dest='ddim', action='store_true') # turn on to use 50 step ddim

parser.add_argument('--ddpm', dest='ddpm', action='store_true') # turn on to use 50 step ddim

args = parser.parse_args()

if args.edit and not args.mask:
    from PyQt5.QtWidgets import QApplication, QMainWindow
    from PyQt5.QtGui import QPainter, QPen
    from PyQt5.QtCore import Qt, QPoint, QRect, QBuffer
    import PyQt5.QtGui as QtGui

    class Draw(QMainWindow):

        def __init__(self, width, height, im):
            super().__init__()
            self.drawing = False
            self.lastPoint = QPoint()

            self.qim = QtGui.QImage(im.tobytes("raw","RGB"), im.width, im.height, QtGui.QImage.Format_RGB888)
            self.image = QtGui.QPixmap.fromImage(self.qim)

            canvas = QtGui.QImage(im.width, im.height, QtGui.QImage.Format_ARGB32)
            self.canvas = QtGui.QPixmap.fromImage(canvas)
            self.canvas.fill(Qt.transparent)

            self.setGeometry(0, 0, im.width, im.height)
            self.resize(self.image.width(), self.image.height())
            self.show()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.drawPixmap(QRect(0, 0, self.image.width(), self.image.height()), self.image)
            painter.drawPixmap(QRect(0, 0, self.canvas.width(), self.canvas.height()), self.canvas)

        def mousePressEvent(self, event):
            if event.button() == Qt.LeftButton:
                self.drawing = True
                self.lastPoint = event.pos()

        def mouseMoveEvent(self, event):
            if event.buttons() and Qt.LeftButton and self.drawing:
                painter = QPainter(self.canvas)
                painter.setPen(QPen(Qt.red, (self.width()+self.height())/20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(self.lastPoint, event.pos())
                self.lastPoint = event.pos()
                self.update()

        def mouseReleaseEvent(self, event):
            if event.button == Qt.LeftButton:
                self.drawing = False

        def getCanvas(self):
            image = self.canvas.toImage()
            buffer = QBuffer()
            buffer.open(QBuffer.ReadWrite)
            image.save(buffer, "PNG")
            pil_im = Image.open(io.BytesIO(buffer.data()))
            return pil_im

        def resizeEvent(self, event):
            self.image = QtGui.QPixmap.fromImage(self.qim)
            self.image = self.image.scaled(self.width(), self.height())

            canvas = QtGui.QImage(self.width(), self.height(), QtGui.QImage.Format_ARGB32)
            self.canvas = QtGui.QPixmap.fromImage(canvas)
            self.canvas.fill(Qt.transparent)

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

device = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
print('Using device:', device)

model_state_dict = torch.load(args.model_path, map_location='cpu')

model_params = {
    'attention_resolutions': '32,16,8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '50',  # Modify this value to decrease the number of
                                 # timesteps.
    'image_size': 32,
    'learn_sigma': False,
    'noise_schedule': 'linear',
    'num_channels': 320,
    'num_heads': 8,
    'num_res_blocks': 2,
    'resblock_updown': False,
    'use_fp16': False,
    'use_scale_shift_norm': False,
    'clip_embed_dim': 768 if 'clip_proj.weight' in model_state_dict else None,
    'image_condition': False,
    #'image_condition': True if model_state_dict['input_blocks.0.0.weight'].shape[1] == 8 else False,
    'super_res_condition': True if 'external_block.0.0.weight' in model_state_dict else False,
}

if args.ddpm:
    model_params['timestep_respacing'] = '1000'
if args.ddim:
    if args.steps:
        model_params['timestep_respacing'] = 'ddim'+str(args.steps)
    else:
        model_params['timestep_respacing'] = 'ddim250'
elif args.steps:
    model_params['timestep_respacing'] = str(args.steps)

model_config = model_and_diffusion_defaults()
model_config.update(model_params)

if args.cpu:
    model_config['use_fp16'] = False

# Load models
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(model_state_dict, strict=True)
model.requires_grad_(False).eval().to(device)

if model_config['use_fp16']:
    model.convert_to_fp16()
else:
    model.convert_to_fp32()

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

# vae
kl_config = OmegaConf.load('kl.yaml')
kl_sd = torch.load(args.kl_path, map_location="cpu")

ldm = instantiate_from_config(kl_config.model)
ldm.load_state_dict(kl_sd, strict=True)

ldm.to(device)
ldm.eval()
ldm.requires_grad_(False)
set_requires_grad(ldm, False)

# clip
clip_version = 'openai/clip-vit-large-patch14'
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_version)
clip_transformer = CLIPTextModel.from_pretrained(clip_version)
clip_transformer.eval().requires_grad_(False).to(device)


def do_run():
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    # clip context
    text = clip_tokenizer([args.text]*args.batch_size, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    text_blank = clip_tokenizer([args.negative]*args.batch_size, truncation=True, max_length=77, return_length=True, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    text_tokens = text["input_ids"].to(device)
    text_blank_tokens = text_blank["input_ids"].to(device)

    text_emb = clip_transformer(input_ids=text_tokens).last_hidden_state
    text_emb_blank = clip_transformer(input_ids=text_blank_tokens).last_hidden_state

    image_embed = None

    # image context
    if args.edit:
        if args.edit.endswith('.npy'):
            with open(args.edit, 'rb') as f:
                im = np.load(f)
                im = torch.from_numpy(im).unsqueeze(0).to(device)

                input_image = torch.zeros(1, 4, args.height//8, args.width//8, device=device)

                y = args.edit_y//8
                x = args.edit_x//8

                ycrop = y + im.shape[2] - input_image.shape[2]
                xcrop = x + im.shape[3] - input_image.shape[3]

                ycrop = ycrop if ycrop > 0 else 0
                xcrop = xcrop if xcrop > 0 else 0

                input_image[0,:,y if y >=0 else 0:y+im.shape[2],x if x >=0 else 0:x+im.shape[3]] = im[:,:,0 if y > 0 else -y:im.shape[2]-ycrop,0 if x > 0 else -x:im.shape[3]-xcrop]

                input_image_pil = ldm.decode(input_image)
                input_image_pil = TF.to_pil_image(input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1))

                input_image *= 0.18215
        else:
            w = args.edit_width if args.edit_width else args.width
            h = args.edit_height if args.edit_height else args.height

            input_image_pil = Image.open(fetch(args.edit)).convert('RGB')
            input_image_pil = ImageOps.fit(input_image_pil, (w, h))

            input_image = torch.zeros(1, 4, args.height//8, args.width//8, device=device)

            im = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device)
            im = 2*im-1
            im = ldm.encode(im).sample()

            y = args.edit_y//8
            x = args.edit_x//8

            input_image = torch.zeros(1, 4, args.height//8, args.width//8, device=device)

            ycrop = y + im.shape[2] - input_image.shape[2]
            xcrop = x + im.shape[3] - input_image.shape[3]

            ycrop = ycrop if ycrop > 0 else 0
            xcrop = xcrop if xcrop > 0 else 0

            input_image[0,:,y if y >=0 else 0:y+im.shape[2],x if x >=0 else 0:x+im.shape[3]] = im[:,:,0 if y > 0 else -y:im.shape[2]-ycrop,0 if x > 0 else -x:im.shape[3]-xcrop]

            input_image_pil = ldm.decode(input_image)
            input_image_pil = TF.to_pil_image(input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1))

            input_image *= 0.18215

        if args.mask:
            mask_image = Image.open(fetch(args.mask)).convert('L')
            mask_image = mask_image.resize((args.width//8,args.height//8), Image.ANTIALIAS)
            mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(device)
        else:
            print('draw the area for inpainting, then close the window')
            app = QApplication(sys.argv)
            d = Draw(args.width, args.height, input_image_pil)
            app.exec_()
            mask_image = d.getCanvas().convert('L').point( lambda p: 255 if p < 1 else 0 )
            mask_image.save('mask.png')
            mask_image = mask_image.resize((args.width//8,args.height//8), Image.ANTIALIAS)
            mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(device)

        mask1 = (mask > 0.5)
        mask1 = mask1.float()

        input_image *= mask1

        image_embed = torch.cat(args.batch_size*2*[input_image], dim=0).float()

    elif model_params['image_condition']:
        # using inpaint model but no image is provided
        image_embed = torch.zeros(args.batch_size*2, 4, args.height//8, args.width//8, device=device)

    kwargs = {
        "context": torch.cat([text_emb, text_emb_blank], dim=0).float(),
        "clip_embed": None,
        "image_embed": image_embed
    }

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + args.guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    cur_t = None
 
    if args.ddpm:
        sample_fn = diffusion.ddpm_sample_loop_progressive
    elif args.ddim:
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    def save_sample(i, sample, clip_score=False):
        for k, image in enumerate(sample['pred_xstart'][:args.batch_size]):
            image /= 0.18215
            im = image.unsqueeze(0)
            out = ldm.decode(im)

            npy_filename = f'output_npy/{args.prefix}{i * args.batch_size + k:05}.npy'
            with open(npy_filename, 'wb') as outfile:
                np.save(outfile, image.detach().cpu().numpy())

            out = TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))

            filename = f'output/{args.prefix}{i * args.batch_size + k:05}.png'
            out.save(filename)

    if args.init_image:
        init = Image.open(args.init_image).convert('RGB')
        init = init.resize((int(args.width),  int(args.height)), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).clamp(0,1)
        h = ldm.encode(init * 2 - 1).sample() *  0.18215
        init = torch.cat(args.batch_size*2*[h], dim=0)
    else:
        init = None

    for i in range(args.num_batches):
        cur_t = diffusion.num_timesteps - 1

        samples = sample_fn(
            model_fn,
            (args.batch_size*2, 4, int(args.height/8), int(args.width/8)),
            clip_denoised=False,
            model_kwargs=kwargs,
            cond_fn=None,
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
