import gc
import io
import math
import sys

from PIL import Image, ImageOps, ImageDraw
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.ops import masks_to_boxes

from tqdm.notebook import tqdm

import numpy as np

from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults, classifier_defaults, create_classifier

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

parser.add_argument('--classifier', type=str, default = '',
                   help='path to the classifier model')

parser.add_argument('--classifier_scale', type = int, required = False, default = 100,
                    help='amount of classifier guidance')

parser.add_argument('--edit', type = str, required = False,
                    help='path to the image you want to edit (either an image file or .npy containing a numpy array of the image embeddings)')

parser.add_argument('--outpaint', type = str, required = False, default = '',
                    help='options: expand (all directions), wider, taller, left, right, top, bottom')

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

parser.add_argument('--ddim', dest='ddim', action='store_true')

parser.add_argument('--ddpm', dest='ddpm', action='store_true')

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
    'timestep_respacing': '50',
    'image_size': 32,
    'learn_sigma': False,
    'noise_schedule': 'linear',
    'num_channels': 320,
    'num_heads': 8,
    'num_res_blocks': 2,
    'resblock_updown': False,
    'use_fp16': False,
    'use_scale_shift_norm': False,
    'clip_embed_dim': None,
    'image_condition': True if model_state_dict['input_blocks.0.0.weight'].shape[1] == 8 else False,
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

# Create output folders
os.makedirs("output", exist_ok = True)
os.makedirs("output_npy", exist_ok = True)
    
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

# load classifier
if args.classifier:
    classifier_config = classifier_defaults()
    classifier_config['classifier_width'] = 128
    classifier_config['classifier_depth'] = 4
    classifier_config['classifier_attention_resolutions'] = '64,32,16,8'
    classifier = create_classifier(**classifier_config)
    classifier.load_state_dict(
        torch.load(args.classifier, map_location="cpu")
    )
    classifier.to(device)
    classifier.convert_to_fp16()
    classifier.eval()

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
        else:
            input_image_pil = Image.open(fetch(args.edit)).convert('RGB')

            im = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device)
            im = 2*im-1
            im = ldm.encode(im).sample()

        if im.shape[3] < 64:
            im2 = torch.zeros(1,4,im.shape[2],64)
            x = (64-im.shape[3])//2
            im2[:,:,:,x:x+im.shape[3]] = im
            im = im2

        if im.shape[2] < 64:
            im2 = torch.zeros(1,4,64,im.shape[3])
            y = (64-im.shape[2])//2
            im2[:,:,y:y+im.shape[2],:] = im
            im = im2


        if args.outpaint == 'expand':
            input_image = torch.zeros(1, 4, im.shape[2]+64, im.shape[3]+64, device=device)
            input_image[:,:,32:32+im.shape[2],32:32+im.shape[3]] = im
            input_image_mask = torch.zeros(1, 1, im.shape[2]+64, im.shape[3]+64, device=device, dtype=torch.bool)
            input_image_mask[:,:,32:32+im.shape[2],32:32+im.shape[3]] = True
        elif args.outpaint == 'wider':
            input_image = torch.zeros(1, 4, im.shape[2], im.shape[3]+64, device=device)
            input_image[:,:,:,32:32+im.shape[3]] = im
            input_image_mask = torch.zeros(1, 1, im.shape[2], im.shape[3]+64, device=device, dtype=torch.bool)
            input_image_mask[:,:,:,32:32+im.shape[3]] = True
        elif args.outpaint == 'taller':
            input_image = torch.zeros(1, 4, im.shape[2]+64, im.shape[3], device=device)
            input_image[:,:,32:32+im.shape[2],:] = im
            input_image_mask = torch.zeros(1, 1, im.shape[2]+64, im.shape[3], device=device, dtype=torch.bool)
            input_image_mask[:,:,32:32+im.shape[2],:] = True
        elif args.outpaint == 'left':
            input_image = torch.zeros(1, 4, im.shape[2], im.shape[3]+32, device=device)
            input_image[:,:,:,32:32+im.shape[3]] = im
            input_image_mask = torch.zeros(1, 1, im.shape[2], im.shape[3]+32, device=device, dtype=torch.bool)
            input_image_mask[:,:,:,32:32+im.shape[3]] = True
        elif args.outpaint == 'right':
            input_image = torch.zeros(1, 4, im.shape[2], im.shape[3]+32, device=device)
            input_image[:,:,:,0:im.shape[3]] = im
            input_image_mask = torch.zeros(1, 1, im.shape[2], im.shape[3]+32, device=device, dtype=torch.bool)
            input_image_mask[:,:,:,0:im.shape[3]] = True
        elif args.outpaint == 'top':
            input_image = torch.zeros(1, 4, im.shape[2]+32, im.shape[3], device=device)
            input_image[:,:,32:32+im.shape[2],:] = im
            input_image_mask = torch.zeros(1, 1, im.shape[2]+32, im.shape[3], device=device, dtype=torch.bool)
            input_image_mask[:,:,32:32+im.shape[2],:] = True
        elif args.outpaint == 'bottom':
            input_image = torch.zeros(1, 4, im.shape[2]+32, im.shape[3], device=device)
            input_image[:,:,0:im.shape[2],:] = im
            input_image_mask = torch.zeros(1, 1, im.shape[2]+32, im.shape[3], device=device, dtype=torch.bool)
            input_image_mask[:,:,0:im.shape[2],:] = True
        else:
            input_image = im
            input_image_mask = torch.ones(1,1,im.shape[2], im.shape[3], device=device, dtype=torch.bool)

        input_image_pil = ldm.decode(input_image)
        input_image_pil = TF.to_pil_image(input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1))

        input_image *= 0.18215

        if args.mask:
            mask_image = Image.open(fetch(args.mask)).convert('L')
            mask_image = mask_image.resize((input_image.shape[3],input_image.shape[2]), Image.ANTIALIAS)
            mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(device)
        else:
            print('draw the area for inpainting, then close the window')
            app = QApplication(sys.argv)
            d = Draw(args.width, args.height, input_image_pil)
            app.exec_()
            mask_image = d.getCanvas().convert('L').point( lambda p: 255 if p < 1 else 0 )
            mask_image.save('mask.png')
            mask_image = mask_image.resize((input_image.shape[3],input_image.shape[2]), Image.ANTIALIAS)
            mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(device)

        mask1 = (mask > 0.5)
        input_image_mask *= mask1

        #mask1 = mask1.float()
        #input_image *= mask1


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

    cond_fn = None

    if args.classifier:
        def cond_fn(x, t, context=None, clip_embed=None, image_embed=None):
            with torch.enable_grad():
                x_in = x[:x.shape[0]//2].detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), torch.ones(x_in.shape[0], dtype=torch.long)]
                return torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    cur_t = None
 
    if args.ddpm:
        sample_fn = diffusion.ddpm_sample_loop_progressive
    elif args.ddim:
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    def save_sample(i, samples, square=None):
        for k, image in enumerate(samples):
            image_scaled = image/0.18215
            im = image_scaled.unsqueeze(0)
            out = ldm.decode(im)

            npy_filename = f'output_npy/{args.prefix}{i * args.batch_size + k:05}.npy'
            with open(npy_filename, 'wb') as outfile:
                np.save(outfile, image_scaled.detach().cpu().numpy())

            out = TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))
            
            if square is not None:
                outdraw = ImageDraw.Draw(out)  
                outdraw.rectangle([(square[0]*8, square[1]*8),(square[0]*8+512, square[1]*8+512)], fill=None, outline ="red")

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

    overlap = 32

    if args.edit:
        for i in range(args.num_batches):
            output = input_image.detach().clone()
            output *= input_image_mask.repeat(1, 4, 1, 1).float()

            mask = input_image_mask.detach().clone()

            box = masks_to_boxes(~mask.squeeze(0))[0]

            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2] + 1)
            y1 = int(box[3] + 1)

            x_num = math.ceil(((x1-x0)-overlap)/(64-overlap))
            y_num = math.ceil(((y1-y0)-overlap)/(64-overlap))

            if x_num < 1:
                x_num = 1
            if y_num < 1:
                y_num = 1

            for y in range(y_num):
                for x in range(x_num):
                    offsetx = x0 + x*(64-overlap)
                    offsety = y0 + y*(64-overlap)

                    if offsetx + 64 > x1:
                        offsetx = x1 - 64
                    if offsetx < 0:
                        offsetx = 0

                    if offsety + 64 > y1:
                        offsety = y1 - 64
                    if offsety < 0:
                        offsety = 0

                    patch_input = output[:,:, offsety:offsety+64, offsetx:offsetx+64]
                    patch_mask = mask[:,:, offsety:offsety+64, offsetx:offsetx+64]

                    if not torch.any(~patch_mask):
                        # region does not require any inpainting
                        output[:,:, offsety:offsety+64, offsetx:offsetx+64] = patch_input
                        continue

                    mask[:,:, offsety:offsety+64, offsetx:offsetx+64] = True

                    patch_init = None

                    if args.skip_timesteps > 0:
                        patch_init = input_image[:,:, offsety:offsety+64, offsetx:offsetx+64]
                        patch_init = torch.cat([patch_init, patch_init], dim=0)

                    skip_timesteps = args.skip_timesteps

                    if not torch.any(patch_mask):
                        # region has no input image, cannot use init
                        patch_init = None
                        skip_timesteps = 0

                    patch_kwargs = {
                        "context": kwargs["context"],
                        "clip_embed": None,
                        "image_embed": torch.cat([patch_input, patch_input], dim=0)
                    }

                    cur_t = diffusion.num_timesteps - 1

                    samples = sample_fn(
                        model_fn,
                        (2, 4, 64, 64),
                        clip_denoised=False,
                        model_kwargs=patch_kwargs,
                        cond_fn=cond_fn,
                        device=device,
                        progress=True,
                        init_image=patch_init,
                        skip_timesteps=skip_timesteps,
                    )

                    for j, sample in enumerate(samples):
                        cur_t -= 1
                        output[0,:, offsety:offsety+64, offsetx:offsetx+64] = sample['pred_xstart'][0]
                        if j % 25 == 0:
                            save_sample(i, output, square=(offsetx, offsety))

                    save_sample(i, output)

    else:
        for i in range(args.num_batches):
            cur_t = diffusion.num_timesteps - 1

            samples = sample_fn(
                model_fn,
                (args.batch_size*2, 4, int(args.height/8), int(args.width/8)),
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
                if j % 20 == 0:
                    save_sample(i, sample['pred_xstart'][:args.batch_size])

            save_sample(i, sample['pred_xstart'][:args.batch_size])

gc.collect()
do_run()
