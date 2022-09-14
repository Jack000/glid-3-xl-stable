# GLID-3-XL-stable

GLID-3-xl-stable is [stable diffusion](https://github.com/CompVis/stable-diffusion) back-ported to the OpenAI guided diffusion codebase, for easier development and training. It also has additional features for conditional generation and super resolution.

[upscale examples](https://github.com/Jack000/glid-3-xl-stable/wiki/Double-diffusion-for-more-detailed-upscaling)

# Install

First install [latent diffusion](https://github.com/CompVis/latent-diffusion)
```
# then
git clone https://github.com/Jack000/glid-3-xl-stable
cd glid-3-xl-stable
pip install -e .

# install mpi and mpi4py for training
sudo apt install libopenmpi-dev
pip install mpi4py

```

# Get model files from stable diffusion

```
# split checkpoint
python split.py sd-v1-3.ckpt

# you should now have diffusion.pt and kl.pt

# alternatively
wget -O diffusion.pt https://huggingface.co/Jack000/glid-3-xl-stable/resolve/main/default/diffusion-1.4.pt
wget -O kl.pt https://huggingface.co/Jack000/glid-3-xl-stable/resolve/main/default/kl-1.4.pt

```

# Generating images
note: best results at 512x512 image size

```
python sample.py --model_path diffusion.pt --batch_size 3 --num_batches 3 --text "a cyberpunk girl with a scifi neuralink device on her head"

# sample with an init image
python sample.py --init_image picture.jpg --skip_timesteps 20 --model_path diffusion.pt --batch_size 3 --num_batches 3 --text "a cyberpunk girl with a scifi neuralink device on her head"

# generated images saved to ./output/
# generated image embeddings saved to ./output_npy/ as npy files
```

# Upscaling
note: best results at 512x512 input and 1024x1024 output (default settings) 11gb vram required
```
# download model
wget -O upscale.pt https://huggingface.co/Jack000/glid-3-xl-stable/resolve/main/super_lg/ema_0.999_040000.pt

python super_large.py --image img.png --skip_timesteps 1

```

# Training/Fine tuning
Train with same flags as guided diffusion. Data directory should contain image and text files with the same name (image1.png image1.txt)

```
# minimum 48gb vram to train
# batch sizes are per-gpu, not total

MODEL_FLAGS="--actual_image_size 512 --lr_warmup_steps 10000 --ema_rate 0.9999 --attention_resolutions 64,32,16 --class_cond False --diffusion_steps 1000 --image_size 64 --learn_sigma False --noise_schedule linear --num_channels 320 --num_heads 8 --num_res_blocks 2 --resblock_updown False --use_fp16 True --use_scale_shift_norm False "
TRAIN_FLAGS="--lr 1e-5 --batch_size 8 --log_interval 10 --save_interval 5000 --kl_model kl.pt --resume_checkpoint diffusion.pt"
export OPENAI_LOGDIR=./logs/
python scripts/image_train_stable.py --data_dir /path/to/image-and-text-files $MODEL_FLAGS $TRAIN_FLAGS

# multi-gpu
mpiexec -n N python scripts/image_train_stable.py --data_dir /path/to/image-and-text-files $MODEL_FLAGS $TRAIN_FLAGS
```

```
# merge checkpoint back into single .pt (for compatibility with other stable diffusion tools)

python merge.py sd-v1-3.ckpt ./logs/finetuned-ema-checkpoint.pt

```

# Training classifier

```
# example configs for 4x3090
MODEL_FLAGS="--actual_image_size 512 --weight_decay 0.15 --classifier_attention_resolutions 64,32,16,8 --image_size 64 --classifier_width 128 --classifier_depth 4 --classifier_use_fp16 True "
TRAIN_FLAGS="--lr 2e-5 --batch_size 20 --log_interval 10 --save_interval 10000 --kl_model kl.pt"
export OPENAI_LOGDIR=./logs_classifier/

mpiexec -n 4 python scripts/classifier_train_stable.py --good_data_dir /your-images/ --bad_data_dir /laion-images/ $MODEL_FLAGS $TRAIN_FLAGS
```

# Training large upscale model

```
# minimum 80gb vram to train
MODEL_FLAGS="--actual_image_size 1024 --lr_warmup_steps 1000 --ema_rate 0.999 --weight_decay 0.005 --attention_resolutions 64,32 --class_cond False --diffusion_steps 1000 --image_size 1024 --learn_sigma True --noise_schedule linear_openai --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True "
TRAIN_FLAGS="--lr 5e-5 --batch_size 6 --microbatch 3 --log_interval 1 --save_interval 5000 --kl_model kl.pt --resume_checkpoint 256x256_diffusion_uncond.pt"
export OPENAI_LOGDIR=./logs_super/
mpiexec -n 8 python scripts/image_train_super.py --data_dir /path/to/image-and-text-files $MODEL_FLAGS $TRAIN_FLAGS

```
