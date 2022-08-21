MODEL_FLAGS="--actual_image_size 256 --lr_warmup_steps 10000 --ema_rate 0.9999 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 32 --learn_sigma False --noise_schedule linear --num_channels 320 --num_heads 8 --num_res_blocks 2 --resblock_updown False --use_fp16 True --use_scale_shift_norm False "
TRAIN_FLAGS="--lr 1e-5 --batch_size 32 --microbatch 8 --log_interval 10 --save_interval 5000 --kl_model kl.pt --resume_checkpoint diffusion.pt"
export OPENAI_LOGDIR=./logs/
export TOKENIZERS_PARALLELISM=false
python scripts/image_train_stable.py --data_dir /path/to/image-and-text-files $MODEL_FLAGS $TRAIN_FLAGS
