MODEL_FLAGS="--actual_image_size 512 --lr_warmup_steps 0 --weight_decay 0.15 --classifier_attention_resolutions 64,32,16,8 --image_size 64 --classifier_width 128 --classifier_depth 4 --classifier_use_fp16 True "
TRAIN_FLAGS="--lr 2e-5 --batch_size 20 --log_interval 10 --save_interval 10000 --kl_model kl.pt"
export OPENAI_LOGDIR=./logs_classifier/

mpiexec -n 4 python scripts/classifier_train_stable.py --good_data_dir /your-images/ --bad_data_dir /laion-images/ $MODEL_FLAGS $TRAIN_FLAGS
