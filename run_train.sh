gpu=$1
threads=20
export OPENBLAS_NUM_THREADS=${threads}
export BLIS_NUM_THREADS=${threads}
export OMP_NUM_THREADS=${threads}
export MKL_NUM_THREADS=${threads}
export NUMEXPR_NUM_THREADS=${threads}
WANDB_PROJECT=wandb CUDA_VISIBLE_DEVICES=$gpu python run_charbert.py \
    --train_file /data/zihan/data/wikipedia_yuxuan/enwiki.txt \
    --validation_file /data/zihan/data/wikipedia_yuxuan/en.dev.txt \
    --do_train \
    --do_eval \
    --learning_rate 1e-4 \
    --max_steps 1000000 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --warmup_steps 10000 \
    --save_steps 50000 \
    --fp16 \
    --max_seq_length 128 \
    --mask_word_cutoff 0.9 \
    --output_dir charbert-base-full-improved-mask-word \
    --report_to wandb \
    --run_name charbert-base-full-improved-mask-word

