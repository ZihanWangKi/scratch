gpu=$1
TASK_NAME=$2
model=$3
output_dir=eval/"${model//\\/$'-'}"/${TASK_NAME}
echo ${output_dir}

CUDA_VISIBLE_DEVICES=$gpu python run_glue_charbert.py \
    --model_name_or_path ${model} \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --fp16 \
    --output_dir ${output_dir}
