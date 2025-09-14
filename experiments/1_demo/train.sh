GPUS=1
PER_DEVICE_BATCH_SIZE=8

ACCELERATE_ARGS="--num_machines 1 --machine_rank 0 --num_processes=${GPUS}"

# * datasets
dataset=experiments/1_demo/data-demo.yaml
dataset_name=$(basename ${dataset%.*})

# hparams
lr=1e-4
mlr=1e-4
vlr=2e-5

chunk_size=30
epoch=50

run_name=${dataset_name}_ck${chunk_size}_gpu${GPUS}_lr${lr}_vlr${vlr}_mlr${mlr}_bs${PER_DEVICE_BATCH_SIZE}

. scripts/env.sh
conda activate eo

accelerate launch $ACCELERATE_ARGS scripts/train.py \
    --vlm-name-or-path ../pretrained/Qwen2.5-VL-3B-Instruct \
    --data-path ${dataset} \
    --chunk-size ${chunk_size} \
    --dataloader-num-workers 8 \
    --freeze-vision-tower False \
    --freeze-llm False \
    --freeze-merger False \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --num-train-epochs ${epoch} \
    --per-device-train-batch-size ${PER_DEVICE_BATCH_SIZE} \
    --learning-rate ${lr} \
    --merger-lr ${mlr} \
    --vision-lr ${vlr} \
    --weight-decay 0.1 \
    --warmup-ratio 0.03 \
    --lr-scheduler-type cosine \
    --gradient-checkpointing True \
    --save-strategy steps \
    --logging-steps 100 \
    --save-steps 5000 \
    --save-total-limit 3 \
    --report-to none \
    --run-name ${run_name} \
    --attn-implementation flash_attention_2
