DEBUG=true
if [ "$DEBUG" = true ]; then
  GPUS=2
  report=none
  data_num_workers=0
  save_steps=200
  logging_steps=1
  PER_DEVICE_BATCH_SIZE=64
  # ACCELERATE_ARGS="--num_machines 1 --num_processes 1 --dynamo_backend=no"
  ACCELERATE_ARGS="--num_machines 1 --num_processes 2 --dynamo_backend=no --multi_gpu"
fi

# distributed settings
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-64}
data_num_workers=${data_num_workers:-8}
report=${report:-wandb}
save_steps=${save_steps:-5000}
logging_steps=${logging_steps:-100}

ACCELERATE_ARGS=${ACCELERATE_ARGS:-"--main_process_ip=\$MASTER_ADDR --main_process_port=\$MASTER_PORT \
  --num_machines ${NODES} --machine_rank 0 --num_processes=${GPUS} --multi_gpu \
  --mixed_precision=bf16 --dynamo_backend=no"}

# * datasets
dataset=experiments/1_demo/data-demo.yaml
dataset_name=$(basename ${dataset%.*})

# hparams
lr=1e-4
mlr=1e-4
vlr=2e-5

chunk_size=16
epoch=50
lerobot_only=True

# fine-tuning
resume_path=
model_name_or_path=
run_name=${dataset_name}_ck${chunk_size}_gpu${GPUS}_lr${lr}_vlr${vlr}_mlr${mlr}_bs${PER_DEVICE_BATCH_SIZE}
echo $run_name

. scripts/env.sh
conda activate eo

export DATASET_NUM_PROCESSES=10
accelerate launch $ACCELERATE_ARGS scripts/train.py \
    ${resume_path:+--output-dir $resume_path} \
    ${model_name_or_path:+--model-name-or-path $model_name_or_path} \
    ${deepspeed:+--deepspeed configs/${deepspeed}.json} \
    --vlm-name-or-path ../pretrained/Qwen2.5-VL-3B-Instruct \
    --train-lerobot-only ${lerobot_only} \
    --data-path ${dataset} \
    --chunk-size ${chunk_size} \
    --dataloader-num-workers ${data_num_workers} \
    --freeze-vision-tower False \
    --freeze-llm False \
    --freeze-merger False \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --num-train-epochs ${epoch} \
    --per-device-train-batch-size ${PER_DEVICE_BATCH_SIZE} \
    --gradient-accumulation-steps 1 \
    --learning-rate ${lr} \
    --merger-lr ${mlr} \
    --vision-lr ${vlr} \
    --weight-decay 0.1 \
    --warmup-ratio 0.03 \
    --lr-scheduler-type cosine \
    --logging-steps ${logging_steps} \
    --gradient-checkpointing True \
    --save-strategy steps \
    --save-steps ${save_steps} \
    --save-total-limit 3 \
    --report-to ${report} \
    --run-name ${run_name} \
    --attn-implementation flash_attention_2
