DEBUG=false
if [ "$DEBUG" = true ]; then
  GPUS=1
  report=none
  data_num_workers=8
  save_steps=1000
  logging_steps=1
  PER_DEVICE_BATCH_SIZE=1
  TORCH_RUN_ARGS="--standalone --nnodes=1"
fi

# distributed settings
GPUS=${GPUS:-128}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

NODES=$((GPUS / GPUS_PER_NODE))
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=1
data_num_workers=${data_num_workers:-8}
report=${report:-wandb}
save_steps=${save_steps:-5000}
logging_steps=${logging_steps:-100}

TORCH_RUN_ARGS=${TORCH_RUN_ARGS:-"--nnodes $NODES --nproc-per-node $GPUS_PER_NODE --master_addr \$MASTER_ADDR --master_port \$MASTER_PORT"}

dataset=configs/pretrain/data-eo-stage1.yaml
# dataset=configs/pretrain/data-eo-stage2.yaml
# dataset=configs/pretrain/data-eo-stage3.yaml

dataset_name=$(basename ${dataset%.*})

# hparams, only base lr is logged
lr=5e-5
mlr=5e-5
vlr=1e-5
chunk_size=16
epoch=5
mipx=64
mapx=$((${mipx} * 2))
state_mode=MEAN_STD
max_packed_length=16384
lerobot_only=False
pack_dataset=True
torch_empty_cache_steps=100
deepspeed=zero1

# fine-tuning
model_name_or_path=
resume_path=
run_name=pretrain_${dataset_name}_ck${chunk_size}_${state_mode}_pix${mipx}-${mapx}_gpu${GPUS}_lr${lr}_vlr${vlr}_mlr${mlr}_bs${PER_DEVICE_BATCH_SIZE}_${max_packed_length}_ep${epoch}${deepspeed}

export DATASET_NUM_PROCESSES=10

torchrun $TORCH_RUN_ARGS onvisfm/train.py \
  ${resume_path:+--output-dir $resume_path} \
  ${model_name_or_path:+--model-name-or-path $model_name_or_path} \
  ${torch_empty_cache_steps:+--torch-empty-cache-steps $torch_empty_cache_steps} \
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
  --gradient-accumulation-steps ${GRADIENT_ACC} \
  --image-min-pixels $((${mipx} * 28 * 28)) \
  --image-max-pixels $((${mapx} * 28 * 28)) \
  --learning-rate ${lr} \
  --merger-lr ${mlr} \
  --vision-lr ${vlr} \
  --weight-decay 0.1 \
  --warmup-ratio 0.001 \
  --lr-scheduler-type cosine \
  --logging-steps ${logging_steps} \
  --gradient-checkpointing True \
  --save-strategy steps \
  --save-steps ${save_steps} \
  --save-total-limit 3 \
  --report-to ${report} \
  --run-name ${run_name} \
  --state-mode ${state_mode} \
  --pack-dataset ${pack_dataset} \
  --max-packed-length ${max_packed_length} \
  --ignore_data_skip True \
  --attn-implementation flash_attention_3 \
  --seed 42