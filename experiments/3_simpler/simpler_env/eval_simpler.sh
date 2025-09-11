. scripts/env.sh

dist_tasks=(
    bridge.sh
    drawer_variant_agg.sh
    drawer_visual_matching.sh
    move_near_variant_agg.sh
    move_near_visual_matching.sh
    pick_coke_can_variant_agg.sh
    pick_coke_can_visual_matching.sh
    put_in_drawer_variant_agg.sh
    put_in_drawer_visual_matching.sh
)

action_ensemble_temp=4

ckpt_path=YOUR_CHECKPOINT_PATH
model_name=eo
job_name=simpler
logging_dir=results_${model_name}/${job_name}_ck${action_ensemble_temp}
mkdir -p $logging_dir

conda activate simpler_env
XDG_RUNTIME_DIR=/usr/lib
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

for task in ${dist_tasks[@]}; do
    bash scripts/$task $ckpt_path $model_name \
    $action_ensemble_temp $logging_dir
done

python tools/calc_metrics_evaluation_videos.py \
--log-dir-root $logging_dir