import os

import numpy as np
import tensorflow as tf
from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

if __name__ == "__main__":
    args = get_args()
    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )
    print(f"**** {args.policy_model} ****")
    if args.policy_model in ["eo", "eo-1"]:
        assert args.ckpt_path is not None
        from simpler_env.policies.eo.eo_model import EOInference

        model = EOInference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
            action_ensemble_temp=args.action_ensemble_temp,
        )
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
