import os
import sys
from pathlib import Path

import torch
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    pass
from transformers import HfArgumentParser

from eo.data.dataset import make_supervised_data_module
from eo.model.modeling_eo1 import EO1VisionFlowMatchingModel
from eo.model.processing_eo1 import OneVisionProcessor
from eo.train.pipeline_config import TrainPipelineConfig
from eo.train.train_utils import (
    aggregate_dataset_length,
    configure_llm,
    configure_processor,
    configure_vision_tower,
    find_target_linear_names,
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
)
from eo.train.trainer import OneVisionTrainer

logger = get_logger(__name__, log_level="INFO")


def train():
    parser = HfArgumentParser(TrainPipelineConfig)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (training_args,) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (training_args,) = parser.parse_args_into_dataclasses()
    training_args.output_dir = broadcast_object_list([training_args.output_dir])[0]
    logger.info(f"set output-dir to {training_args.output_dir}")

    # configure model
    compute_dtype = (
        torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    if training_args.model_name_or_path is None:
        model = EO1VisionFlowMatchingModel.from_pretrained(
            training_args.vlm_name_or_path,
            build_projector=False,
            torch_dtype=compute_dtype,
            attn_implementation=training_args.attn_implementation,
            action_act=training_args.action_act,
        )
        model.build_projector(dtype=torch.float32)
    else:
        model = EO1VisionFlowMatchingModel.from_pretrained(
            training_args.model_name_or_path,
            torch_dtype=compute_dtype,
            attn_implementation=training_args.attn_implementation,
        )

    # load processor and resize embeddings
    processor = OneVisionProcessor.from_pretrained(
        training_args.processor_name_or_path,
        padding_side="right",
        use_fast=True,
    )
    smart_tokenizer_and_embedding_resize(processor, model)

    # configure model
    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)
    model.config.action_chunk_size = training_args.chunk_size

    # lora peft tuning
    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                lora_namespan_exclude=lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        logger.info("adding LoRA to the model...", main_process_only=True)
        model = get_peft_model(model, peft_config)

    # load dataset
    data_module = make_supervised_data_module(processor=processor, args=training_args)
    configure_processor(processor, data_module["train_dataset"], training_args)

    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.warning(
        f"{total_params=}, {trainable_params=}, [{trainable_params / total_params * 100}%]",
        main_process_only=True,
    )

    trainer = OneVisionTrainer(model=model, processing_class=processor, args=training_args, **data_module)

    # aggregate data lengths for packing
    if training_args.pack_dataset:
        import numpy as np

        dataset = data_module["train_dataset"].dataset
        lengths = None
        if trainer.accelerator.is_main_process:
            lengths = aggregate_dataset_length(dataset)
        lengths = broadcast_object_list([lengths])[0]
        dataset.cached_lengths = lengths

        data_module["train_dataset"]._pack()
        packed_lens = data_module["train_dataset"].packed_lengths

        logger.info(
            f"group length {len(lengths)=}, {min(lengths)=}, {max(lengths)=},",
            f"packed data {len(packed_lens)=}, {min(packed_lens)=}, {max(packed_lens)=}, {np.mean(packed_lens)=}",
            main_process_only=True,
        )
    else:
        dataset = data_module["train_dataset"]

    if trainer.accelerator.is_main_process:
        dataset.info_qwen_vision_fetch()
        input_ids = dataset[0]["input_ids"]
        print(f"sample: {processor.tokenizer.decode(input_ids)}")

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        logger.info("resume from checkpoint")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    model.config.use_cache = True
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=f"{training_args.output_dir}/checkpoint-final-{trainer.state.global_step}"
    )


if __name__ == "__main__":
    train()
