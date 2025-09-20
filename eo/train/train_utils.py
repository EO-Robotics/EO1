import os

import torch
import transformers
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


def set_requires_grad(parameters, requires_grad):
    """Set the requires_grad attribute for the parameters."""
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(vlm, training_args, compute_dtype, device):
    """Configure the vision tower."""
    vision_tower = vlm.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = vlm.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    merger_params = vlm.visual.merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)


def configure_llm(vlm, training_args):
    """Configure the LLM."""
    lm_head = vlm.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = vlm.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def configure_processor(processor, dataset, training_args):
    """Configure the processor."""
    if training_args.chat_template:
        import json

        chat_template = json.load(open(training_args.chat_template))["chat_template"]
        processor.chat_template = processor.tokenizer.chat_template = chat_template
        logger.info("Set chat template", main_process_only=True)

    if dataset.lerobot_dataset:
        logger.info(f"Set features, stats and mode {training_args.state_mode}", main_process_only=True)
        robo_config = dataset.lerobot_dataset.configuration
        robo_config["max_action_dim"] = training_args.max_action_dim
        robo_config["action_chunk_size"] = training_args.chunk_size
        processor.set_normalization(robo_config)

        logger.info("Set qwen2.5 VL min-max pixels", main_process_only=True)
        processor.image_processor.min_pixels = training_args.image_min_pixels
        processor.image_processor.max_pixels = training_args.image_max_pixels


def smart_tokenizer_and_embedding_resize(
    processor: transformers.ProcessorMixin,
    vlm: transformers.PreTrainedModel,
):
    """Smart tokenizer and embedding resize."""
    from eo.constants import (
        ACTION_END_TOKEN,
        ACTION_START_TOKEN,
        DEFAULT_ACTION_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_STATE_TOKEN,
        DEFAULT_VIDEO_TOKEN,
        PASS_ACTION_TOKEN,
        STATE_END_TOKEN,
        STATE_START_TOKEN,
        TASK_VLA_TOKEN,
        VISION_START_TOKEN,
    )

    tokenizer = processor.tokenizer
    eo1_special_tokens = [
        ACTION_START_TOKEN,
        DEFAULT_ACTION_TOKEN,
        PASS_ACTION_TOKEN,
        ACTION_END_TOKEN,
        STATE_START_TOKEN,
        DEFAULT_STATE_TOKEN,
        STATE_END_TOKEN,
        TASK_VLA_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(eo1_special_tokens, special_tokens=True)

    # NOTE: qwen2.5 vl vocab 151936 > tokenizer 151664 + 8, we don't need to resize embeddings
    token_dict = {
        "state_token_id": DEFAULT_STATE_TOKEN,
        "action_token_start_id": ACTION_START_TOKEN,
        "action_token_id": DEFAULT_ACTION_TOKEN,
        "action_pass_id": PASS_ACTION_TOKEN,
        "vision_token_start_id": VISION_START_TOKEN,
        "image_token_id": DEFAULT_IMAGE_TOKEN,
        "video_token_id": DEFAULT_VIDEO_TOKEN,
    }

    for key, token in token_dict.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        setattr(vlm.model.config, key, token_id)
        setattr(vlm.config.text_config, key, token_id)

    processor.action_token_id = vlm.model.config.action_token_id
    processor.action_pass_id = vlm.model.config.action_pass_id
    return num_new_tokens


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=None, verbose=True):
    """Find the target linear names for LoRA."""
    if lora_namespan_exclude is None:
        lora_namespan_exclude = []
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def aggregate_dataset_length(dataset):
    """Aggregate the lengths of the dataset, used for dataset packing."""
    import bisect

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dataset.mm_dataset.sample_actions = False
    total_data_len = len(dataset)
    mm_dataset_len = len(dataset.mm_dataset)

    num_workers = int(os.environ.get("DATASET_NUM_PROCESSES", 8))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    data_iter = iter(dataloader)
    logger.info(f"Using {num_workers} workers to aggregate lengths ...")

    # get lerobot lengths
    cumulative_sizes = dataset.lerobot_dataset.cumulative_sizes
    lerobot_lengths = [len(dataset[mm_dataset_len + idx]["input_ids"]) for idx in [0] + cumulative_sizes[:-1]]
    for i, (repo_id, length) in enumerate(
        zip(dataset.lerobot_dataset.repo_ids, lerobot_lengths, strict=False)
    ):
        print(f"repo_id {repo_id}, avg len {length}, dataset len {len(dataset.lerobot_dataset._datasets[i])}")

    lengths = []
    for i in tqdm(range(total_data_len), desc="Aggregating lengths"):
        if i < mm_dataset_len:
            mm_data = dataset.mm_dataset[i]
            len_i = mm_data.get("seq_length")
            if len_i is None:
                item = next(data_iter)
                len_i = item["input_ids"].shape[-1]
        else:
            dataset_idx = bisect.bisect_right(cumulative_sizes, i - mm_dataset_len)
            len_i = lerobot_lengths[dataset_idx]
        lengths.append(len_i)

    dataset.mm_dataset.sample_actions = True
    dataset.cached_lengths = lengths
    dataset.__setattr__("cached_lengths", lengths)
    assert len(lengths) == total_data_len, f"Length mismatch: {len(lengths)} != {total_data_len}"
    return lengths
