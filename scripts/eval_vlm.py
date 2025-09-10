from transformers import AutoModel, AutoProcessor

"""set model name or path"""
model_name_or_path = "../pretrained/Qwen2.5-VL-3B-Instruct"
model_name_or_path = "outputs/eofm/2025-06-22/01-14-04-eofm_pretrain_data-eo-stage1_ck16_MEAN_STD_pix64-128_gpu80_lr5e-5_vlr1e-5_mlr5e-5_bs1_16384_ep10zero1-pack-eo/checkpoint-75000"
# model_name_or_path="outputs/eofmdev/2025-06-03/10-57-54-eofm_data-libero_ck8_MEAN_STD_pix64-128_gpu8_lr1e-4_vlr2e-5_mlr1e-4_bs64_ep50_linear-le/checkpoint-final-26750"

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
model = AutoModel.from_pretrained(
    model_name_or_path,
    # attn_implementation="flex_attention",
    # torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/cpfs01/shared/optimal/vla_next/ERQA/decoded_data/images/example_000000_image_00.jpg",
            },
            {
                "type": "text",
                "text": "If the yellow robot gripper follows the yellow trajectory, what will happen? Choices: A. Robot puts the soda on the wooden steps. B. Robot moves the soda in front of the wooden steps. C. Robot moves the soda to the very top of the wooden steps. D. Robot picks up the soda can and moves it up. Please answer directly with only the letter of the correct option and nothing else.",
            },
        ],
    },
]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/cpfs01/shared/optimal/vla_next/ERQA/decoded_data/images/example_000008_image_00.jpg",
            },
            {
                "type": "text",
                "text": "How should the Kuka robot with orange arm and grey grippers move to pick up the spatula? Choices: A. Move right. B. Move up. C. Move lower. D. Close gripper. Please answer directly with only the letter of the correct option and nothing else.",
            },
        ],
    },
]

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/cpfs01/shared/optimal/vla_next/ERQA/decoded_data/images/example_000021_image_00.jpg",
            },
            {
                "type": "text",
                "text": "Which beverage can is the farthest on the left? Choices: A. Pepsi. B. Water bottle. C. Red Bull. D. Coke. Please answer directly with only the letter of the correct option and nothing else.",
            },
        ],
    },
]

times = 0
past_key_values = None

while True:
    if times > 0:
        prompt = input("Enter your prompt: ")
        if prompt == "q":
            exit(0)
        messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to("cuda")

    input_length = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs, max_new_tokens=1024, past_key_values=past_key_values, return_dict_in_generate=True
    )

    past_key_values = outputs.past_key_values
    generated_ids = outputs.sequences

    completion = processor.decode(generated_ids[0, input_length:], skip_special_tokens=False)
    print(completion)

    messages.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
    times += 1
