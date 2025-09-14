from transformers import AutoProcessor, AutoModel

"""set model name or path"""
model_name_or_path = "../pretrained/Qwen2.5-VL-3B-Instruct"  # or EO-3B
model = AutoModel.from_pretrained(
    model_name_or_path,
    device_map="auto"
    # attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(model_name_or_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "demo_data/refcoco/images/COCO_train2014_000000168643_2.jpg"},
            {
                "type": "text",
                "text": "If the yellow robot gripper follows the yellow trajectory, what will happen? Choices: A. Robot puts the soda on the wooden steps. B. Robot moves the soda in front of the wooden steps. C. Robot moves the soda to the very top of the wooden steps. D. Robot picks up the soda can and moves it up. Please answer directly with only the letter of the correct option and nothing else.",
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
