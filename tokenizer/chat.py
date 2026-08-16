SPECIAL = {
    "bos": "<|bos|>", "eos": "<|eos|>", "pad": "<|pad|>",
    "system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>",
}

def format_chat(messages, add_generation_prompt=False):
    text = SPECIAL["bos"]
    for message in messages:
        role = message["role"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported role: {role}")
        text += SPECIAL[role] + message["content"] + SPECIAL["eos"]
    if add_generation_prompt:
        text += SPECIAL["assistant"]
    return text

