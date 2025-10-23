import torch
import base64, io
import numpy as np
from qwen_vl_utils import process_vision_info
from scipy.spatial.transform import Rotation as R
from torchvision.transforms.functional import to_pil_image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def tensor_to_base64_img(t: torch.Tensor, fmt: str = "JPEG") -> str:
    """
    Args
    ----
    t   : torch.Tensor, shape (3, H, W). 0‑1 float or 0‑255 uint8
    fmt : "JPEG" or "PNG", etc.

    Returns
    -------
    data‑URI
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
    """
    # 1) CPU, uint8
    t = t.detach().cpu()
    if t.dtype != torch.uint8:
        t = (t.clamp(0, 1) * 255).byte()

    # 2) Tensor → PIL
    pil_img = to_pil_image(t)

    # 3) PIL → bytes
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=95)  # quality 95
    b = buf.getvalue()

    # 4) bytes → Base64 + data‑URI
    b64 = base64.b64encode(b).decode("utf‑8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def build_messages(cmd: str, image_list: list):
    message = [
        {"role": "user", "content": [{"type": "text", "text": cmd}]}
    ]
    if len(image_list) > 0:
        for image in image_list:
            base64_image = tensor_to_base64_img(image)
            message[0]["content"].append(
                {"type": "image", "image": base64_image}
            )
    return message


def build_messages_rewrite(cmd: str, image_list: list):
    message = [
        {"role": "user", "content": [{"type": "text",
                                      "text": f"Rewrite the instruction in neutral, concise English. The wording and syntax could be altered. Must only output the rewrite. \nOriginal Instruction: {cmd}"}]}
    ]
    if len(image_list) > 0:
        for image in image_list:
            base64_image = tensor_to_base64_img(image)
            message[0]["content"].append(
                {"type": "image", "image": base64_image}
            )
    return message


def quat2gevc(quat, g_axis=np.float32([0.0, 0.0, -1.0])):
    # quat: w, x, y, z
    rot = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    gvec = rot.T @ g_axis
    return gvec


def get_move_flag(cmd_vel):
    return (np.linalg.norm(cmd_vel, axis=1) > 0.2).astype(np.float32)[:, None]


def get_rewrite_model(config):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.rewrite_model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        attn_implementation="flash_attention_2" if config.bf16 else "eager",
    )
    processor = AutoProcessor.from_pretrained(config.rewrite_model)
    processor.tokenizer.padding_side = "left"
    return model, processor


def rewrite_instruction(model, proc, language_cmds: list, image_list: list = None):
    messages = [build_messages_rewrite(language_cmds[i], image_list[i] if image_list else []) for i in
                range(len(language_cmds))]
    language_cmds = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    vlm_inputs = proc(text=language_cmds, images=image_inputs, return_tensors="pt", padding=True)
    vlm_inputs = vlm_inputs.to(device=model.device, dtype=model.dtype)

    generated_ids = model.generate(
        **vlm_inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.9,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.05,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(vlm_inputs.input_ids, generated_ids)
    ]
    output_texts = proc.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts
