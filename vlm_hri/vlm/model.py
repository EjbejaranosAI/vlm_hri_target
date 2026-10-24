"""Carga del VLM (Qwen2-VL) con cuantización 4-bit opcional."""

from __future__ import annotations

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from ..config import _PATCH, _VLM_ATTN, VLM_COMPILE, VLM_ID, VLM_JEV_ID, VLM_LOAD_IN_4BIT, vlm_backend, vlm_resize_limits


def load_vlm(model_id: str, device: torch.device):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    _, max_pixels, _ = vlm_resize_limits()
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=256 * _PATCH, max_pixels=max_pixels
    )
    kwargs: dict = {"torch_dtype": dtype}
    if VLM_LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["attn_implementation"] = _VLM_ATTN
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if not VLM_LOAD_IN_4BIT:
        model = model.to(device)
    model.eval()
    if VLM_COMPILE:
        model = torch.compile(model, mode="reduce-overhead")
    return model, processor


def backend_model_id() -> str:
    return VLM_JEV_ID if vlm_backend() == "vlm_jev" else VLM_ID


def load_backend(device: torch.device, *, warmup: bool = True):
    """(model, processor) según VLM_BACKEND. Con vlm_jev es (JevEngine,
    policy del YAML) -- PoseStreamSession lo despacha a vlm.jev."""
    if vlm_backend() == "vlm_jev":
        from .jev import load_jev, warmup_jev

        engine, policy = load_jev(VLM_JEV_ID, device)
        if warmup:
            warmup_jev(engine, policy)
        return engine, policy
    from .inference import warmup_vlm

    model, processor = load_vlm(VLM_ID, device)
    if warmup:
        warmup_vlm(model, processor)
    return model, processor
