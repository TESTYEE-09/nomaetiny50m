from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import training.model as model_module
from training.model import TinyLM, ModelConfig


def export_rope(x, positions, theta):
    d = x.shape[-1]
    inv = 1 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))
    angles = positions.float()[:, None] * inv[None, :]
    c = angles.cos()[None, None, :, :].to(x.dtype)
    s = angles.sin()[None, None, :, :].to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


model_module.rope = export_rope


def export_rms_norm(self, x):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + 1e-6).to(x.dtype)
    return normalized * self.weight


model_module.RMSNorm.forward = export_rms_norm


class BrowserModel(nn.Module):
    def __init__(self, model: TinyLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids, *past):
        x = self.model.embed(input_ids)
        past_len = past[0].shape[2]
        positions = torch.arange(past_len, past_len + input_ids.shape[1], device=x.device)
        present = []
        for index, block in enumerate(self.model.blocks):
            residual = x
            y = block.n1(x)
            b, t, _ = y.shape
            q = block.attn.q(y).view(b, t, block.attn.h, block.attn.d).transpose(1, 2)
            k = block.attn.k(y).view(b, t, block.attn.kv, block.attn.d).transpose(1, 2)
            v = block.attn.v(y).view(b, t, block.attn.kv, block.attn.d).transpose(1, 2)
            q, k = export_rope(q, positions, block.attn.theta), export_rope(k, positions, block.attn.theta)
            k = torch.cat((past[index * 2], k), dim=2)
            v = torch.cat((past[index * 2 + 1], v), dim=2)
            present.extend((k, v))
            repeat = block.attn.h // block.attn.kv
            kr = k.repeat_interleave(repeat, 1)
            vr = v.repeat_interleave(repeat, 1)
            scores = torch.matmul(q, kr.transpose(-2, -1)) / (block.attn.d ** 0.5)
            query_positions = torch.arange(t, device=x.device)[:, None] + past_len
            key_positions = torch.arange(k.shape[2], device=x.device)[None, :]
            scores = scores.masked_fill(key_positions > query_positions, -10000.0)
            attention = torch.softmax(scores.float(), dim=-1).to(x.dtype)
            attended = torch.matmul(attention, vr).transpose(1, 2).contiguous().view(b, t, -1)
            x = residual + block.attn.o(attended)
            y = block.n2(x)
            x = x + block.d(torch.nn.functional.silu(block.g(y)) * block.u(y))
        logits = torch.nn.functional.linear(self.model.norm(x[:, -1:, :]), self.model.embed.weight)
        return (logits[:, 0, :].float(), *present)


def main():
    target = ROOT / "site" / "public" / "model" / "tiny50m.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(ROOT / "export" / "model-fp16.pt", map_location="cpu", weights_only=True)
    state = payload.get("model", payload)
    model = TinyLM(ModelConfig()).eval()
    model.load_state_dict(state)
    wrapped = BrowserModel(model).eval()
    sample = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    cache = tuple(torch.zeros(1, 2, 2, 64) for _ in range(24))
    input_names = ["input_ids"] + [f"past_{i}_{kind}" for i in range(12) for kind in ("key", "value")]
    output_names = ["logits"] + [f"present_{i}_{kind}" for i in range(12) for kind in ("key", "value")]
    dynamic_axes = {"input_ids": {1: "sequence"}}
    for name in input_names[1:]:
        dynamic_axes[name] = {2: "past_sequence"}
    for name in output_names[1:]:
        dynamic_axes[name] = {2: "total_sequence"}
    torch.onnx.export(
        wrapped,
        (sample, *cache),
        target,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        dynamo=False,
        external_data=False,
    )
    print(target, target.stat().st_size)


if __name__ == "__main__":
    main()
