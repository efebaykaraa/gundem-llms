# SPDX-License-Identifier: Apache-2.0
# Adapted from ahmadw/GPT-X3-150M-ONNX, revision 695dd12882449179a75845635adfd4b526e89b64.
# Upstream: https://huggingface.co/ahmadw/GPT-X3-150M-ONNX
# Modifications: reusable model wrapper and runtime; loading/export CLI removed.

import importlib
import torch
from torch import nn
from torch.nn import functional as F
from .runtime import state_specs

def l2_normalize(x):
    # Matches FLA's rsqrt(sum(x*x) + 1e-6) used by the reference model.
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)

def short_conv_step(projected, weight, history):
    window = torch.cat((history, projected.transpose(1, 2)), -1)
    return F.silu((window * weight[:, 0]).sum(-1)), window[..., 1:]

class GPTX3DecodeStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.rotary = importlib.import_module(type(model).__module__).rotary
        self.state_names = [s.name for s in state_specs(self.config.to_dict())]
        cos, sin = self._rope_tables()
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _rope_tables(self):
        c = self.config
        freq = 1.0 / (c.rope_theta ** (torch.arange(0, c.rotary_dim, 2, dtype=torch.float32) / c.rotary_dim))
        angles = torch.outer(torch.arange(c.max_position_embeddings, dtype=torch.float32), freq)
        return angles.cos(), angles.sin()

    def _attention(self, attn, x, cos, sin, past_k, past_v):
        c, batch = self.config, x.shape[0]
        heads, dim, kv_heads = c.num_attention_heads, c.head_dim, c.num_key_value_heads
        q, gate = attn.q_gate_proj(x).view(batch, 1, heads, 2 * dim).chunk(2, -1)
        q = attn.q_norm(q.transpose(1, 2))
        k = attn.k_norm(attn.k_proj(x).view(batch, 1, kv_heads, dim).transpose(1, 2))
        v = attn.v_proj(x).view(batch, 1, kv_heads, dim).transpose(1, 2)
        q, k = self.rotary(q, cos, sin), self.rotary(k, cos, sin)
        k, v = torch.cat((past_k, k), 2), torch.cat((past_v, v), 2)
        groups = heads // kv_heads
        y = F.scaled_dot_product_attention(q, k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1))
        y = y * torch.sigmoid(gate.transpose(1, 2))
        return attn.o_proj(y.transpose(1, 2).reshape(batch, 1, heads * dim)), [k, v]

    def _gdn2(self, kernel, x, recurrent, conv_q, conv_k, conv_v):
        c, batch = self.config, x.shape[0]
        dim = c.gdn_head_dim
        heads = c.hidden_size // dim
        q, conv_q = short_conv_step(kernel.q_proj(x), kernel.q_conv1d.weight, conv_q)
        k, conv_k = short_conv_step(kernel.k_proj(x), kernel.k_conv1d.weight, conv_k)
        v, conv_v = short_conv_step(kernel.v_proj(x), kernel.v_conv1d.weight, conv_v)
        q, k, v = (t.view(batch, heads, dim) for t in (q, k, v))
        decay = F.softplus(kernel.f_proj(x) + kernel.dt_bias).view(batch, heads, dim)
        decay = -kernel.A_log.exp()[None, :, None] * decay
        erase_gate = kernel.b_proj(x).sigmoid().view(batch, heads, dim)
        write_gate = kernel.w_proj(x).sigmoid().view(batch, heads, dim)
        q, k = l2_normalize(q) * dim**-0.5, l2_normalize(k)
        decayed = recurrent * decay.exp().unsqueeze(-1)
        erase = ((erase_gate * k).unsqueeze(-1) * decayed).sum(-2)
        update = write_gate * v - erase
        recurrent = decayed + k.unsqueeze(-1) * update.unsqueeze(-2)
        o = (q.unsqueeze(-1) * recurrent).sum(-2)
        o = o * torch.rsqrt(o.square().mean(-1, keepdim=True) + c.gdn_norm_eps)
        gate = kernel.g_proj(x).view(batch, heads, dim)
        y = o * kernel.o_norm.weight * torch.sigmoid(gate)
        return kernel.o_proj(y.reshape(batch, 1, -1)), [recurrent, conv_q, conv_k, conv_v]

    def forward(self, input_ids, position, *states):
        transformer = self.model.transformer
        x = transformer["wte"](input_ids)
        cos, sin = self.rope_cos[position][:, None, None], self.rope_sin[position][:, None, None]
        remaining, new_states = list(states), []
        for block in transformer["h"]:
            h = block.ln_1(x)
            if block.full_attention:
                out, layer_states = self._attention(block.attn, h, cos, sin, *remaining[:2])
                remaining = remaining[2:]
            else:
                out, layer_states = self._gdn2(block.attn.kernel, h, *remaining[:4])
                remaining = remaining[4:]
            x = x + out
            x = x + block.mlp(block.ln_2(x))
            new_states += layer_states
        logits = self.model.lm_head(transformer["ln_f"](x))[:, 0]
        return (logits, *new_states)

def example_inputs(wrapper, batch_size=2, past_length=3):
    c = wrapper.config
    specs = state_specs(c.to_dict())
    states = [torch.zeros(batch_size, *(d or past_length for d in s.shape)) for s in specs]
    input_ids = torch.randint(0, c.vocab_size, (batch_size, 1))
    position = torch.full((batch_size,), past_length, dtype=torch.long)
    return (input_ids, position, *states), specs

def dynamic_shapes(specs):
    batch, past = torch.export.Dim("batch"), torch.export.Dim("past_length")
    state_shapes = tuple({0: batch, **({2: past} if s.shape[1] is None else {})} for s in specs)
    return ({0: batch}, {0: batch}, state_shapes)

def export(wrapper, output_path):
    args, specs = example_inputs(wrapper)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, args, str(output_path), dynamo=True, external_data=False,
        input_names=["input_ids", "position", *(f"state.{s.name}" for s in specs)],
        output_names=["logits", *(f"new_state.{s.name}" for s in specs)],
        dynamic_shapes=dynamic_shapes(specs),
    )
