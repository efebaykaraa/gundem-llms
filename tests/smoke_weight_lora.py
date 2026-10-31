"""Check LoRA on direct F.linear(weight) paths, without any model download."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model
from lab.modeling import adapt_direct_weight_lora

torch.manual_seed(42)
model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                                    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
model.config.model_type = 'limite'
model = get_peft_model(model, LoraConfig(target_modules=['q_proj'], r=2, lora_alpha=4, lora_dropout=0))
model = adapt_direct_weight_lora(model)
layer = model.base_model.model.model.layers[0].self_attn.q_proj
with torch.no_grad():
    layer.lora_B['default'].weight.fill_(.1)
x = torch.randn(2, 3, 16)
direct = torch.nn.functional.linear(x, layer.weight)
ordinary = layer(x)
assert torch.allclose(direct, ordinary, atol=1e-6)
direct.square().sum().backward()
assert layer.lora_A['default'].weight.grad.abs().sum() > 0
assert layer.lora_B['default'].weight.grad.abs().sum() > 0
before = direct.detach()
merged = model.merge_and_unload()
actual = merged.model.layers[0].self_attn.q_proj(x)
assert torch.allclose(before, actual, atol=1e-6)
print('PASS: direct-weight LoRA participates in forward, backward and standard PEFT merge.')
