"""Prove the common loss matches HF causal loss and correct accumulated gradients."""
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers.loss.loss_utils import ForCausalLMLoss
from lab.loss import assistant_causal_loss

torch.manual_seed(42)
labels = torch.tensor([[-100, -100, 3, 4, 5, 2], [-100, -100, -100, 6, 2, -100]])
logits = torch.randn(2, 6, 16, requires_grad=True)
count = labels[..., 1:].ne(-100).sum()
common = assistant_causal_loss(SimpleNamespace(logits=logits), labels, count)
reference = ForCausalLMLoss(logits, labels, 16, num_items_in_batch=count)
assert torch.allclose(common, reference, atol=1e-6)
joint_gradient = torch.autograd.grad(common, logits, retain_graph=True)[0]
accumulated = sum(assistant_causal_loss(SimpleNamespace(logits=logits[i:i+1]), labels[i:i+1], count)
                  for i in range(2))
accumulated_gradient = torch.autograd.grad(accumulated, logits)[0]
assert torch.allclose(joint_gradient, accumulated_gradient, atol=1e-6)
print('PASS: common loss equals HF causal loss; masked-token gradient accumulation is exact.')
