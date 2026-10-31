"""A common causal, assistant-token-normalized objective for custom architectures."""
LOSS_CONTRACT = 'assistant-causal-token-mean-v1'


def assistant_causal_loss(outputs, labels, num_items_in_batch=None):
    import torch.nn.functional as F
    logits = outputs.logits[..., :-1, :].contiguous().float()
    targets = labels[..., 1:].contiguous().to(logits.device)
    total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                            ignore_index=-100, reduction='sum')
    denominator = num_items_in_batch if num_items_in_batch is not None else targets.ne(-100).sum()
    return total / denominator
