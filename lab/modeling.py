"""Shared tokenizer/prompt and text-model loading for training and evaluation."""
import hashlib
import sys
from pathlib import Path

PLAIN_TEMPLATE = "{{ bos_token or '' }}{% for m in messages %}{{ m['role'] | capitalize }}: {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"
CHATML_TEMPLATE = "{{ bos_token or '' }}{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


def configure_tokenizer(tokenizer, prompt_style='auto'):
    if prompt_style not in ('auto', 'native', 'plain', 'task-chatml'):
        raise ValueError(f'Unknown prompt style: {prompt_style}')
    if prompt_style == 'plain' or (prompt_style == 'auto' and not tokenizer.chat_template):
        tokenizer.chat_template = PLAIN_TEMPLATE
    elif prompt_style == 'task-chatml':
        tokenizer.chat_template = CHATML_TEMPLATE
    elif not tokenizer.chat_template:
        raise ValueError('No native chat template; choose plain explicitly or use auto')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer requires EOS and padding tokens')
    return tokenizer


def template_hash(tokenizer):
    return hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()


def load_model(path, dtype, **kwargs):
    """Use the documented Qwen3.5 text-only class; other models use AutoCausalLM."""
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(path, trust_remote_code=True,
                                       **{k: v for k, v in kwargs.items() if k in ('revision', 'local_files_only')})
    if config.model_type == 'qwen3_5':
        from transformers import Qwen3_5ForCausalLM
        model_class = Qwen3_5ForCausalLM
    elif config.model_type == 'spark2_5':
        # Spark's pinned remote code uses the older tied-weight and mask APIs.
        # Adapt interface metadata only; tensor operations and weights stay intact.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        model_class = get_class_from_dynamic_module(config.auto_map['AutoModelForCausalLM'], path,
                                                     **{k: v for k, v in kwargs.items() if k == 'revision'})
        model_class._tied_weights_keys = {'lm_head.weight': 'model.embedding.weight'}
        module = sys.modules[model_class.__module__]
        def mask_adapter(function):
            def compatible(*args, **kw):
                if 'input_embeds' in kw:
                    kw['inputs_embeds'] = kw.pop('input_embeds')
                kw.pop('cache_position', None)
                return function(*args, **kw)
            compatible._gundem_compat = True
            return compatible
        for name in ('create_causal_mask', 'create_sliding_window_causal_mask'):
            function = getattr(module, name)
            if not getattr(function, '_gundem_compat', False):
                setattr(module, name, mask_adapter(function))
    else:
        model_class = AutoModelForCausalLM
    kwargs.pop('trust_remote_code', None)
    model, info = model_class.from_pretrained(path, dtype=dtype, trust_remote_code=True,
                                            output_loading_info=True, **kwargs)
    if info.get('missing_keys') or info.get('mismatched_keys'):
        raise ValueError(f'Checkpoint weights were not fully loaded: {info}')
    return model


def adapt_direct_weight_lora(model):
    """Limite uses F.linear(module.weight), bypassing PEFT's Linear.forward.

    Expose the differentiable effective weight for that architecture. Normal PEFT
    forward still applies its low-rank branch once; base_layer is unchanged.
    Saved adapters remain standard PEFT adapters and can be merged normally.
    """
    if getattr(model.config, 'model_type', '') != 'limite':
        return model
    from peft.tuners.lora.layer import Linear

    class WeightAwareLinear(Linear):
        @property
        def weight(self):
            weight = self.get_base_layer().weight
            if not self.disable_adapters and not self.merged:
                for adapter in self.active_adapters:
                    if adapter in self.lora_A:
                        weight = weight + self.get_delta_weight(adapter)
            return weight

    for module in model.modules():
        if isinstance(module, Linear):
            module.__class__ = WeightAwareLinear
    return model
