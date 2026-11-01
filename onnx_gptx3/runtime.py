# SPDX-License-Identifier: Apache-2.0
# Adapted from ahmadw/GPT-X3-150M-ONNX, revision 695dd12882449179a75845635adfd4b526e89b64.
# Upstream: https://huggingface.co/ahmadw/GPT-X3-150M-ONNX
# Modifications: reusable model wrapper and runtime; loading/export CLI removed.

import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import onnxruntime as ort
MODEL_DIR=Path(__file__).resolve().parent
DEFAULT_ONNX_PATH=MODEL_DIR/"model.onnx"

@dataclass(frozen=True)
class StateSpec:
    name: str
    shape: tuple  # batch dimension excluded; None marks the growing KV axis

def load_config(model_dir=MODEL_DIR):
    return json.loads((model_dir / "config.json").read_text())

def is_full_attention_layer(config, index):
    return (index + 1) % (config["gdn_layers_per_gqa"] + 1) == 0

def state_specs(config):
    """Ordered recurrent/KV state tensors shared by the exporter and the runtime."""
    hidden, gdn_dim = config["hidden_size"], config["gdn_head_dim"]
    kv_shape = (config["num_key_value_heads"], None, config["head_dim"])
    conv_shape = (hidden, config["conv_size"] - 1)
    specs = []
    for i in range(config["num_hidden_layers"]):
        if is_full_attention_layer(config, i):
            specs += [StateSpec(f"{i}.key", kv_shape), StateSpec(f"{i}.value", kv_shape)]
            continue
        specs.append(StateSpec(f"{i}.recurrent", (hidden // gdn_dim, gdn_dim, gdn_dim)))
        specs += [StateSpec(f"{i}.conv_{n}", conv_shape) for n in "qkv"]
    return specs

def initial_states(specs, batch_size):
    return {f"state.{s.name}": np.zeros((batch_size, *(d or 0 for d in s.shape)), np.float32)
            for s in specs}

class GPTX3OnnxModel:
    def __init__(self, onnx_path=DEFAULT_ONNX_PATH, model_dir=MODEL_DIR, providers=None, session_options=None):
        self.config = load_config(Path(model_dir))
        self.specs = state_specs(self.config)
        self.session = ort.InferenceSession(str(onnx_path), sess_options=session_options,
                                            providers=providers or ["CPUExecutionProvider"])

    def new_states(self, batch_size=1):
        return initial_states(self.specs, batch_size)

    def step(self, token_ids, position, states):
        """Feed one token per sequence; returns logits [batch, vocab] and the updated states."""
        feeds = {"input_ids": np.asarray(token_ids, np.int64).reshape(-1, 1),
                 "position": np.full(len(token_ids), position, np.int64), **states}
        logits, *new_states = self.session.run(None, feeds)
        return logits, {f"state.{s.name}": t for s, t in zip(self.specs, new_states)}

    def prefill(self, token_ids):
        states = self.new_states()
        logits = None
        for position, token in enumerate(token_ids):
            logits, states = self.step([token], position, states)
        return logits, states
