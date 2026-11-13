#!/usr/bin/env python3
"""H4: the composed model with a per-layer deliberation dial.

W_l(beta) = Thinking_l + (beta_l - 1) * P_l   for the 36 decoder layers (all 7 matrices and the norm vectors in P);
beta_36 drives the embedding table and the tied output head (they are one tensor in Qwen3-4B).
beta = 1 everywhere is the Thinking model exactly; beta = 0 everywhere is D = Thinking - P.
P is held factored (P x = A (B x)), so the extra cost is small. beta is read at call time from self.beta, so a
hypernetwork's output (a tensor that requires grad) can be set before each forward and gradients reach it.
"""
import os
import re
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cutconfig import ROOT, THINKING_DIR as T_DIR  # noqa: E402

N_LAYERS = 36
EMB = 36  # index of the embedding / output-head dial


class Composed:
    def __init__(self, dev, checkpointing=True):
        self.dev = dev
        self.model = AutoModelForCausalLM.from_pretrained(T_DIR, dtype=torch.bfloat16).to(dev)
        self.model.config.use_cache = False
        for p in self.model.parameters():
            p.requires_grad_(False)
        if checkpointing:
            # non-reentrant: gradients reach beta through the hooks even though every model weight is frozen
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.train()  # HF checkpointing only runs in train mode; Qwen3 has no dropout
        else:
            self.model.eval()
        self.beta = torch.ones(N_LAYERS + 1, device=dev)
        P = safe_open(f"{ROOT}/hyper/P_q80.safetensors", framework="pt", device="cpu")
        keys = set(P.keys())
        self.hooks = []
        mods = dict(self.model.named_modules())
        n_lin = n_norm = 0
        for k in sorted(keys):
            if k.endswith(".A"):
                name = k[:-2]
                mod = mods[name[:-len(".weight")]]
                layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
                A = P.get_tensor(k).to(dev)
                B = P.get_tensor(name + ".B").to(dev)
                self.hooks.append(mod.register_forward_hook(self._lin_hook(layer, A, B)))
                n_lin += 1
            elif not k.endswith(".B") and "layers." in k:  # a norm vector
                mod = mods[k[:-len(".weight")]]
                layer = int(re.search(r"layers\.(\d+)\.", k).group(1))
                self.hooks.append(mod.register_forward_hook(self._norm_hook(layer, P.get_tensor(k).to(dev))))
                n_norm += 1
        if "model.embed_tokens.weight" in keys:
            Pe = P.get_tensor("model.embed_tokens.weight").to(dev)
            self.hooks.append(self.model.model.embed_tokens.register_forward_hook(self._emb_hook(Pe)))
            self.hooks.append(self.model.lm_head.register_forward_hook(self._head_hook(Pe)))
        self.summary = f"hooks: {n_lin} matrices, {n_norm} norm vectors, embedding+head {'yes' if 'model.embed_tokens.weight' in keys else 'no'}"

    def _scale(self, i, dtype):
        return (self.beta[i] - 1.0).to(dtype)

    def _lin_hook(self, layer, A, B):
        def hook(mod, inp, out):
            x = inp[0]
            return out + self._scale(layer, out.dtype) * F.linear(F.linear(x, B), A)
        return hook

    def _norm_hook(self, layer, p):
        def hook(mod, inp, out):
            x = inp[0]
            h = x.float()
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + mod.variance_epsilon)
            return out + self._scale(layer, out.dtype) * (p.to(out.dtype) * h.to(out.dtype))
        return hook

    def _emb_hook(self, Pe):
        def hook(mod, inp, out):
            return out + self._scale(EMB, out.dtype) * F.embedding(inp[0], Pe)
        return hook

    def _head_hook(self, Pe):
        def hook(mod, inp, out):
            return out + self._scale(EMB, out.dtype) * F.linear(inp[0], Pe)
        return hook

    def set_beta(self, beta):
        self.beta = beta
