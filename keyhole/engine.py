"""One forward engine for behaviour and patching.

A prompt = shared PREFIX (KV cache computed once per group) + SUFFIX rows run in a batch.
Rows are right-padded: with causal attention a padded tail never changes earlier positions,
so logits and states at real positions are exact. Hooks read or overwrite the residual
stream (block outputs) and the attention / MLP sublayer outputs at given (row, position)s.
"""
import copy
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from keyhole import config as C


@dataclass
class Plan:
    """What the hooks do during one forward. Keys of patch/capture dicts are
    ("r", layer) for the residual stream and ("attn"|"mlp", layer) for sublayer outputs."""
    patch: dict = field(default_factory=dict)     # key -> (rows, pos, values[n, d])
    capture: dict = field(default_factory=dict)   # key -> (rows, pos)
    got: dict = field(default_factory=dict)       # key -> captured values[n, d]


def find_layers(model, n_layers):
    found = [(n, m) for n, m in model.named_modules()
             if isinstance(m, torch.nn.ModuleList) and n.endswith("layers") and len(m) == n_layers]
    found.sort(key=lambda x: ("language_model" not in x[0], len(x[0])))
    if not found:
        raise RuntimeError("decoder layers not found")
    return found[0][1]


def sublayers(layer):
    """(attention-output module, mlp-output module, tuple_output_for_attn)."""
    if hasattr(layer, "post_feedforward_layernorm"):      # post-norm blocks: OLMo 3, Gemma 4
        return layer.post_attention_layernorm, layer.post_feedforward_layernorm, False
    return layer.self_attn, layer.mlp, True                # pre-norm blocks: Qwen 3


def repeat_cache(cache, B):
    """Copy a batch-1 cache into a batch-B cache without touching the original."""
    c = copy.copy(cache)
    c.layers = []
    for layer in cache.layers:
        l2 = copy.copy(layer)
        if getattr(layer, "keys", None) is not None and layer.keys.numel():
            l2.keys = layer.keys.repeat_interleave(B, dim=0) if B > 1 else layer.keys.clone()
            l2.values = layer.values.repeat_interleave(B, dim=0) if B > 1 else layer.values.clone()
        c.layers.append(l2)
    return c


class Engine:
    def __init__(self, key, device=None, dtype=torch.bfloat16, model=None, tok=None):
        m = C.MODELS.get(key, {})
        self.key = key
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = tok or AutoTokenizer.from_pretrained(m["repo"], revision=m["rev"])
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(m["repo"], revision=m["rev"], dtype=dtype,
                                                         attn_implementation="sdpa")
        self.model = model.to(self.device).eval()
        tc = self.model.config.get_text_config()
        self.L = tc.num_hidden_layers
        self.d = tc.hidden_size
        self.layers = find_layers(self.model, self.L)
        self.plan = None
        self._hooks = []
        for i, layer in enumerate(self.layers):
            self._hooks.append(layer.register_forward_hook(self._make_hook(("r", i), False)))
            attn, mlp, tup = sublayers(layer)
            self._hooks.append(attn.register_forward_hook(self._make_hook(("attn", i), tup)))
            self._hooks.append(mlp.register_forward_hook(self._make_hook(("mlp", i), False)))
        self.pad_id = 0   # any valid id: padded positions are never read
        self._stop = None

    # ------------------------------------------------------------------ hooks
    def _make_hook(self, key, tuple_out):
        def hook(module, args, out):
            p = self.plan
            if p is None:
                return None
            h = out[0] if (tuple_out or isinstance(out, tuple)) else out
            if key in p.capture:
                rows, pos = p.capture[key]
                p.got[key] = h[rows, pos].detach().clone()
            if key in p.patch:
                rows, pos, vals = p.patch[key]
                h[rows, pos] = vals.to(h.dtype)
            return None
        return hook

    def t(self, x, dtype=torch.long):
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    # --------------------------------------------------------------- forwards
    @torch.inference_mode()
    def prefix_cache(self, prefix_ids):
        self.plan = None
        out = self.model(input_ids=self.t([prefix_ids]), use_cache=True, logits_to_keep=1)
        return out.past_key_values

    @torch.inference_mode()
    def forward(self, cache, rows, keep, plan=None):
        """rows: list of token lists (suffixes). keep: sorted list of positions whose logits to return.
        Returns logits [B, len(keep), V] (float32) and the plan with captures."""
        B, T = len(rows), max(len(r) for r in rows)
        ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
        kv = repeat_cache(cache, B) if cache is not None else None
        self.plan = plan
        try:
            out = self.model(input_ids=ids.to(self.device), past_key_values=kv, use_cache=kv is not None,
                             logits_to_keep=self.t(keep))
        finally:
            self.plan = None
        return out.logits.float(), plan

    def stop_ids(self):
        if self._stop is None:
            eos = self.model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
            vocab = self.tok.convert_ids_to_tokens(list(range(len(self.tok))))
            nl = {i for i, t in enumerate(vocab) if t is not None and ("\n" in t or "Ċ" in t or "<0x0A>" in t)}
            self._stop = eos | nl
        return self._stop

    @torch.inference_mode()
    def generate(self, cache, rows, max_new=C.MAX_NEW_TOKENS):
        """Greedy decoding for rows of EQUAL length (no padding). Returns list of generated id lists."""
        assert len({len(r) for r in rows}) == 1
        B = len(rows)
        kv = repeat_cache(cache, B) if cache is not None else None
        stop = self.stop_ids()
        ids = self.t(rows)
        out_ids = [[] for _ in range(B)]
        done = [False] * B
        self.plan = None
        for step in range(max_new):
            out = self.model(input_ids=ids, past_key_values=kv, use_cache=True, logits_to_keep=1)
            kv = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1).tolist()
            for i, tkn in enumerate(nxt):
                if not done[i]:
                    out_ids[i].append(tkn)
                    if tkn in stop:
                        done[i] = True
            if all(done):
                break
            ids = self.t([[x] for x in nxt])
        return out_ids

    def close(self):
        for h in self._hooks:
            h.remove()
