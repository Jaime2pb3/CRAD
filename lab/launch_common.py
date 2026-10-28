#!/usr/bin/env python3
"""Shared plumbing for crad_launch.py and capture_launch.py (Colab rebuild, 2026-10-10).

TwoQuarks Research — Luis Jaime Ledesma Perez

Rebuilt because the original lab infrastructure stayed on the RTX 3060 machine.
Contract with the instrument: CRAD_v1.3.py and crad_base.py are loaded UNMODIFIED
from src/ (hash-pinned in src/PINNED.json, checked by run.preflight).  Nothing here
changes what CRAD computes; it only observes:

  - which RecogModel instance was built (so runtime.json can say what was loaded),
  - what the run cost (wall time) and on what software/hardware it ran.

MOCK MODE (rung 3 of the GPU ladder, CPU only)
  CRAD_MOCK_MODEL=1 replaces RecogModel.__init__ with a tiny random fp16 network
  that honours the same interface (tok, lm.generate, lm.model.layers, hooks).
  It exists to test the capture/contract/patch wiring end to end without a GPU.
  Mock runs are stamped "mock": true in runtime.json and fingerprint, and
  run.consistency() flags them, so they can never pass as real data.
"""
from __future__ import annotations

import importlib.metadata as md
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

MOCK_ENV = "CRAD_MOCK_MODEL"
STATE = {"mw": None, "model_id": None, "use_4bit": None}


def is_mock() -> bool:
    return os.environ.get(MOCK_ENV, "") == "1"


def load_v13_module(src: Path):
    """Import the pinned CRAD_v1.3.py (dotted filename) exactly as capture.load_v13 does."""
    src = Path(src).resolve()
    sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("crad_v1_3_pinned", src / "CRAD_v1.3.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crad_v1_3_pinned"] = mod
    spec.loader.exec_module(mod)
    import crad_base  # importable now: src is on sys.path
    return mod, crad_base


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("torch", "transformers", "accelerate", "bitsandbytes", "numpy",
                "huggingface_hub", "safetensors", "tokenizers"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    out["colab_release"] = os.environ.get("COLAB_RELEASE_TAG")
    return out


def gpu_info() -> dict:
    try:
        import torch
    except ImportError:
        return {"available": False}
    if not torch.cuda.is_available():
        return {"available": False, "cuda_build": getattr(torch.version, "cuda", None)}
    p = torch.cuda.get_device_properties(0)
    info = {"available": True, "name": p.name, "capability": f"{p.major}.{p.minor}",
            "total_memory_gb": round(p.total_memory / 1e9, 2), "cuda_build": torch.version.cuda,
            "count": torch.cuda.device_count()}
    try:
        import subprocess
        info["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        info["driver"] = None
    return info


def install_model_probe(crad_base) -> None:
    """Wrap RecogModel.__init__ to remember the instance (and swap in the mock if asked)."""
    cls = crad_base.RecogModel
    orig_init = cls.__init__

    def init(self, model_id, use_4bit=True):
        STATE["model_id"], STATE["use_4bit"] = model_id, use_4bit
        if is_mock():
            _mock_init(self, model_id, use_4bit)
        else:
            orig_init(self, model_id, use_4bit=use_4bit)
        STATE["mw"] = self

    cls.__init__ = init


def runtime_info(t0: float, error: str | None = None) -> dict:
    mw = STATE["mw"]
    rt = {"wall_seconds": round(time.time() - t0, 3), "error": error, "mock": is_mock(),
          "model_id": STATE["model_id"], "versions": versions(), "gpu": gpu_info()}
    if mw is None:
        return rt
    lm = mw.lm
    dm = {str(k): str(v) for k, v in (getattr(lm, "hf_device_map", None) or {}).items()}
    if is_mock():
        quant = "mock"
    elif getattr(lm, "is_loaded_in_4bit", False):
        quant = "nf4"
    elif getattr(lm, "is_loaded_in_8bit", False):
        quant = "int8"
    else:
        quant = "fp16"
    try:
        param_dtype = str(next(lm.parameters()).dtype)
    except Exception:
        param_dtype = None
    rt.update({
        "model_commit": _loaded_commit(lm)[0],
        "model_commit_source": _loaded_commit(lm)[1],
        "n_layers": int(mw.n_layers),
        "device": str(mw.device),
        "device_map": dm,
        "offload": any(v in ("cpu", "disk") for v in dm.values()),
        "quantization": quant,
        "requested_4bit": STATE["use_4bit"],
        "param_dtype": param_dtype,
        "attn_implementation": getattr(lm.config, "_attn_implementation", None),
    })
    try:
        import torch
        if torch.cuda.is_available():
            rt["cuda_max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return rt


def write_json(path: Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str) + "\n")


# --------------------------------------------------------------- mock model --
def _mock_init(self, model_id, use_4bit):
    """Tiny random fp16 decoder with the RecogModel surface. CPU, deterministic weights."""
    import torch
    from torch import nn

    class Tok:
        """Character-level tokenizer; id 0 = eos."""
        eos_token_id = 0
        chars = [chr(c) for c in range(32, 127)] + ["\n"]
        stoi = {c: i + 1 for i, c in enumerate(chars)}

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            text = "".join(f"<{m['role']}>{m['content']}\n" for m in messages)
            return text + ("<assistant>" if add_generation_prompt else "")

        def encode(self, text):
            return [self.stoi.get(c, self.stoi["?"]) for c in text]

        def __call__(self, text, return_tensors=None, add_special_tokens=True):
            ids = self.encode(text)
            if return_tensors == "pt":
                return _Enc({"input_ids": torch.tensor([ids]),
                             "attention_mask": torch.ones(1, len(ids), dtype=torch.long)})
            return {"input_ids": ids}

        def decode(self, ids, skip_special_tokens=False):
            ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            return "".join(self.chars[i - 1] for i in ids if i > 0)

    class _Enc(dict):
        def to(self, device):
            return _Enc({k: v.to(device) for k, v in self.items()})

    class Block(nn.Module):
        def __init__(self, d, g):
            super().__init__()
            self.lin = nn.Linear(d, d)
            with torch.no_grad():
                self.lin.weight.copy_(torch.randn(d, d, generator=g) * (0.9 / d ** 0.5))
                self.lin.bias.copy_(torch.randn(d, generator=g) * 0.05)

        def forward(self, h):
            return (h + torch.tanh(self.lin(h)),)          # HF-style tuple output

    class LM(nn.Module):
        def __init__(self, vocab=len(Tok.chars) + 1, d=32, n_layers=28):
            super().__init__()
            g = torch.Generator().manual_seed(20261010)
            self.embed = nn.Embedding(vocab, d)
            self.pos = nn.Embedding(8192, d)
            with torch.no_grad():
                self.embed.weight.copy_(torch.randn(vocab, d, generator=g))
                self.pos.weight.copy_(torch.randn(8192, d, generator=g) * 0.1)
            self.model = SimpleNamespace(layers=nn.ModuleList([Block(d, g) for _ in range(n_layers)]))
            self._layers = self.model.layers                 # registered as submodule
            self.head = nn.Linear(d, vocab, bias=False)
            with torch.no_grad():
                self.head.weight.copy_(torch.randn(vocab, d, generator=g) * 0.6)
            self.config = SimpleNamespace(num_hidden_layers=n_layers, _commit_hash="mock-00000000",
                                          _attn_implementation="mock")

        def _forward(self, ids):
            pos = torch.arange(ids.shape[1]) % 8192
            h = (self.embed(ids) + self.pos(pos)[None]).to(torch.float16)
            hs = [h]
            for blk in self.model.layers:
                out = blk(h)
                h = out[0] if isinstance(out, tuple) else out
                hs.append(h)
            hn = h / (h.float().pow(2).mean(-1, keepdim=True).sqrt().to(h.dtype) + 1e-3)
            hs[-1] = hn                                       # HF: last hidden state is post-norm
            return self.head(hn.float()), tuple(hs)

        @torch.no_grad()
        def generate(self, input_ids=None, attention_mask=None, max_new_tokens=8, do_sample=False,
                     temperature=1.0, top_p=1.0, output_hidden_states=False,
                     return_dict_in_generate=False, pad_token_id=0, output_logits=False, **_):
            seq = input_ids.clone()
            all_hs, all_logits = [], []
            for step in range(int(max_new_tokens)):
                logits, hs = self._forward(seq)
                last = logits[:, -1, :]
                all_logits.append(last.clone())
                if output_hidden_states:
                    all_hs.append(hs if step == 0 else tuple(x[:, -1:, :] for x in hs))
                if do_sample:
                    probs = torch.softmax(last / max(float(temperature), 1e-6), dim=-1)
                    sp, si = torch.sort(probs, descending=True)
                    keep = torch.cumsum(sp, -1) - sp < float(top_p)
                    sp = sp * keep
                    nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1))
                else:
                    nxt = last.argmax(-1, keepdim=True)
                seq = torch.cat([seq, nxt], dim=1)
                if int(nxt) == int(pad_token_id):
                    break
            if return_dict_in_generate:
                return SimpleNamespace(sequences=seq, hidden_states=tuple(all_hs) if output_hidden_states else None,
                                       logits=tuple(all_logits) if output_logits else None)
            return seq

    self.torch = torch
    self.tok = Tok()
    print(f"[load] MOCK model standing in for {model_id} (CPU, random fp16, NOT real data)", flush=True)
    self.lm = LM().eval()
    self.lm.to(torch.float16)
    self.lm.embed.float(); self.lm.pos.float(); self.lm.head.float()
    for p in self.lm.parameters():
        p.requires_grad_(False)
    self.device = torch.device("cpu")
    self.n_layers = int(self.lm.config.num_hidden_layers)
    print(f"[load] OK: {self.n_layers} layers on {self.device}", flush=True)


def _loaded_commit(lm):
    """Commit of the weights actually loaded.  transformers < 5 stored it in config._commit_hash;
    5.x dropped it (resolves the revision once per load), so read the HF cache the loader used:
    refs/main -> snapshots/<sha>/config.json."""
    c = getattr(lm.config, "_commit_hash", None)
    if c:
        return c, "config._commit_hash"
    try:
        from huggingface_hub import try_to_load_from_cache
        f = try_to_load_from_cache(STATE["model_id"] or lm.config._name_or_path, "config.json", revision="main")
        if isinstance(f, str):
            return Path(f).parent.name, "hf_cache refs/main -> snapshot"
    except Exception:
        pass
    return None, None
