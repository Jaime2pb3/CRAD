#!/usr/bin/env python3
"""Rung 4 — GPU preflight for the CRAD lab (Colab rebuild, 2026-10-10).

Usage:  python preflight_gpu.py configs/<config>.json

The last cheap check before captures and patching spend compute units.  It loads
the model EXACTLY as CRAD does (crad_base.RecogModel from the pinned src/) and
verifies, on this GPU and this software stack, the assumptions the capture and
the patch gates rely on.  BLOCKING checks stop the ladder (exit 1):

  B1  model entirely on the GPU (no CPU/disk offload), band fits n_layers
  B2  loaded commit == the manifest run.preflight resolved
  B3  crad_base.verify_hidden_capture passes
  B4  hidden states are float16 -> capture.npz stores phi fp16-exact
  B5  same seed, same process: traced generation reproduces tokens AND phi bit-exactly
      (the "unpatched run reproduces the captured turn" gate depends on this)
  B6  self-patch hook: replacing the residual after block l at step k with its own
      value leaves the next-token log-probs unchanged (< 1e-3 nats) -- the hook
      in patch_probe/patch_batch writes where phi is read, on THIS transformers

FINDING (recorded, not blocking):
  F1  plain generation == traced generation under the same seed
      (CRAD falsification criterion 5: an invasive measurement path)

It also measures seconds per traced turn so the notebook can estimate the cost of
the capture and of patch_batch from a measured number, not a projection.
Writes preflight/<UTC>_<gpu>.json.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))

import launch_common as lc  # noqa: E402
import run as runmod  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: preflight_gpu.py configs/<config>.json")
    cfg = json.loads(Path(sys.argv[1]).read_text())
    a = cfg["args"]
    manifest = runmod.preflight(cfg)
    v13, crad_base = lc.load_v13_module(runmod.SRC)
    lc.install_model_probe(crad_base)
    checks = []

    def check(code, name, ok, detail="", blocking=True):
        checks.append({"code": code, "name": name, "ok": bool(ok), "detail": str(detail), "blocking": blocking})
        tag = "ok" if ok else ("FAIL" if blocking else "FINDING")
        print(f"  [{tag}] {code} {name}{(' — ' + str(detail)) if detail else ''}", flush=True)

    t0 = time.time()
    v13.seed_everything(int(a["seed"]))
    mw = crad_base.RecogModel(a["model"], use_4bit=not a["no4bit"])
    torch = mw.torch
    load_s = time.time() - t0
    rt = lc.runtime_info(t0)
    print(f"[preflight] loaded in {load_s:.1f}s · {rt.get('quantization')} · {rt.get('device')} "
          f"· GPU {rt['gpu'].get('name')}", flush=True)

    on_gpu = (mw.device.type == "cuda") or lc.is_mock()
    check("B1", "model fully on the accelerator, no CPU/disk offload", on_gpu and not rt.get("offload"),
          f"device {mw.device}, device_map {rt.get('device_map')}")
    check("B1", "CRAD band and patch layers fit the model", int(a["layer-hi"]) <= mw.n_layers,
          f"band [{a['layer-lo']},{a['layer-hi']}), n_layers {mw.n_layers}")
    check("B2", "loaded commit == manifest", manifest is None or rt.get("model_commit") == manifest["revision"],
          f"loaded {rt.get('model_commit')} · manifest {(manifest or {}).get('revision')}")
    check("B3", "crad_base.verify_hidden_capture", crad_base.verify_hidden_capture(mw))

    temp = runmod.effective_temps(a)["analysis"]
    n = int(a["n-tokens"])
    lo, hi = int(a["layer-lo"]), int(a["layer-hi"])
    ramp = crad_base.build_real_ramp(a["secret"], n_turns=8)
    hist = [{"role": "system", "content": ramp.system}, {"role": "user", "content": ramp.turns_R[0]}]

    ids = mw._ids(hist)
    with torch.no_grad():
        out = mw.lm.generate(**ids, max_new_tokens=4, do_sample=False, output_hidden_states=True,
                             return_dict_in_generate=True, pad_token_id=mw.tok.eos_token_id)
    hdt = out.hidden_states[0][1].dtype
    check("B4", "hidden states are float16 (fp16-exact capture)", hdt == torch.float16, str(hdt))

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def traced(seed):
        v13.seed_everything(seed)
        return mw.gen_with_trace(hist, n=n, temp=temp, layer_lo=lo, layer_hi=hi, turn_idx=0, global_start=0)

    v13.seed_everything(777)
    plain = mw.gen(hist, n=n, temp=temp)
    sync(); t = time.time()
    tr1 = traced(777)
    sync(); sec_turn = time.time() - t
    tr2 = traced(777)
    toks1, toks2 = [f["token"] for f in tr1["frames"]], [f["token"] for f in tr2["frames"]]
    phi_eq = len(toks1) == len(toks2) and all(np.array_equal(x["phi"], y["phi"])
                                             for x, y in zip(tr1["frames"], tr2["frames"]))
    check("B5", f"traced generation reproducible at T={temp} (tokens + phi bit-exact)",
          toks1 == toks2 and phi_eq, f"{len(toks1)} tokens")
    check("F1", "plain == traced generation, same seed (falsification criterion 5)",
          plain == tr1["text"], "" if plain == tr1["text"] else f"plain {plain[:60]!r} vs traced {tr1['text'][:60]!r}",
          blocking=False)

    # B6: self-patch at the band edge and at the deepest non-final layer (HF: last hidden state is post-norm)
    layers = sorted({lo, mw.n_layers - 2})
    step_k = 3
    for l in layers:
        with torch.no_grad():
            base = mw.lm.generate(**ids, max_new_tokens=step_k + 2, do_sample=False, output_hidden_states=True,
                                  output_logits=True, return_dict_in_generate=True,
                                  pad_token_id=mw.tok.eos_token_id)
        h_own = base.hidden_states[step_k][l + 1][0, -1].clone()
        lp_base = base.logits[step_k][0].float().log_softmax(-1)
        state = {"n": 0}

        def hook(_m, _i, o, h_own=h_own):
            k = state["n"]; state["n"] += 1
            if k != step_k:
                return o
            h = o[0] if isinstance(o, tuple) else o
            h[:, -1, :] = h_own.to(h.dtype)
            return o

        hd = mw.lm.model.layers[l].register_forward_hook(hook)
        try:
            with torch.no_grad():
                pat = mw.lm.generate(**ids, max_new_tokens=step_k + 2, do_sample=False, output_logits=True,
                                     return_dict_in_generate=True, pad_token_id=mw.tok.eos_token_id)
        finally:
            hd.remove()
        diff = float((pat.logits[step_k][0].float().log_softmax(-1) - lp_base).abs().max())
        check("B6", f"self-patch hook at layer {l}, step {step_k}", diff < 1e-3 and state["n"] > step_k,
              f"max|Δ log p| = {diff:.2e}, hook calls {state['n']}")

    gpu = rt["gpu"]
    gname = re.sub(r"[^A-Za-z0-9]+", "-", gpu.get("name") or ("mock-cpu" if lc.is_mock() else "cpu")).strip("-")
    report = {"config": cfg["name"], "manifest": manifest, "runtime": lc.runtime_info(t0),
              "load_seconds": round(load_s, 2), "analysis_temp": temp, "n_tokens": n,
              "sec_per_traced_turn": round(sec_turn, 3), "tokens_in_timed_turn": len(toks1),
              "tokens_per_second": round(len(toks1) / max(sec_turn, 1e-9), 2), "checks": checks,
              "blocking_failures": [c for c in checks if c["blocking"] and not c["ok"]],
              "findings": [c for c in checks if not c["blocking"] and not c["ok"]]}
    out_dir = LAB / "preflight"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = out_dir / f"{stamp}_{gname}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"[preflight] {sec_turn:.2f}s per traced turn of {len(toks1)} tokens "
          f"({report['tokens_per_second']} tok/s) · wrote {path}")
    if report["blocking_failures"]:
        print("[preflight] RED: a blocking check failed. Do NOT run captures on this runtime.")
        return 1
    print("[preflight] GREEN" + (" (with findings, see above)" if report["findings"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
