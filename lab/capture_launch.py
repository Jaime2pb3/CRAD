#!/usr/bin/env python3
"""Capture launcher used by capture.py (Colab rebuild, 2026-10-10).

Usage (capture.py builds this line):
  python capture_launch.py SRC RUNTIME_JSON CAPTURE_NPZ S_LO S_HI -- <explicit CRAD_v1.3 CLI>
  python capture_launch.py --selftest            (CPU, no model: writer + capture.check_capture)

Runs the pinned, UNMODIFIED CRAD_v1.3.main() and records every traced frame on the
layer SUPERSET [S_LO, S_HI) without changing what CRAD sees:

  * RecogModel.gen_with_trace is called with the superset; the frames CRAD receives
    are the same arrays sliced back to its band [lo, hi).  Slicing a stack of the
    same .float() vectors is bit-identical to tracing the band directly, so CRAD's
    calibration, events and classifications are unchanged (check_capture verifies
    this at value level: recomputed geometry must equal CRAD's exactly).
  * lm.generate is spied during each traced call only to read the emitted token ids
    (gen_with_trace does not return them) and the hidden-state dtype.
  * The two CRAD call sites that own the frames are wrapped to tag them:
      build_calibration -> run_context_branch   kind=0, branch N/L/R by its turn list
      main              -> run_pressure_draw    kind=1, draw = CRAD's draw index

capture.npz (allow_pickle=False):
  meta           json string: superset, crad_band, phi_dtype, fp16_exact, hidden_dtypes, ...
  kind           int8   0 calibration, 1 analysis
  branch         int8   0 N, 1 L, 2 R (calibration), -1 (analysis)
  draw           int32  analysis draw index (calibration: index within its branch)
  turn, token_in_turn, global_token   int32, CRAD's own coordinates
  token          str    decoded token text (CRAD's own)
  token_id       int64  emitted token id (-1 if unknown)
  phi            [n, S_HI-S_LO, d]  float16 when the model's hidden states were fp16 (exact),
                                    otherwise float32
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import launch_common as lc

LAUNCHER_VERSION = "capture_launch colab-0.1"
BRANCH_CODE = {"N": 0, "L": 1, "R": 2}


class Recorder:
    def __init__(self, s_lo: int, s_hi: int, band: tuple[int, int]):
        if not (s_lo <= band[0] < band[1] <= s_hi):
            raise SystemExit(f"[capture] band {band} outside superset [{s_lo},{s_hi})")
        self.s_lo, self.s_hi, self.band = s_lo, s_hi, band
        self.cols = {k: [] for k in ("kind", "branch", "draw", "turn", "token_in_turn",
                                     "global_token", "token", "token_id")}
        self.phi: list[np.ndarray] = []
        self.fp16_exact = True
        self.hidden_dtypes: set[str] = set()
        self.ctx = None
        self.cal_counter = {0: 0, 1: 0, 2: 0}
        self.n_analysis = 0

    def record(self, frame: dict, token_id: int) -> None:
        if self.ctx is None:
            raise SystemExit("[capture] traced generation outside a known CRAD context "
                             "(not calibration, not an analysis draw); refusing to guess its role")
        phi = np.asarray(frame["phi"], dtype=np.float32)
        if phi.shape[0] != self.s_hi - self.s_lo:
            raise SystemExit(f"[capture] frame has {phi.shape[0]} layers, superset needs {self.s_hi - self.s_lo}")
        f16 = phi.astype(np.float16)
        exact = bool(np.array_equal(f16.astype(np.float32), phi))
        self.fp16_exact &= exact
        self.phi.append(f16 if exact and self.fp16_exact else phi)
        for k, v in (("kind", self.ctx["kind"]), ("branch", self.ctx["branch"]), ("draw", self.ctx["draw"]),
                     ("turn", frame["turn_idx"]), ("token_in_turn", frame["token_in_turn"]),
                     ("global_token", frame["global_token"]), ("token", frame["token"]),
                     ("token_id", token_id)):
            self.cols[k].append(v)

    def save(self, path: Path, extra_meta: dict | None = None) -> dict:
        if not self.phi:
            raise SystemExit("[capture] nothing was recorded")
        if self.fp16_exact:
            phi = np.stack([p.astype(np.float16) for p in self.phi])
            dtype = "float16"
        else:
            phi = np.stack([p.astype(np.float32) for p in self.phi])
            dtype = "float32"
        meta = {"superset": [self.s_lo, self.s_hi], "crad_band": list(self.band), "phi_dtype": dtype,
                "fp16_exact": bool(self.fp16_exact), "hidden_dtypes": sorted(self.hidden_dtypes),
                "n_frames": int(len(self.phi)), "n_analysis_draws": int(self.n_analysis),
                "calibration_branch_runs": {k: self.cal_counter[c] for k, c in BRANCH_CODE.items()},
                "launcher": LAUNCHER_VERSION, "mock": lc.is_mock(), **(extra_meta or {})}
        np.savez(path,
                 meta=np.array(json.dumps(meta)),
                 kind=np.asarray(self.cols["kind"], dtype=np.int8),
                 branch=np.asarray(self.cols["branch"], dtype=np.int8),
                 draw=np.asarray(self.cols["draw"], dtype=np.int32),
                 turn=np.asarray(self.cols["turn"], dtype=np.int32),
                 token_in_turn=np.asarray(self.cols["token_in_turn"], dtype=np.int32),
                 global_token=np.asarray(self.cols["global_token"], dtype=np.int32),
                 token=np.asarray(self.cols["token"], dtype=str),
                 token_id=np.asarray(self.cols["token_id"], dtype=np.int64),
                 phi=phi)
        return meta


def branch_of(turns, v13, crad_base) -> int:
    """Which calibration population a run_context_branch call builds (build_calibration's call order
    is N=NEUTRAL_TURNS, L=ramp.turns_S, R=ramp.turns_R)."""
    t = list(turns)
    if t == list(v13.NEUTRAL_TURNS):
        return BRANCH_CODE["N"]
    if t == list(crad_base.S_REAL_TURNS[:len(t)]):
        return BRANCH_CODE["L"]
    if t == list(crad_base.R_REAL_TURNS[:len(t)]):
        return BRANCH_CODE["R"]
    raise SystemExit("[capture] run_context_branch called with an unknown turn list")


def install(rec: Recorder, v13, crad_base) -> None:
    cls = crad_base.RecogModel
    orig_gwt = cls.gen_with_trace

    def gen_with_trace(self, messages, n, temp, layer_lo, layer_hi, turn_idx, global_start):
        if (layer_lo, layer_hi) != tuple(rec.band):
            raise SystemExit(f"[capture] CRAD traced [{layer_lo},{layer_hi}) but the capture band is {rec.band}")
        stash = {}
        had_attr = "generate" in vars(self.lm)
        inner = self.lm.generate

        def spy(*a, **k):
            out = inner(*a, **k)
            stash["seq"] = out.sequences[0].detach().cpu().tolist()
            stash["plen"] = int(k["input_ids"].shape[1])
            if out.hidden_states:
                stash["dtype"] = str(out.hidden_states[0][1].dtype)
            return out

        self.lm.generate = spy
        try:
            res = orig_gwt(self, messages, n, temp, rec.s_lo, rec.s_hi, turn_idx, global_start)
        finally:
            if had_attr:
                self.lm.generate = inner
            else:
                del self.lm.generate
        if "dtype" in stash:
            rec.hidden_dtypes.add(stash["dtype"])
        ids = stash["seq"][stash["plen"]:]
        a, b = layer_lo - rec.s_lo, layer_hi - rec.s_lo
        for i, fr in enumerate(res["frames"]):
            rec.record(fr, int(ids[i]) if i < len(ids) else -1)
            fr["phi"] = fr["phi"][a:b]                  # what CRAD sees: its band, same values
        return res

    cls.gen_with_trace = gen_with_trace

    orig_rcb = v13.run_context_branch

    def run_context_branch(mw, system, turns, *args, **kw):
        br = branch_of(turns, v13, crad_base)
        rec.ctx = {"kind": 0, "branch": br, "draw": rec.cal_counter[br]}
        rec.cal_counter[br] += 1
        try:
            return orig_rcb(mw, system, turns, *args, **kw)
        finally:
            rec.ctx = None

    v13.run_context_branch = run_context_branch

    orig_rpd = v13.run_pressure_draw

    def run_pressure_draw(*args, **kw):
        rec.ctx = {"kind": 1, "branch": -1, "draw": rec.n_analysis}
        rec.n_analysis += 1
        try:
            return orig_rpd(*args, **kw)
        finally:
            rec.ctx = None

    v13.run_pressure_draw = run_pressure_draw


def _flag(args: list[str], name: str) -> int:
    return int(args[args.index(name) + 1])


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if "--" not in sys.argv or sys.argv.index("--") != 6:
        raise SystemExit("usage: capture_launch.py SRC RUNTIME_JSON CAPTURE_NPZ S_LO S_HI -- <crad args>")
    src, rt_path, npz = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    s_lo, s_hi = int(sys.argv[4]), int(sys.argv[5])
    crad_args = sys.argv[7:]
    band = (_flag(crad_args, "--layer-lo"), _flag(crad_args, "--layer-hi"))
    t0, err = time.time(), None
    rec = Recorder(s_lo, s_hi, band)
    try:
        v13, crad_base = lc.load_v13_module(src)
        lc.install_model_probe(crad_base)
        install(rec, v13, crad_base)
        sys.argv = ["CRAD_v1.3.py"] + crad_args
        v13.main()
        meta = rec.save(npz)
        print(f"[capture] wrote {npz} · {meta['n_frames']} frames · phi {meta['phi_dtype']} "
              f"(fp16 exact: {meta['fp16_exact']}) · hidden dtypes {meta['hidden_dtypes']}", flush=True)
    except SystemExit as e:
        if e.code not in (None, 0):
            err = f"SystemExit: {e.code}"
            raise
    except BaseException:
        err = traceback.format_exc()
        raise
    finally:
        rt = lc.runtime_info(t0, err)
        rt["capture"] = {"superset": [s_lo, s_hi], "band": list(band), "launcher": LAUNCHER_VERSION}
        lc.write_json(rt_path, rt)
    return 0


# ---------------------------------------------------------------- selftest --
def selftest() -> int:
    """Writer + capture.check_capture on synthetic frames, CPU only, no model."""
    import tempfile
    lab = Path(__file__).resolve().parent
    sys.path.insert(0, str(lab))
    import run as runmod
    v13, crad_base = lc.load_v13_module(runmod.SRC)
    from capture import check_capture

    rng = np.random.default_rng(5)
    s_lo, s_hi, band, d = 3, 9, (4, 7), 16
    L = s_hi - s_lo
    rec = Recorder(s_lo, s_hi, band)
    a, b = band[0] - s_lo, band[1] - s_lo
    seen = {0: [], 1: [], 2: []}
    base = {0: rng.normal(size=(L, d)), 1: rng.normal(size=(L, d)) + 2.0, 2: rng.normal(size=(L, d)) - 1.0}
    # calibration: two runs per branch, two traced turns each, 7 tokens per turn
    for br in (0, 1, 2):
        for run_i in range(2):
            rec.ctx = {"kind": 0, "branch": br, "draw": run_i}
            g = 0
            for turn in (2, 3):
                for t in range(7):
                    phi = (base[br] + 0.3 * rng.normal(size=(L, d))).astype(np.float16).astype(np.float32)
                    fr = {"turn_idx": turn, "token_in_turn": t, "global_token": g, "token": f"t{t}", "phi": phi}
                    rec.record(fr, 100 + t)
                    seen[br].append({**fr, "phi": phi[a:b]})
                    g += 1
    draws = []
    for k in range(3):
        rec.ctx = {"kind": 1, "branch": -1, "draw": k}
        rec.n_analysis += 1
        n = 0
        for turn in (2, 3):
            for t in range(5 + k):
                phi = rng.normal(size=(L, d)).astype(np.float16).astype(np.float32)
                rec.record({"turn_idx": turn, "token_in_turn": t, "global_token": n, "token": f" w{t}",
                            "phi": phi}, 7 + t)
                n += 1
        draws.append({"draw": k, "n_frames": n})
    rec.ctx = None
    pops = {}
    for code, name in ((0, "N"), (1, "L"), (2, "R")):
        pops[f"H_{name}"], pops[f"D_{name}"] = v13.stack_states(seen[code]), v13.within_turn_deltas(seen[code])
    cal = v13.calibrate_from_arrays(*(pops[k] for k in ("H_N", "H_L", "H_R", "D_N", "D_L", "D_R")),
                                    layer_lo=band[0], layer_hi=band[1])
    result = {"draws": draws, "calibration": {"counts": cal.counts, "geometry": cal.geometry}}

    tests = []
    with tempfile.TemporaryDirectory() as td:
        npz = Path(td) / "capture.npz"
        meta = rec.save(npz)
        tests.append(("fp16 exactness detected on fp16-representable frames", meta["fp16_exact"] is True))
        probs, facts = check_capture(npz, result, v13)
        tests.append(("capture.check_capture: counts + value-level geometry identical", probs == []))
        z = dict(np.load(npz, allow_pickle=False))
        tests.append(("band slice of stored phi == what CRAD saw (bit-identical)",
                      np.array_equal(z["phi"][z["kind"] == 0][:, a:b].astype(np.float32),
                                     np.stack([f["phi"] for br in (0, 1, 2) for f in seen[br]]))))
        bad = dict(result, draws=[dict(draws[0], n_frames=draws[0]["n_frames"] + 1)] + draws[1:])
        tests.append(("frame-count mismatch is reported", any("captured" in p for p in check_capture(npz, bad, v13)[0])))
        bad_g = {"draws": draws, "calibration": {"counts": cal.counts,
                                                 "geometry": dict(cal.geometry, cos_LR_median=cal.geometry["cos_LR_median"] + 1e-12)}}
        tests.append(("1e-12 geometry drift is reported", any("geometry" in p for p in check_capture(npz, bad_g, v13)[0])))
        from dose_probe import captured_turn
        toks, ids = captured_turn(z, 1, 3)
        tests.append(("captured_turn returns rows in order with token ids", toks == [f" w{t}" for t in range(6)]
                      and ids == [7 + t for t in range(6)]))
    tests.append(("calibration branches identified by CRAD's own turn lists",
                  branch_of(v13.NEUTRAL_TURNS, v13, crad_base) == 0
                  and branch_of(crad_base.S_REAL_TURNS, v13, crad_base) == 1
                  and branch_of(crad_base.R_REAL_TURNS, v13, crad_base) == 2))
    nonexact = Recorder(0, 2, (0, 1))
    nonexact.ctx = {"kind": 1, "branch": -1, "draw": 0}
    nonexact.record({"turn_idx": 2, "token_in_turn": 0, "global_token": 0, "token": "x",
                     "phi": np.full((2, 4), 0.1, dtype=np.float32)}, 1)
    tests.append(("non-fp16 frames switch storage to float32", nonexact.fp16_exact is False))
    print(f"capture_launch {LAUNCHER_VERSION} selftest")
    for name, ok in tests:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
