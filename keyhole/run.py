"""Run the whole GPU experiment, model by model. Resumable: finished stages leave .done files.

python -m keyhole.run --out runs/main [--models q4 olmo q8 g12] [--limit N] [--hours 5.5]
"""
import argparse
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from keyhole import behavior, patching
from keyhole import config as C
from keyhole import prompts as P
from keyhole.engine import Engine


class Log:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.f.write(line + "\n"); self.f.flush()


def smoke(eng, items, layout, log):
    """S1: prefix-cache path equals a full forward without cache (incl. a dose-L prompt).
    S3: patching the final position at the last layer from plain reproduces plain's first-token logits."""
    pool, assign = P.load_fillers(layout)
    tok = eng.tok
    res = {}
    it = items[0]
    diffs = []
    for cond in (["plain", "null_pre", "json_S", "log_M"] + ([] if eng.device == "mps" else ["email_L"])):
        tr = P.make_trial(tok, it, cond, pool, assign)
        keep = list(range(len(tr.suffix_ids) - 3, len(tr.suffix_ids)))
        a, _ = eng.forward(eng.prefix_cache(tr.prefix_ids), [tr.suffix_ids], keep)
        ids = tr.prefix_ids + tr.suffix_ids
        with torch.inference_mode():
            b = eng.model(input_ids=eng.t([ids]), use_cache=False,
                          logits_to_keep=eng.t([len(tr.prefix_ids) + k for k in keep])).logits.float()
        la, lb = torch.log_softmax(a[0], -1), torch.log_softmax(b[0], -1)
        top = lb.argmax(-1)
        diffs.append({"cond": cond, "n": len(ids), "max_abs_dlogp_top": (la.gather(1, top[:, None]) - lb.gather(1, top[:, None])).abs().max().item(),
                      "argmax_equal": bool((la.argmax(-1) == top).all())})
    res["S1"] = diffs
    res["S1_pass"] = all(d["max_abs_dlogp_top"] <= 0.25 for d in diffs) and sum(d["argmax_equal"] for d in diffs) >= len(diffs) - 1
    plain = P.make_trial(tok, it, "plain", pool, assign)
    wr = P.make_trial(tok, it, "json_S", pool, assign)
    from keyhole.engine import Plan
    L = eng.L - 1
    one = torch.zeros(1, dtype=torch.long, device=eng.device)
    a, p = eng.forward(eng.prefix_cache(plain.prefix_ids), [plain.suffix_ids], [plain.final],
                       Plan(capture={("r", L): (one, eng.t([plain.final]))}))
    b, _ = eng.forward(eng.prefix_cache(wr.prefix_ids), [wr.suffix_ids], [wr.final],
                       Plan(patch={("r", L): (one, eng.t([wr.final]), p.got[("r", L)])}))
    res["S3_max_abs"] = (torch.log_softmax(a[0, 0], -1) - torch.log_softmax(b[0, 0], -1)).abs().max().item()
    res["S3_pass"] = res["S3_max_abs"] < 1e-2
    log(f"[{eng.key}] smoke: {json.dumps(res)}")
    return res


def gate(beh):
    known = set(beh[(beh.condition == "plain") & (beh.known == True) & (beh.split == "dev")].item_id)
    d = beh[beh.item_id.isin(known)]
    ret = d.groupby("condition").retained.mean()          # known answer still in the reply
    floor = float(np.mean([ret["null_pre"], ret["null_post"]]))
    struct = float(np.mean([ret[f"{f}_L"] for f in C.STRUCTURED]))
    return {"n_known_dev": len(known), "floor": floor, "struct_L": struct, "drop": floor - struct,
            "pass": (floor - struct) >= C.GATE_DROP and len(known) >= C.GATE_MIN_KNOWN_DEV,
            "retained": {k: float(v) for k, v in ret.items()},
            "acc": {k: float(v) for k, v in d.groupby("condition").correct.mean().items()}}


def patch_model(eng, beh, items, out, layout, grid, n_patch, log, deadline):
    pool, assign = P.load_fillers(layout)
    pairs, n_elig = patching.select_pairs(beh, items, eng.key, n_patch)
    log(f"[{eng.key}] eligible failures {n_elig}, selected pairs {len(pairs)}")
    byid = {it["id"]: it for it in items}
    trials = {}
    for p in pairs:
        for iid, cond in [(p["item_id"], "plain"), (p["donor_id"], "plain"), (p["item_id"], p["condition"])]:
            if (iid, cond) not in trials:
                trials[(iid, cond)] = P.make_trial(eng.tok, byid[iid], cond, pool, assign)
    pairs.sort(key=lambda p: (trials[(p["item_id"], p["condition"])].prefix_key, p["pair_id"]))
    gold = beh[beh.condition == "plain"].set_index("item_id")["g_ids"].apply(list).to_dict()
    json.dump({"pairs": pairs, "n_eligible": n_elig, "grid": grid}, open(out / f"pairs_{eng.key}_{layout}.json", "w"))
    df, skipped = patching.run(eng, pairs, items, trials, gold, beh, out, layout, grid, log, deadline)
    patching._CACHE.clear()
    return {"n_pairs": len(pairs), "n_eligible": n_elig, "skipped": skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/main")
    ap.add_argument("--models", nargs="*", default=C.MODEL_ORDER)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hours", type=float, default=5.5, help="hard budget counted from the first launch")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_patch", type=int, default=C.N_PATCH)
    ap.add_argument("--conds", nargs="*", default=None, help="restrict conditions (local tests only)")
    ap.add_argument("--finish", action="store_true", help="write ALL.done after this invocation")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "run.log")
    items = P.load_items()
    if a.limit:
        items = items[:a.limit]
    man_path = out / "manifest.json"
    man = json.load(open(man_path)) if man_path.exists() else {
        "start": time.strftime("%Y-%m-%dT%H:%M:%S"), "start_epoch": time.time(), "torch": torch.__version__,
        "transformers": __import__("transformers").__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "models": {}, "guard": [],
        "n_patch": a.n_patch, "grid_mode": "all", "layout": "tb64"}
    json.dump(man, open(man_path, "w"), indent=1)
    start = man["start_epoch"]
    deadline = start + a.hours * 3600
    prereg = out / "prereg.json"
    if not prereg.exists():
        json.dump({k: getattr(C, k) for k in dir(C) if k.isupper() and k not in ("ROOT", "DATA")}, open(prereg, "w"),
                  indent=1, default=str)
    factor = {"q4": 1.0, "olmo": 1.9, "q8": 1.9, "g12": 2.9}
    for key in a.models:
        if (out / f"{key}.done").exists():
            log(f"[{key}] already done"); continue
        if time.time() > deadline - 900:
            man["guard"].append(f"skip {key}: out of time"); log(f"[{key}] skipped: out of time"); continue
        t0 = time.time()
        rec = man["models"].setdefault(key, {})
        layout, n_patch, grid_mode = man["layout"], man["n_patch"], man["grid_mode"]
        eng = None
        try:
            eng = Engine(key, device=a.device)
            rec.update(L=eng.L, d=eng.d, load_s=time.time() - t0, layers_module=type(eng.layers[0]).__name__)
            rec["smoke"] = smoke(eng, items, layout, log)
            beh_path = out / f"behavior_{key}_{layout}.parquet"
            beh = pd.read_parquet(beh_path) if beh_path.exists() else behavior.run(eng, items, layout, out, log, a.conds)[0]
            if key == C.MODEL_ORDER[0] and "gate" not in man and not a.conds:
                gr = gate(beh)
                log(f"[gate] {json.dumps(gr)}")
                man["gate"] = {"tb64": gr}
                if not gr["pass"]:
                    beh_e1 = behavior.run(eng, items, "tb512", out, log)[0]
                    g1 = gate(beh_e1)
                    man["gate"]["tb512"] = g1
                    log(f"[gate E1] {json.dumps(g1)}")
                    if g1["pass"] or g1["drop"] > gr["drop"]:
                        layout = "tb512"; beh = beh_e1
                    man["null_branch"] = not (gr["pass"] or g1["pass"])
                man["layout"] = layout
                json.dump(man, open(man_path, "w"), indent=1)
            rec["behavior_s"] = time.time() - t0
            grid = list(range(eng.L)) if grid_mode == "all" else sorted(set(range(0, eng.L, 2)) | {eng.L - 1})
            tp = time.time()
            rec["patch"] = patch_model(eng, beh, items, out, layout, grid, n_patch, log, deadline - 900)
            rec["patch_s"] = time.time() - tp
            rec["total_s"] = time.time() - t0
            rec["grid_mode"], rec["n_patch"] = grid_mode, n_patch
            # guard: project the remaining models from this model's measured time
            remaining = [k for k in C.MODEL_ORDER if k != key and not (out / f"{k}.done").exists() and k not in man["models"]]
            unit = rec["total_s"] / factor.get(key, 1.0)
            projected = time.time() - start + sum(unit * factor.get(k, 1.0) for k in remaining)
            if projected > a.hours * 3600 * 0.9 and man["grid_mode"] == "all":
                man["grid_mode"] = "even"; man["guard"].append(f"G1 after {key}: projected {projected / 3600:.2f} h")
                projected -= 0.3 * sum(unit * factor.get(k, 1.0) for k in remaining)
            if projected > a.hours * 3600 * 0.9 and man["n_patch"] > 160:
                man["n_patch"] = 160; man["guard"].append(f"G2 after {key}")
            log(f"[{key}] done in {rec['total_s']:.0f}s; projected total {projected / 3600:.2f} h")
            (out / f"{key}.done").write_text("ok")
        except Exception as e:
            import traceback
            rec["error"] = traceback.format_exc()
            log(f"[{key}] FAILED: {e}")
        finally:
            json.dump(man, open(man_path, "w"), indent=1)
            if eng is not None:
                eng.close()
            del eng
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if a.finish:
        man["end"] = time.strftime("%Y-%m-%dT%H:%M:%S"); man["hours"] = (time.time() - start) / 3600
        json.dump(man, open(man_path, "w"), indent=1)
        (out / "ALL.done").write_text("ok")
        log(f"all done in {(time.time() - start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
