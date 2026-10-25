"""Every number and table in the paper, as LaTeX, from results/results.json and the run outputs.

    python -m keyhole.numbers --results results --run runs/main --out paper/generated
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from keyhole import config as C
from keyhole import prompts as P

NAMES = {"q4": "Qwen3-4B", "olmo": "OLMo-3-7B", "q8": "Qwen3-8B", "g12": "Gemma-4-12B"}
FAM = {"json": "Tool result", "html": "Web page", "log": "Server log", "email": "Email thread",
       "prose": "Plain prose"}
STRUCT3 = ["json", "html", "email"]


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}\\%"


def prange(xs, nd=0):
    lo, hi = min(xs), max(xs)
    a, b = f"{100 * lo:.{nd}f}", f"{100 * hi:.{nd}f}"
    return f"{a}\\%" if a == b else f"{a}\\%--{b}\\%"


def frange(xs, nd=2):
    lo, hi = min(xs), max(xs)
    a, b = f"{lo:.{nd}f}", f"{hi:.{nd}f}"
    return a if a == b else f"{a}--{b}"


def pooled_rate(conds, names, key):
    """n-weighted mean of a per-condition rate over several conditions."""
    num = sum(conds[c][key]["mean"] * conds[c][key]["n"] for c in names)
    den = sum(conds[c][key]["n"] for c in names)
    return num / den


def prange_int(xs):
    return str(min(xs)) if min(xs) == max(xs) else f"{min(xs)}--{max(xs)}"


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default="paper/generated")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.load(open(Path(a.results) / "results.json"))
    M = res["models"]
    keys = [k for k in C.MODEL_ORDER if k in M]
    pkeys = [k for k in keys if "patching" in M[k]]
    items = P.load_items()
    m = {}
    m["NItems"] = thousands(len(items))
    for fam, name in [("FACT", "NFact"), ("ARITH", "NArith"), ("TRANS", "NTrans")]:
        m[name] = str(sum(it["family"] == fam for it in items))
    m["NModels"] = str(len(keys))
    known = {k: M[k]["behaviour"]["n_known"] / M[k]["behaviour"]["n_items"] for k in keys}
    m["KnownMin"], m["KnownMax"] = pct(min(known.values())), pct(max(known.values()))
    m["TailTokens"] = "500"

    # prompt lengths, from the run outputs (median over models' wrapped prompts)
    toks = {d: [] for d in C.DOSES}
    for k in keys:
        b = pd.read_parquet(Path(a.run) / f"behavior_{k}_{res['layout']}.parquet", columns=["condition", "n_prompt_tok"])
        for d in C.DOSES:
            toks[d] += list(b[b.condition.str.endswith("_" + d) & ~b.condition.isin(C.NULLS)].n_prompt_tok)
    med = {d: float(np.median(toks[d])) for d in C.DOSES}
    m["LenS"] = thousands(int(round(med["S"], -2)))
    m["LenM"] = thousands(int(round(med["M"], -3)))
    m["LenL"] = thousands(int(round(med["L"], -3)))

    # behaviour: losses per model, all three lengths together
    def lost(k, fams):
        c = M[k]["behaviour"]["conditions"]
        return 1 - pooled_rate(c, [f"{f}_{d}" for f in fams for d in C.DOSES], "retained")
    m["FloorLost"] = prange([1 - M[k]["behaviour"]["floor"]["mean"] for k in keys], 1)
    m["LostStruct"] = prange([lost(k, STRUCT3) for k in keys])
    m["LostProse"] = prange([lost(k, ["prose"]) for k in keys])
    m["LostLog"] = prange([lost(k, ["log"]) for k in keys])
    worst = max(((1 - M[k]["behaviour"]["conditions"][f"log_{d}"]["retained"]["mean"], k, d) for k in keys
                 for d in C.DOSES))
    m["LostLogMax"] = pct(worst[0])
    m["LostLogMaxModel"] = NAMES[worst[1]]

    # the shorter tail (Qwen3-4B only)
    if "q4_tb64" in res:
        c64, c512 = res["q4_tb64"]["conditions"], M["q4"]["behaviour"]["conditions"]
        other = [f"{f}_{d}" for f in ["json", "html", "email", "prose"] for d in C.DOSES]
        logs = [f"log_{d}" for d in C.DOSES]
        m["LostLogShortTail"] = pct(1 - pooled_rate(c64, logs, "retained"))
        m["LostLogLongTail"] = pct(1 - pooled_rate(c512, logs, "retained"))
        m["LostOtherShortTail"] = pct(1 - pooled_rate(c64, other, "retained"))
        m["LostOtherLongTail"] = pct(1 - pooled_rate(c512, other, "retained"))
        rows = []
        for f in C.FAMILIES:
            names = [f"{f}_{d}" for d in C.DOSES]
            rows.append(f"{FAM[f]} & {pct(pooled_rate(c64, names, 'retained'))} & {pct(pooled_rate(c512, names, 'retained'))} \\\\")
        tab = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule", "Wrapper & 64 tokens after & 512 tokens after \\\\", "\\midrule",
               *rows, "\\bottomrule", "\\end{tabular}"]
        (out / "table_tail.tex").write_text("\n".join(tab) + "\n")

    # what replaces lost answers (pooled over models)
    P_ = res["pooled"]
    lt = P_["lost_types"]
    share = {f: {t: lt[f][t] / max(1, sum(lt[f].values())) for t in ("task", "scope", "wrong")} for f in C.FAMILIES}
    m["TypeLogTask"] = pct(share["log"]["task"])
    m["TypeProseScope"] = pct(share["prose"]["scope"])
    rows = [f"{FAM[f]} & {pct(share[f]['task'])} & {pct(share[f]['scope'])} & {pct(share[f]['wrong'])} & "
            f"{thousands(sum(lt[f].values()))} \\\\" for f in C.FAMILIES]
    tab = ["\\begin{tabular}{@{}lcccr@{}}", "\\toprule",
           "Wrapper & About the & Not & Wrong & $n$ \\\\", " & wrapper & there & answer & \\\\", "\\midrule",
           *rows, "\\bottomrule", "\\end{tabular}"]
    (out / "table_types.tex").write_text("\n".join(tab) + "\n")

    # which models lose most to logs
    worst = [k for k in keys if lost(k, ["log"]) > max(lost(k, [f]) for f in C.FAMILIES if f != "log") + 0.05]
    mild = [k for k in keys if k not in worst]

    def join_names(ks):
        n = [NAMES[k] for k in ks]
        return n[0] if len(n) == 1 else ", ".join(n[:-1]) + " and " + n[-1]
    m["LogWorstModels"] = join_names(worst) if worst else "no model"
    m["LogMildModels"] = join_names(mild) if mild else "no model"
    m["NLogWorst"] = str(len(worst))
    m["NLogWorstWord"] = ["no", "one", "two", "three", "all"][len(worst)] if len(worst) < 4 else "all"
    m["LogMildVerb"] = "loses" if len(mild) == 1 else "lose"

    # what wrapped prompts produce, by how sure the model is of the answer
    tc = P_["type_by_confidence"]
    m["WrongLow"], m["WrongHigh"] = pct(tc["wrong"][0]["mean"]), pct(tc["wrong"][-1]["mean"], 1)
    m["ScopeLow"], m["ScopeHigh"] = pct(tc["scope"][0]["mean"]), pct(tc["scope"][-1]["mean"], 1)
    m["TaskRange"] = prange([r["mean"] for r in tc["task"]], 1)
    m["KeptLowAll"], m["KeptHighAll"] = pct(tc["kept"][0]["mean"]), pct(tc["kept"][-1]["mean"])
    tg = P_["type_by_confidence_group"]
    m["LogTaskLow"], m["LogTaskHigh"] = pct(tg["log"]["task"][0]["mean"]), pct(tg["log"]["task"][-1]["mean"])
    m["OtherWrongLow"], m["OtherWrongHigh"] = pct(tg["other"]["wrong"][0]["mean"]), pct(tg["other"]["wrong"][-1]["mean"], 1)
    m["OtherScopeLow"], m["OtherScopeHigh"] = pct(tg["other"]["scope"][0]["mean"]), pct(tg["other"]["scope"][-1]["mean"], 1)
    m["OtherScopeMid"] = pct(tg["other"]["scope"][3]["mean"])
    m["NullLow"] = pct(P_["null_lost_by_confidence"][0]["mean"])
    m["NullRest"] = pct(max(r["mean"] for r in P_["null_lost_by_confidence"][1:]), 1)
    m["BinN"] = ", ".join(thousands(n) for n in P_["type_bin_n"])
    m["KnownTestMin"] = thousands(min(P_["n_known_test"].values()))
    m["KnownTestMax"] = thousands(max(P_["n_known_test"].values()))

    # exact-format accuracy vs kept (pooled over models and lengths)
    rows, ex_all = [], []
    for f in C.FAMILIES:
        names = [f"{f}_{d}" for d in C.DOSES]
        kept = np.mean([pooled_rate(M[k]["behaviour"]["conditions"], names, "retained") for k in keys])
        exact = np.mean([pooled_rate(M[k]["behaviour"]["conditions"], names, "correct") for k in keys])
        ex_all.append(exact)
        rows.append(f"{FAM[f]} & {pct(kept)} & {pct(exact)} \\\\")
    nul = np.mean([pooled_rate(M[k]["behaviour"]["conditions"], C.NULLS, "correct") for k in keys])
    nulk = np.mean([pooled_rate(M[k]["behaviour"]["conditions"], C.NULLS, "retained") for k in keys])
    rows.append(f"Neutral sentence & {pct(nulk)} & {pct(nul)} \\\\")
    m["ExactMin"], m["ExactMax"] = pct(min(ex_all)), pct(max(ex_all))
    tab = ["\\begin{tabular}{@{}lcc@{}}", "\\toprule", "Wrapper & Kept & Exactly right \\\\", "\\midrule", *rows,
           "\\bottomrule", "\\end{tabular}"]
    (out / "table_exact.tex").write_text("\n".join(tab) + "\n")

    # models table
    rows = []
    for k in keys:
        b = M[k]["behaviour"]
        npairs = M[k]["patching"]["n_pairs_test"] if k in pkeys else 0
        rows.append(f"{NAMES[k]} & {M[k]['L']} & {thousands(b['n_known'])} ({pct(b['n_known'] / b['n_items'])}) & "
                    f"{thousands(b['n_known_test'])} & {npairs} \\\\")
    tab = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule", "Model & Layers & Known & Known test & Pairs \\\\", "\\midrule", *rows,
           "\\bottomrule", "\\end{tabular}"]
    (out / "table_models.tex").write_text("\n".join(tab) + "\n")

    # patching
    if pkeys:
        npairs = [M[k]["patching"]["n_pairs_dev"] + M[k]["patching"]["n_pairs_test"] for k in pkeys]
        m["NPairs"] = f"{min(npairs)}--{max(npairs)}" if min(npairs) != max(npairs) else str(npairs[0])
        m["NPatchQfour"] = str(C.N_PATCH)
        m["NPatchOther"] = "160"
        swap_peak, donor_kept, restore_peak, restore_all, handoff, astar, dstar = [], [], [], [], [], [], []
        for k in pkeys:
            p = M[k]["patching"]
            t = p["top1"]
            first = [r["mean"] for r in t["D_donor"] if r["depth"] <= 0.5]
            swap_peak.append(max(first))
            donor_kept.append(t["donor_kept_in_wrapper"]["mean"])
            restore_peak.append(max(r["mean"] for r in t["A_own"]))
            restore_all.append(t["AFULL_own"]["mean"])
            b = t["B_own"]
            handoff.append(next(r["depth"] for r in b if r["mean"] >= 0.5))
            astar.append(p["A_star"]["mean"])
            dstar.append(-p["D_star"]["mean"])
        m["NPairsTest"] = prange_int([M[k]["patching"]["n_pairs_test"] for k in pkeys])
        m["DonorTopOne"] = prange([M[k]["patching"]["top1"]["donor_top1_in_wrapper"]["mean"] for k in pkeys])
        m["NoPatch"] = prange([M[k]["patching"]["top1"]["BASE_own"]["mean"] for k in pkeys])
        grp = lambda k, g: M[k]["patching"]["top1"]["by_group"].get(g)
        m["RestorePeakOther"] = prange([max(r["mean"] for r in grp(k, "other")["A_own"]) for k in pkeys if grp(k, "other")])
        m["RestorePeakLog"] = prange([max(r["mean"] for r in grp(k, "log")["A_own"]) for k in pkeys if grp(k, "log")])
        m["NLogPairs"] = prange_int([grp(k, "log")["n"] for k in pkeys if grp(k, "log")])
        m["SwapPeakOther"] = prange([max(r["mean"] for r in grp(k, "other")["D_donor"] if r["depth"] <= 0.5)
                                     for k in pkeys if grp(k, "other")])
        m["SwapPeak"] = prange(swap_peak)
        m["DonorKept"] = prange(donor_kept)
        m["RestorePeak"] = pct(max(restore_peak))
        m["RestoreAll"] = prange(restore_all)
        m["HandoffDepth"] = frange(handoff, 2)
        m["AStar"] = prange(astar)
        m["DStar"] = prange(dstar)
        m["DStarNeg"] = f"$-{max(dstar):.2f}$ to $-{min(dstar):.2f}$"
        rows = []
        for k in pkeys:
            p = M[k]["patching"]
            f = lambda r: f"{r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]"
            rows.append(f"{NAMES[k]} & {p['lstar']} & {f(p['A_star'])} & {f(p['D_star'])} & {f(p['AFULL'])} \\\\")
        tab = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule", "Model & $\\ell^*$ & Own & Other & All layers \\\\",
               "\\midrule", *rows, "\\bottomrule", "\\end{tabular}"]
        (out / "table_repair.tex").write_text("\n".join(tab) + "\n")

    man = json.load(open(Path(a.run) / "manifest.json"))
    sk = [len(man["models"][k]["patch"]["skipped"]) for k in pkeys]
    m["NSkipped"] = ", ".join(str(x) for x in sk[:-1]) + " and " + str(sk[-1])
    g = res.get("gate") or {}
    if "tb64" in g:
        m["GateShortDrop"] = f"{100 * g['tb64']['drop']:.1f}"
    if "tb512" in g:
        m["GateLongDrop"] = f"{100 * g['tb512']['drop']:.1f}"
    m["GPUHours"] = f"{res['hours']:.1f}" if res.get("hours") else "XX"

    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
