# -*- coding: utf-8 -*-
"""
control_analyze_multi.py -- decomposition from the multi-sample run.

Averages the N draws per (id, condition) into a per-instance mean adherence, then runs the
same paired contrasts as control_analyze on those per-instance means. Averaging shrinks the
sampling-noise part of each instance's score, tightening the paired SEM.

Reads data/scores_control_multi.jsonl ; writes data/control_decomp_multi.json.
"""
import os, json, math
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def paired(per_inst, a, b):
    ids = [i for i in per_inst if a in per_inst[i] and b in per_inst[i]]
    d = [per_inst[i][a] - per_inst[i][b] for i in ids]
    n = len(d); mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    sem = math.sqrt(var / n)
    return {"a": a, "b": b, "n": n, "mean": round(mean, 3),
            "sem": round(sem, 3), "z": round(mean / sem, 2) if sem else None}

def main():
    # collect draws -> per (id,cond) list of scores
    cell = defaultdict(list)
    nsamp = defaultdict(int)
    for l in open(f"{BASE}/data/scores_control_multi.jsonl"):
        d = json.loads(l)
        if d["adh"] is None:
            continue
        cell[(d["id"], d["condition"])].append(d["adh"])
        nsamp[d["condition"]] = max(nsamp[d["condition"]], len(cell[(d["id"], d["condition"])]))

    conds = ("oracle_c", "oracle_dup", "allctx_c", "far_c")
    per_inst = defaultdict(dict)
    for (iid, cond), vals in cell.items():
        per_inst[iid][cond] = sum(vals) / len(vals)

    means = {}
    draws = {}
    for c in conds:
        xs = [per_inst[i][c] for i in per_inst if c in per_inst[i]]
        if xs:
            means[c] = round(sum(xs) / len(xs), 3)
            draws[c] = sum(len(cell[(i, c)]) for i in per_inst if c in per_inst[i])

    contrasts = {
        "oracle_minus_allctx": paired(per_inst, "oracle_c", "allctx_c"),
        "dup_minus_oracle":    paired(per_inst, "oracle_dup", "oracle_c"),
        "allctx_minus_dup":    paired(per_inst, "allctx_c", "oracle_dup"),
        "far_minus_allctx":    paired(per_inst, "far_c", "allctx_c"),
    }
    out = {"n_instances": len(per_inst), "samples_per_cell": dict(nsamp),
           "total_draws": draws, "means": means, "contrasts": contrasts}
    with open(f"{BASE}/data/control_decomp_multi.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"instances={len(per_inst)}  samples/cell={dict(nsamp)}")
    print("means:", means)
    for k, c in contrasts.items():
        print(f"  {k:22s} {c['mean']:+.3f} ±{c['sem']:.3f}  z={c['z']}  (n={c['n']})")
    if "oracle_dup" in means:
        oa = contrasts["oracle_minus_allctx"]["mean"]
        lo = -contrasts["dup_minus_oracle"]["mean"]
        di = -contrasts["allctx_minus_dup"]["mean"]
        print(f"\nceiling {oa:+.3f} = length {lo:+.3f} + near-distractor {di:+.3f}")

if __name__ == "__main__":
    main()
