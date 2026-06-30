# -*- coding: utf-8 -*-
"""
control_analyze.py -- decompose the +0.45 ceiling into proximity vs length/clutter.

Paired over the instances scored in all three conditions (oracle_c, allctx_c, far_c).
Reports each pair's mean difference, paired SEM, and z = mean/SEM. The contrasts:

  oracle_c - allctx_c : the gap to be explained (should reproduce the ~+0.45 ceiling)
  far_c    - allctx_c : distractor PROXIMITY effect at matched length/count/position
                        (> 0  => near-miss interference is real = genuine selection value)
  oracle_c - far_c    : residual length/clutter from 6 FAR facets
                        (~ 0  => raw context length is NOT the driver)

Reads data/scores_control.jsonl ; writes data/control_decomp.json.
Run with any python (stdlib only).
"""
import os, json, math

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def paired(rows, a, b):
    xs = [(r["adh"][a], r["adh"][b]) for r in rows
          if r["adh"].get(a) is not None and r["adh"].get(b) is not None]
    d = [x - y for x, y in xs]
    n = len(d)
    mean = sum(d) / n
    var = sum((di - mean) ** 2 for di in d) / (n - 1)
    sem = math.sqrt(var / n)
    return {"a": a, "b": b, "n": n, "mean": round(mean, 3),
            "sem": round(sem, 3), "z": round(mean / sem, 2) if sem else None}

def main():
    rows = [json.loads(l) for l in open(f"{BASE}/data/scores_control.jsonl")]
    conds = ("oracle_c", "oracle_dup", "allctx_c", "far_c")
    means = {}
    for c in conds:
        ss = [r["adh"][c] for r in rows if r["adh"].get(c) is not None]
        if ss:
            means[c] = round(sum(ss) / len(ss), 3)

    contrasts = {
        "oracle_minus_allctx": paired(rows, "oracle_c", "allctx_c"),   # the +0.45 ceiling
        "dup_minus_oracle":    paired(rows, "oracle_dup", "oracle_c"), # pure length (distractor-free)
        "allctx_minus_dup":    paired(rows, "allctx_c", "oracle_dup"), # near-distractor at fixed length
        "far_minus_allctx":    paired(rows, "far_c", "allctx_c"),      # proximity: near -> far
    }
    out = {"n_instances": len(rows), "means": means, "contrasts": contrasts}
    with open(f"{BASE}/data/control_decomp.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"n = {len(rows)}")
    print("means:", means)
    for k, c in contrasts.items():
        print(f"  {k:22s} {c['mean']:+.3f} ±{c['sem']:.3f}  z={c['z']}  (n={c['n']})")
    # clean decomposition of the ceiling at fixed length:
    #   (oracle - allctx) = (oracle - dup)[ -length ] + (dup - allctx)[ -near-distractor ]
    if "oracle_dup" in means:
        oa = contrasts["oracle_minus_allctx"]["mean"]
        lo = -contrasts["dup_minus_oracle"]["mean"]    # length cost (oracle - dup)
        di = -contrasts["allctx_minus_dup"]["mean"]    # near-distractor cost (dup - allctx)
        print(f"\nceiling {oa:+.3f} = length {lo:+.3f} + near-distractor {di:+.3f}")

if __name__ == "__main__":
    main()
