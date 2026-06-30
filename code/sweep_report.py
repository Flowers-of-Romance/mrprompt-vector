# -*- coding: utf-8 -*-
"""
sweep_report.py -- measured vs projected adherence for the embedding-retriever methods.

For each method: measured mean adherence, hit/miss split, paired difference vs all-facets
(7.43), and the projected value from route_sweep.json for comparison. Tests whether replacing
projection with measurement changes the Result-3 null (methods tie all-facets).

Reads data/scores_sweep.jsonl + route_sweep.json
Writes data/sweep_measured.json
"""
import os, json, math

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLCTX = 7.43   # single-sample all-facets anchor (same as the rest of the article body)

def main():
    rows = [json.loads(l) for l in open(f"{BASE}/data/scores_sweep.jsonl")]
    proj = json.load(open(f"{BASE}/route_sweep.json"))

    by = {}
    for r in rows:
        if r["adh"] is None:
            continue
        by.setdefault(r["method"], []).append((r.get("hit"), r["adh"]))

    out = {}
    print(f"{'method':26s} {'R@1':>5} {'proj':>6} {'meas':>6} {'vsAll':>7} {'hit':>5} {'miss':>5} n")
    for m in sorted(by, key=lambda k: -proj.get(k, {}).get("R@1", 0)):
        vals = by[m]
        ss = [s for _, s in vals]
        mean = sum(ss) / len(ss)
        hit = [s for h, s in vals if h == 1]
        miss = [s for h, s in vals if h == 0]
        hm = sum(hit) / len(hit) if hit else None
        mm = sum(miss) / len(miss) if miss else None
        # paired SEM of (adh - ALLCTX) is just SEM of adh (ALLCTX is a constant anchor)
        var = sum((s - mean) ** 2 for s in ss) / (len(ss) - 1)
        sem = math.sqrt(var / len(ss))
        d = mean - ALLCTX
        out[m] = {"R@1": proj.get(m, {}).get("R@1"), "proj_adh": proj.get(m, {}).get("proj_adh_top1"),
                  "measured_adh": round(mean, 3), "vs_allfacets": round(d, 3),
                  "sem": round(sem, 3), "z": round(d / sem, 2) if sem else None,
                  "hit_adh": round(hm, 3) if hm is not None else None,
                  "miss_adh": round(mm, 3) if mm is not None else None, "n": len(ss)}
        print(f"{m:26s} {out[m]['R@1']:>5} {str(out[m]['proj_adh']):>6} {mean:>6.2f} "
              f"{d:>+7.2f} {('%.2f'%hm) if hm else ' -':>5} {('%.2f'%mm) if mm else ' -':>5} {len(ss)}")

    with open(f"{BASE}/data/sweep_measured.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nall-facets anchor = {ALLCTX}.  wrote data/sweep_measured.json")

if __name__ == "__main__":
    main()
