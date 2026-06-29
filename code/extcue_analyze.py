# -*- coding: utf-8 -*-
"""
extcue_analyze.py -- Phase 3 final numbers + chart data.

Adherence (1-10, to the cued facet), all on the SAME 100 instances:
  base               (mrprompt-repro)         <- floor
  allctx = mrprompt  (all facets in context)  <- the competitor the reproduction validated
  oracle             (true facet injected)    <- ceiling
  extcue             (cue_only top-1)          extcue_wrongkey (control)
  extcue_body_top1   (body key, top-1)         extcue_cuesitu_top3 (body key, top-3, model picks)

Key questions:
  causality : extcue - extcue_wrongkey  vs  in-context mrprompt - wrongkey
  ceiling   : oracle - allctx
  does retrieval beat all-facets-in-context? : extcue_cuesitu_top3 - allctx
Outputs extcue_results.json + data/chart_data.json.
"""
import os, json, math, statistics

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def mean(v): return sum(v) / len(v) if v else float("nan")
def sem(v): return statistics.stdev(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")

def main():
    ext = {r["id"]: r for r in (json.loads(l) for l in open(f"{BASE}/data/scores_extcue.jsonl"))}
    inc = {r["id"]: r for r in (json.loads(l) for l in open(f"{BASE}/data/scores_faithful_incontext.jsonl"))}
    routing = json.load(open(f"{BASE}/extcue_routing2.json"))
    ids = [i for i in ext if i in inc]

    def ext_adh(c): return [ext[i]["adh"][c] for i in ids if ext[i]["adh"].get(c) is not None]
    def inc_adh(c): return [inc[i]["adh"][c] for i in ids if inc[i]["adh"].get(c) is not None]
    def paired(getter_a, getter_b):
        d = []
        for i in ids:
            a, b = getter_a(i), getter_b(i)
            if a is not None and b is not None:
                d.append(a - b)
        return mean(d), sem(d), len(d)
    eA = lambda c: (lambda i: ext[i]["adh"].get(c))
    iA = lambda c: (lambda i: inc[i]["adh"].get(c))

    levels = {
        "base": round(mean(inc_adh("base")), 3),
        "allctx(mrprompt)": round(mean(inc_adh("mrprompt")), 3),
        "oracle": round(mean(ext_adh("oracle")), 3),
        "extcue(cue_only_top1)": round(mean(ext_adh("extcue")), 3),
        "extcue_wrongkey": round(mean(ext_adh("extcue_wrongkey")), 3),
        "extcue_body_top1": round(mean(ext_adh("extcue_body_top1")), 3),
        "extcue_cuesitu_top3": round(mean(ext_adh("extcue_cuesitu_top3")), 3),
    }
    def fmt(m, s, n): return {"d": round(m, 3), "sem": round(s, 3), "n": n}
    contrasts = {
        # GATING result first: the max uplift perfect routing could buy at this scale.
        "ceiling(oracle - allctx)": fmt(*paired(eA("oracle"), iA("mrprompt"))),
        "retrieval_vs_allctx(cuesitu_top3 - mrprompt)": fmt(*paired(eA("extcue_cuesitu_top3"), iA("mrprompt"))),
        "body_top1_vs_allctx(body_top1 - mrprompt)": fmt(*paired(eA("extcue_body_top1"), iA("mrprompt"))),
        "external_causality(extcue - wrongkey)": fmt(*paired(eA("extcue"), eA("extcue_wrongkey"))),
        "incontext_causality(mrprompt - wrongkey)": fmt(*paired(iA("mrprompt"), iA("mrprompt_wrongkey"))),
    }
    # adherence split by retrieval correctness -- separates "facet value" from "routing quality".
    # body_top1: hit = top-1 == cued; cuesitu_top3: hit3 = cued present in top-3.
    g = [json.loads(l) for l in open(f"{BASE}/data/generations_extcue2.jsonl")] \
        if os.path.exists(f"{BASE}/data/generations_extcue2.jsonl") else []
    hit1 = {x["id"]: x.get("hit") for x in g if x["condition"] == "extcue_body_top1"}
    hit3 = {x["id"]: x.get("hit3") for x in g if x["condition"] == "extcue_cuesitu_top3"}
    def split(cond, flags, val):
        return round(mean([ext[i]["adh"][cond] for i in ids
                    if flags.get(i) == val and ext[i]["adh"].get(cond) is not None]), 3), \
               len([i for i in ids if flags.get(i) == val and ext[i]["adh"].get(cond) is not None])
    bt1_hit, n_bt1_hit = split("extcue_body_top1", hit1, 1)
    bt1_miss, n_bt1_miss = split("extcue_body_top1", hit1, 0)
    t3_hit, n_t3_hit = split("extcue_cuesitu_top3", hit3, 1)
    t3_miss, n_t3_miss = split("extcue_cuesitu_top3", hit3, 0)
    by_hit = {"body_top1_hit_mean": bt1_hit, "n_hit": n_bt1_hit,
              "body_top1_miss_mean": bt1_miss, "n_miss": n_bt1_miss,
              "top3_hit_mean": t3_hit, "n_top3_hit": n_t3_hit,
              "top3_miss_mean": t3_miss, "n_top3_miss": n_t3_miss}

    # routing causality: real - wrong at k=1 and k=3 per key (the column that shows thin keys
    # lose their cue-addressing at k=3 -> "grab 3 near facets" rather than route)
    KEYS = ("cue_only", "cue_situ", "body")
    route_causal = {k: {"R@1": round(routing["recall@1"][f"{k}_real"] - routing["recall@1"][f"{k}_wrong"], 3),
                        "R@3": round(routing["recall@3"][f"{k}_real"] - routing["recall@3"][f"{k}_wrong"], 3)}
                    for k in KEYS}

    R = {"n": len(ids), "adh_levels": levels, "contrasts": contrasts, "by_hit": by_hit,
         "routing_causality(real-wrong)": route_causal, "routing": routing}
    print(json.dumps(R, ensure_ascii=False, indent=2))
    json.dump(R, open(f"{BASE}/extcue_results.json", "w"), ensure_ascii=False, indent=2)

    chart = {
        "levels": [{"label": k, "v": v} for k, v in levels.items()],
        "causality": [
            {"label": "in-context: mrprompt − wrongkey", "d": contrasts["incontext_causality(mrprompt - wrongkey)"]["d"],
             "sem": contrasts["incontext_causality(mrprompt - wrongkey)"]["sem"], "g": "i"},
            {"label": "external: extcue − wrongkey", "d": contrasts["external_causality(extcue - wrongkey)"]["d"],
             "sem": contrasts["external_causality(extcue - wrongkey)"]["sem"], "g": "e"},
        ],
        "routing": {"chance1": round(routing["chance@1"], 3), "chance3": round(routing["chance@3"], 3),
                    "recall@1": routing["recall@1"], "recall@3": routing["recall@3"]},
        "by_hit": by_hit,
    }
    json.dump(chart, open(f"{BASE}/data/chart_data.json", "w"), ensure_ascii=False, indent=2)
    print("\nwrote extcue_results.json + data/chart_data.json")

if __name__ == "__main__":
    main()
