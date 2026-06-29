# -*- coding: utf-8 -*-
"""
extcue_route2.py -- Phase 3 routing: how rich does the KEY have to be, and does top-k help?

Key richness ladder (what represents a facet's address):
  cue_only : cue_phrases                                   (literal cue keys)
  cue_situ : cue_phrases + situation
  body     : situation + emotional_state + behavior_pattern + thinking_pattern  (facet content)
Query = full STM. Metrics: recall@1 (top-1 == cued) and recall@3 (cued in top-3), for
real keys and a wrongkey control (cued facet's key overwritten by a neighbour's).

Saves per-instance top-1 and top-3 selections for each key to
data/extcue_selections_v2.jsonl (used by generation). Writes extcue_routing2.json.
"""
import os, json
import torch
from transformers import AutoTokenizer, AutoModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES = f"{BASE}/data/instances_faithful.jsonl"
EMB = "BAAI/bge-m3"
KEYS = ("cue_only", "cue_situ", "body")

def facets_of(s):
    return s.get("scene_facets") or s.get("Personality", {}).get("scene_facets", [])

def key_text(f, mode):
    cue = "；".join(f.get("cue_phrases") or [])
    if mode == "cue_only":
        return cue or f.get("title", "")
    if mode == "cue_situ":
        return (cue + "。" + (f.get("situation") or "")).strip("。")
    return "。".join(str(f.get(k) or "") for k in
                     ("situation", "emotional_state", "behavior_pattern", "thinking_pattern")).strip("。")

class Embedder:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(EMB)
        self.model = AutoModel.from_pretrained(EMB, dtype=torch.bfloat16).cuda().eval()
    @torch.no_grad()
    def __call__(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        h = self.model(**enc).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(h.float(), dim=-1).cpu()

def topk_idx(sims, k):
    return sims.argsort(descending=True)[:k].tolist()

def main():
    insts = [json.loads(l) for l in open(INSTANCES)]
    emb = Embedder()
    Q = emb([it["stm"] for it in insts])

    rec1 = {(km, t): [] for km in KEYS for t in ("real", "wrong")}
    rec3 = {(km, t): [] for km in KEYS for t in ("real", "wrong")}
    chance1, chance3 = [], []
    selections = []
    for i, it in enumerate(insts):
        facets = facets_of(it["schema"]); n = len(facets); c = it["cued_index"]
        if n < 2 or not (0 <= c < n):
            continue
        chance1.append(1.0 / n); chance3.append(min(3, n) / n)
        other = (c + 1) % n
        srec = {"id": it["id"], "cued_index": c}
        for km in KEYS:
            real_keys = [key_text(f, km) for f in facets]
            wf = {**facets[c], "situation": facets[other].get("situation", ""),
                  "emotional_state": facets[other].get("emotional_state", ""),
                  "behavior_pattern": facets[other].get("behavior_pattern", ""),
                  "thinking_pattern": facets[other].get("thinking_pattern", ""),
                  "cue_phrases": facets[other].get("cue_phrases", [])}
            wrong_keys = list(real_keys); wrong_keys[c] = key_text(wf, km)
            for tag, keys in (("real", real_keys), ("wrong", wrong_keys)):
                sims = emb(keys) @ Q[i]
                t1 = topk_idx(sims, 1); t3 = topk_idx(sims, 3)
                rec1[(km, tag)].append(float(t1[0] == c))
                rec3[(km, tag)].append(float(c in t3))
                if tag == "real":
                    srec[km] = {"top1": t1[0], "top3": t3, "hit1": int(t1[0] == c), "hit3": int(c in t3)}
        selections.append(srec)

    def m(d, k): return sum(d[k]) / len(d[k])
    c1, c3 = sum(chance1)/len(chance1), sum(chance3)/len(chance3)
    print(f"\n=== Phase 3 routing, n={len(chance1)}, chance@1={c1:.3f} chance@3={c3:.3f} ===")
    print(f"{'key':9s} {'R@1 real':>9s} {'R@1 wrong':>10s} {'R@3 real':>9s} {'R@3 wrong':>10s}")
    for km in KEYS:
        print(f"{km:9s} {m(rec1,(km,'real')):9.3f} {m(rec1,(km,'wrong')):10.3f} "
              f"{m(rec3,(km,'real')):9.3f} {m(rec3,(km,'wrong')):10.3f}")

    with open(f"{BASE}/data/extcue_selections_v2.jsonl", "w") as fo:
        for s in selections:
            fo.write(json.dumps(s, ensure_ascii=False) + "\n")
    out = {"n": len(chance1), "chance@1": c1, "chance@3": c3,
           "recall@1": {f"{km}_{t}": m(rec1, (km, t)) for km in KEYS for t in ("real", "wrong")},
           "recall@3": {f"{km}_{t}": m(rec3, (km, t)) for km in KEYS for t in ("real", "wrong")}}
    json.dump(out, open(f"{BASE}/extcue_routing2.json", "w"), ensure_ascii=False, indent=2)
    print("saved selections_v2 + extcue_routing2.json")

if __name__ == "__main__":
    main()
