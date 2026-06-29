# -*- coding: utf-8 -*-
"""
extcue_route.py -- Phase 1 (routing) for the external-cue method, full version.

Tests whether the cue can ROUTE when facet bodies are not in the prompt: given the STM
as query, nearest-neighbour over each facet's cue should retrieve the cued facet.

Sweeps:
  query  in {full_stm, last1, last2}   (what we match against)
  key    in {cue_only, cue_situ}       (what represents the facet's address)
  cond   in {real, wrong}              (wrong = cued facet's cue overwritten by neighbour)
Reports top-1 accuracy vs chance, and saves the per-instance selection for the chosen
faithful setting (query=full_stm, key=cue_only) to data/extcue_selections.jsonl so the
generation stage does not need to reload the embedder.

Embedding: BAAI/bge-m3 (CLS pooling), bf16 on GPU.
"""
import os, json
import torch
from transformers import AutoTokenizer, AutoModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES = f"{BASE}/data/instances_faithful.jsonl"
EMB = "BAAI/bge-m3"
GEN_QUERY, GEN_KEY = "full_stm", "cue_only"   # faithful setting persisted for generation

def facets_of(s):
    return s.get("scene_facets") or s.get("Personality", {}).get("scene_facets", [])

def key_text(f, mode):
    cue = "；".join(f.get("cue_phrases") or [])
    if mode == "cue_only":
        return cue or f.get("title", "")
    return (cue + "。" + (f.get("situation") or "")).strip("。")

def query_text(stm, mode):
    if mode == "full_stm":
        return stm
    lines = [l for l in stm.splitlines() if l.strip()]
    return "\n".join(lines[-1:]) if mode == "last1" else "\n".join(lines[-2:])

class Embedder:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(EMB)
        self.model = AutoModel.from_pretrained(EMB, dtype=torch.bfloat16).cuda().eval()
    @torch.no_grad()
    def __call__(self, texts):
        enc = self.tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        h = self.model(**enc).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(h.float(), dim=-1).cpu()

def main():
    insts = [json.loads(l) for l in open(INSTANCES)]
    emb = Embedder()
    QUERIES = ("full_stm", "last1", "last2")
    KEYS = ("cue_only", "cue_situ")

    # precompute query embeddings per variant
    Qemb = {qm: emb([query_text(it["stm"], qm) for it in insts]) for qm in QUERIES}

    acc = {(qm, km, t): [] for qm in QUERIES for km in KEYS for t in ("real", "wrong")}
    chance = []
    selections = []
    for i, it in enumerate(insts):
        facets = facets_of(it["schema"]); n = len(facets); c = it["cued_index"]
        if n < 2 or not (0 <= c < n):
            continue
        chance.append(1.0 / n)
        other = (c + 1) % n
        for km in KEYS:
            real_keys = [key_text(f, km) for f in facets]
            wf = {**facets[c], "situation": facets[other].get("situation", ""),
                  "cue_phrases": facets[other].get("cue_phrases", [])}
            wrong_keys = list(real_keys); wrong_keys[c] = key_text(wf, km)
            Kr, Kw = emb(real_keys), emb(wrong_keys)
            for qm in QUERIES:
                q = Qemb[qm][i]
                sr, sw = int((Kr @ q).argmax()), int((Kw @ q).argmax())
                acc[(qm, km, "real")].append(float(sr == c))
                acc[(qm, km, "wrong")].append(float(sw == c))
                if qm == GEN_QUERY and km == GEN_KEY:
                    selections.append({"id": it["id"], "cued_index": c,
                                       "extcue": {"sel": sr, "hit": int(sr == c)},
                                       "extcue_wrongkey": {"sel": sw, "hit": int(sw == c)}})

    def m(k):
        v = acc[k]; return sum(v) / len(v) if v else float("nan")
    ch = sum(chance) / len(chance)
    print(f"\n=== extcue routing, n={len(chance)}, chance={ch:.3f} ===")
    print(f"{'query':9s} {'key':9s} {'real':>7s} {'wrong':>7s} {'real-wrong':>11s}")
    for qm in QUERIES:
        for km in KEYS:
            r, w = m((qm, km, "real")), m((qm, km, "wrong"))
            print(f"{qm:9s} {km:9s} {r:7.3f} {w:7.3f} {r-w:+11.3f}")

    with open(f"{BASE}/data/extcue_selections.jsonl", "w") as fo:
        for s in selections:
            fo.write(json.dumps(s, ensure_ascii=False) + "\n")
    summary = {"n": len(chance), "chance": ch, "gen_setting": {"query": GEN_QUERY, "key": GEN_KEY},
               "acc": {f"{qm}|{km}|{t}": m((qm, km, t)) for qm in QUERIES for km in KEYS for t in ("real", "wrong")}}
    json.dump(summary, open(f"{BASE}/extcue_routing.json", "w"), ensure_ascii=False, indent=2)
    print("saved selections + extcue_routing.json")

if __name__ == "__main__":
    main()
