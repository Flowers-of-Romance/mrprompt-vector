# -*- coding: utf-8 -*-
"""
extcue_retrieve.py -- Phase 1 of the "external cue" experiment (mrprompt-vector).

Question: if facet bodies are NOT in the prompt, can the cue keys ROUTE? Given the STM
as query, does nearest-neighbour over each facet's cue retrieve the facet the assemble
step marked as cued (cued_index)? And does corrupting the key (wrongkey) destroy that?

This is the cheap test the in-context reproduction (mrprompt-repro) could not do: there
the body is visible so the cue is bypassed; here only the cue is available for selection.

What we embed as the facet's address:
  cue_only : cue_phrases only           <- the literal "cue keys" (most faithful test)
  cue_situ : cue_phrases + situation     <- richer key
Conditions:
  real     : true keys                                  -> acc_real
  wrongkey : cued facet's key overwritten by neighbour  -> acc_wrong  (mirrors mrprompt-repro)
  random   : 1 / n_facets                               -> chance

Embedding: BAAI/bge-m3 (multilingual, CLS pooling, no instruction), bf16 on GPU.
Input : data/instances_faithful.jsonl   Output: extcue_retrieval.json + stdout table.
No generation, no API.
"""
import os, json
import torch
from transformers import AutoTokenizer, AutoModel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES = f"{BASE}/data/instances_faithful.jsonl"
EMB = "BAAI/bge-m3"

def facets_of(schema):
    if "scene_facets" in schema:
        return schema.get("scene_facets", [])
    return schema.get("Personality", {}).get("scene_facets", [])

def key_text(f, mode):
    cue = "；".join(f.get("cue_phrases") or [])
    if mode == "cue_only":
        return cue or f.get("title", "")
    return (cue + "。" + (f.get("situation") or "")).strip("。")

class Embedder:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(EMB)
        self.model = AutoModel.from_pretrained(EMB, dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def __call__(self, texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i+bs], padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to("cuda")
            h = self.model(**enc).last_hidden_state[:, 0]      # CLS (bge-m3 dense)
            out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
        return torch.cat(out)

def main():
    insts = [json.loads(l) for l in open(INSTANCES)]
    emb = Embedder()

    Q = emb([it["stm"] for it in insts])                       # query = whole STM

    res = {(m, t): [] for m in ("cue_only", "cue_situ") for t in ("real", "wrong")}
    chance = []
    for qi, it in enumerate(insts):
        facets = facets_of(it["schema"]); n = len(facets); c = it["cued_index"]
        if n < 2 or not (0 <= c < n):
            continue
        chance.append(1.0 / n)
        other = (c + 1) % n
        for mode in ("cue_only", "cue_situ"):
            real_keys = [key_text(f, mode) for f in facets]
            wrong_keys = list(real_keys)
            wf = {**facets[c], "situation": facets[other].get("situation", ""),
                  "cue_phrases": facets[other].get("cue_phrases", [])}
            wrong_keys[c] = key_text(wf, mode)
            for tag, keys in (("real", real_keys), ("wrong", wrong_keys)):
                K = emb(keys)
                sel = int((K @ Q[qi]).argmax())
                res[(mode, tag)].append(1.0 if sel == c else 0.0)

    def acc(k):
        v = res[k]; return sum(v) / len(v) if v else float("nan")
    n_eval = len(res[("cue_only", "real")])
    ch = sum(chance) / len(chance)
    print(f"\n=== extcue retrieval (Phase 1), n={n_eval}, chance={ch:.3f} ===")
    print(f"{'key':10s} {'real acc':>10s} {'wrongkey acc':>14s} {'real - wrong':>14s}")
    for mode in ("cue_only", "cue_situ"):
        r, w = acc((mode, "real")), acc((mode, "wrong"))
        print(f"{mode:10s} {r:10.3f} {w:14.3f} {r-w:+14.3f}")
    print("\nreal >> chance => the cue carries routing info (addressable);")
    print("real >> wrongkey => corrupting the matching key breaks routing (key is load-bearing).")

    with open(f"{BASE}/extcue_retrieval.json", "w") as fo:
        json.dump({"n": n_eval, "chance": ch,
                   "acc": {f"{m}_{t}": acc((m, t)) for m in ("cue_only","cue_situ") for t in ("real","wrong")}},
                  fo, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
