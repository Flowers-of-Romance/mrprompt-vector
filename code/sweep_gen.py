# -*- coding: utf-8 -*-
"""
sweep_gen.py -- MEASURE (not project) the embedding-retriever methods.

route_sweep.py gave each method's R@1/R@3 and a PROJECTED adherence (R@1 * 7.74 + (1-R@1) *
7.31). The projection assumes the hit-group mean (7.74, from body_top1) is constant across
methods, which need not hold. Here we measure the methods that were only projected: inject each
method's top-1 facet and generate, exactly like body_top1 / oracle (extcue_gen2.facets_system,
single facet), so the result is directly comparable to body_top1 (7.46) / all-facets (7.43).

Skipped (already measured elsewhere, or trivial):
  bge-m3 dense body = body_top1 (scores_extcue), bge-m3 dense cue = extcue, random = chance.

Same model / settings as the rest (Qwen3-8B, thinking-OFF, max_new=1024, temp 0.7 / top_p 0.8),
single sample, matching the existing measured top-1 conditions. Resumable.

Reads  data/route_sweep_selections.jsonl + data/instances_faithful.jsonl
Writes data/generations_sweep.jsonl. Run with ~/comfy-rocm/bin/python.
"""
import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faithful_render as R
import extcue_gen2 as E2     # reuse facets_system / facets_of -> identical rendering to body_top1

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES = f"{BASE}/data/instances_faithful.jsonl"
SELECTIONS = f"{BASE}/data/route_sweep_selections.jsonl"
OUT = f"{BASE}/data/generations_sweep.jsonl"
GEN = "Qwen/Qwen3-8B"
MAX_NEW = 1024
SKIP = {"random", "bge-m3 dense body", "bge-m3 dense cue"}   # already measured / trivial

def main():
    insts = {it["id"]: it for it in (json.loads(l) for l in open(INSTANCES))}
    sels = [json.loads(l) for l in open(SELECTIONS) if json.loads(l)["method"] not in SKIP]
    methods = sorted({s["method"] for s in sels})
    print(f"measuring {len(methods)} methods x {len(insts)} instances: {methods}")

    tok = AutoTokenizer.from_pretrained(GEN)
    model = AutoModelForCausalLM.from_pretrained(GEN, dtype=torch.bfloat16, device_map="auto").eval()

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                d = json.loads(l); done.add((str(d["id"]), d["method"]))
            except Exception:
                pass
    out = open(OUT, "a")
    # method-outer so the highest-R@1 method (bge-large-zh) can be judged as soon as it finishes
    for method in methods:
        n_made = 0
        for s in (x for x in sels if x["method"] == method):
            key = (str(s["id"]), method)
            if key in done:
                continue
            it = insts[s["id"]]
            facets = E2.facets_of(it["schema"]); role = it["role"]
            sel = s["top1"]
            system_text = E2.facets_system(it["schema"], [facets[sel]], role)
            msgs = R.build_messages(system_text, it["stm"], role)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=True,
                                   temperature=0.7, top_p=0.8, pad_token_id=tok.eos_token_id)
            resp = tok.decode(g[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
            out.write(json.dumps({"id": s["id"], "role": role, "method": method,
                                  "cued_index": s["cued_index"], "sel": sel,
                                  "hit": s["hit1"], "response": resp}, ensure_ascii=False) + "\n")
            out.flush(); n_made += 1
        print(f"[method {method}] {n_made} new")
    out.close()
    print("done:", OUT)

if __name__ == "__main__":
    main()
