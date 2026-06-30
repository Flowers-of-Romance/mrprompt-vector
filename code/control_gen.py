# -*- coding: utf-8 -*-
"""
control_gen.py -- Deconfound the +0.45 "ceiling" (oracle - allctx).

oracle (1 cued facet) beats allctx (all same-character facets) by +0.45. That gap is
confounded: removing the 6 non-cued facets BOTH drops the distractors AND shortens the
context. The two cannot be separated from oracle vs allctx alone.

This run separates them with a length-/count-/position-matched control whose ONLY changed
variable is distractor proximity. Three conditions, all rendered through the SAME wrapper
(facet_ltm_text full + Magic-If), so the only difference is which facets populate the list:

  oracle_c : cued facet only                              (short, no distractor)
  allctx_c : all of THIS character's facets               (long, NEAR distractors) = mrprompt
  far_c    : cued facet at its slot + the other (n-1)     (long, FAR distractors)
             facets replaced by facets drawn from OTHER characters

Decomposition (paired, same 100 instances):
  oracle_c - allctx_c = the +0.45 to be explained (should reproduce)
  far_c    - allctx_c = effect of distractor PROXIMITY at matched length/count/position
                        -> isolates near-miss interference (= selection value, if > 0)
  oracle_c - far_c    = residual length/clutter from 6 FAR facets

If far_c ~ oracle_c (far distractors harmless) and far_c >> allctx_c -> the +0.45 is
near-miss interference (genuine selection value), NOT context length.
If far_c ~ allctx_c -> any clutter hurts equally -> it IS a length/clutter penalty.

Same model / budget / sampling as the rest of the project (Qwen3-8B, thinking-OFF,
max_new=1024, temp 0.7 / top_p 0.8). Run with ~/comfy-rocm/bin/python.
Writes data/generations_control.jsonl (resumable).
"""
import os, json, copy, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faithful_render as R
import faithful_prompts as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES = f"{BASE}/data/instances_faithful.jsonl"
OUT = f"{BASE}/data/generations_control.jsonl"
GEN = "Qwen/Qwen3-8B"
MAX_NEW = 1024
CONDS = ("oracle_c", "allctx_c", "far_c", "oracle_dup")

def facets_of(s):
    return s.get("scene_facets") or s.get("Personality", {}).get("scene_facets", [])

def set_facets(schema, facets):
    s = copy.deepcopy(schema)
    if "scene_facets" in s:
        s["scene_facets"] = facets
    else:
        s.setdefault("Personality", {})["scene_facets"] = facets
    return s

def system_text_for(schema, role):
    """Render the facet-LTM (Magic-If, Fig.19) -- identical wrapper for all three conditions.
    facet_ltm_text reads schema.name for the header but the STM speaker label is `role`, so we
    pass role through magicif_system exactly as faithful_render.system_for(mrprompt) does."""
    info = R.facet_ltm_text(schema, mode="full")
    return F.magicif_system(role, info)

def build_pool(insts):
    """Global pool of (role, facet) across all instances; used to draw FAR distractors."""
    pool = []
    for it in insts:
        for f in facets_of(it["schema"]):
            pool.append((it["role"], f))
    return pool

def far_facets(pool, role, k, seed):
    """Deterministically draw k facets from characters OTHER than `role` (no duplicates by id)."""
    cand = [f for (r, f) in pool if r != role]
    rng = random.Random(seed)
    rng.shuffle(cand)
    return cand[:k]

def schema_for(cond, it, pool):
    schema, facets = it["schema"], facets_of(it["schema"])
    ci = it["cued_index"]
    if cond == "oracle_c":
        return set_facets(schema, [facets[ci]])
    if cond == "allctx_c":
        return set_facets(schema, facets)            # unchanged: all same-character facets
    if cond == "oracle_dup":
        # distractor-free length control: cued facet repeated to the same slot count as allctx.
        # Same length as allctx, ZERO wrong-situation distractor -> isolates pure length/redundancy.
        return set_facets(schema, [facets[ci]] * len(facets))
    # far_c: keep cued facet at its slot, replace the rest with other-character facets
    fars = far_facets(pool, it["role"], len(facets) - 1, seed=int(it["id"]))
    newf, fi = [], 0
    for i in range(len(facets)):
        if i == ci:
            newf.append(facets[i])
        else:
            newf.append(fars[fi]); fi += 1
    return set_facets(schema, newf)

def main():
    insts = [json.loads(l) for l in open(INSTANCES)]
    pool = build_pool(insts)

    tok = AutoTokenizer.from_pretrained(GEN)
    model = AutoModelForCausalLM.from_pretrained(GEN, dtype=torch.bfloat16, device_map="auto").eval()

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                d = json.loads(l); done.add((str(d["id"]), d["condition"]))
            except Exception:
                pass
    out = open(OUT, "a")
    # condition-outer so oracle_c (and allctx_c) finish first and can be judged early
    for cond in CONDS:
        ntok = []
        for n, it in enumerate(insts):
            if (str(it["id"]), cond) in done:
                continue
            role = it["role"]
            schema = schema_for(cond, it, pool)
            system_text = system_text_for(schema, role)
            msgs = R.build_messages(system_text, it["stm"], role)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt").to(model.device)
            ntok.append(int(ids.input_ids.shape[1]))
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=True,
                                   temperature=0.7, top_p=0.8, pad_token_id=tok.eos_token_id)
            resp = tok.decode(g[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
            out.write(json.dumps({"id": it["id"], "role": role, "condition": cond,
                                  "cued_index": it["cued_index"],
                                  "n_facets": len(facets_of(schema)),
                                  "prompt_tokens": int(ids.input_ids.shape[1]),
                                  "response": resp}, ensure_ascii=False) + "\n")
            out.flush()
        if ntok:
            print(f"[cond {cond} done]  mean prompt_tokens={sum(ntok)/len(ntok):.0f} (n={len(ntok)})")
        else:
            print(f"[cond {cond} already complete]")
    out.close()
    print("done:", OUT)

if __name__ == "__main__":
    main()
