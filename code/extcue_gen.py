# -*- coding: utf-8 -*-
"""
extcue_gen.py -- Phase 2 generation for the external-cue method.

Unlike in-context MRPrompt (which puts ALL facets in the prompt), here the cue is the
only routing path: we retrieve ONE facet by matching the STM against each facet's cue,
then inject only that facet's body (with core_traits) into the paper's Magic-If prompt,
and generate. Two conditions:

  extcue           : retrieve with the real cue keys
  extcue_wrongkey  : the cued facet's cue is overwritten by a neighbour's before retrieval
                     (mirrors mrprompt-repro's wrongkey) -> retrieval is misled

Same model / budget / sampling as mrprompt-repro (Qwen3-8B, thinking-OFF, max_new=1024,
temp 0.7 / top_p 0.8), so adherence is comparable to the in-context numbers there.

Stage 1 (bge-m3) computes selections, frees GPU; Stage 2 (Qwen3-8B) generates.
Writes data/generations_extcue.jsonl (resumable).
"""
import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faithful_render as R
import faithful_prompts as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES = f"{BASE}/data/instances_faithful.jsonl"
SELECTIONS = f"{BASE}/data/extcue_selections.jsonl"   # produced by extcue_route.py
OUT = f"{BASE}/data/generations_extcue.jsonl"
GEN = "Qwen/Qwen3-8B"
MAX_NEW = 1024

def facets_of(schema):
    if "scene_facets" in schema:
        return schema.get("scene_facets", [])
    return schema.get("Personality", {}).get("scene_facets", [])

def single_facet_system(schema, facet, role):
    """LTM = core_traits + exactly ONE facet body, wrapped in the paper's Magic-If (Fig.19)."""
    parts = [f"人物：{role}"]
    if schema.get("global_summary"):
        parts.append("概述：" + schema["global_summary"])
    ct = R._traits_of(schema)
    if ct:
        parts.append("核心特质：")
        for t in ct:
            parts.append(f"  - {t.get('trait','')}：{t.get('desc','')}")
    parts.append("当前情境facet：")
    parts.append(f"  [{facet.get('title','facet')}]")
    parts.append(R._facet_block(facet, drop_keys=False))
    return F.magicif_system(role, "\n".join(parts))

def main():
    insts = [json.loads(l) for l in open(INSTANCES)]
    sel = {s["id"]: s for s in (json.loads(l) for l in open(SELECTIONS))}
    hit = {c: sum(sel[i][c]["hit"] for i in sel) / len(sel) for c in ("extcue", "extcue_wrongkey")}
    print(f"{len(insts)} instances | retrieval hit-rate: extcue={hit['extcue']:.3f}  wrongkey={hit['extcue_wrongkey']:.3f}")

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
    for n, it in enumerate(insts):
        facets = facets_of(it["schema"]); role = it["role"]
        for cond in ("extcue", "extcue_wrongkey"):
            if (str(it["id"]), cond) in done:
                continue
            s = sel[it["id"]][cond]; chosen = facets[s["sel"]]
            system_text = single_facet_system(it["schema"], chosen, role)
            msgs = R.build_messages(system_text, it["stm"], role)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=True,
                                   temperature=0.7, top_p=0.8, pad_token_id=tok.eos_token_id)
            resp = tok.decode(g[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
            out.write(json.dumps({"id": it["id"], "role": role, "condition": cond,
                                  "cued_index": it["cued_index"], "sel_index": s["sel"],
                                  "hit": s["hit"], "response": resp}, ensure_ascii=False) + "\n")
            out.flush()
        print(f"[{n+1}/{len(insts)}] {role}")
    out.close()
    print("done:", OUT)

if __name__ == "__main__":
    main()
