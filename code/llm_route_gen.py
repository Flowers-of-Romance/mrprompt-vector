# -*- coding: utf-8 -*-
"""
llm_route_gen.py -- Phase 4 generation: MEASURE (not project) the task adherence of the
two LLM routers. The earlier route_sweep / llm_route runs only gave routing accuracy (R@1);
projecting adherence from body_top1's hit/miss means assumes the hit population is the same
across methods, which need not hold for the LLM routers. So we actually generate.

For each instance we inject the single facet the LLM router selected (sel) into the paper's
Magic-If (Fig.19) and generate with Qwen3-8B under the SAME settings as extcue_gen2
(thinking-OFF, max_new=1024, temp 0.7 / top_p 0.8). One arm per router:
  llm_router   : sel from data/llm_route_router.jsonl   (pick 1 of all facets, R@1=0.57)
  llm_twostage : sel from data/llm_route_twostage.jsonl (pick 1 of cue_situ top-3, R@1=0.51)

Single-facet injection mirrors the 'oracle' arm of extcue_gen2, so the adherence is directly
comparable to oracle (7.88) / allctx (7.43) / body_top1 (7.46).
Writes data/generations_llmroute.jsonl (resumable).
"""
import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faithful_render as R
import faithful_prompts as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES = f"{BASE}/data/instances_faithful.jsonl"
OUT = f"{BASE}/data/generations_llmroute.jsonl"
GEN = "Qwen/Qwen3-8B"
MAX_NEW = 1024
ARMS = {"llm_router": f"{BASE}/data/llm_route_router.jsonl",
        "llm_twostage": f"{BASE}/data/llm_route_twostage.jsonl"}

def facets_of(s):
    return s.get("scene_facets") or s.get("Personality", {}).get("scene_facets", [])

def _persona_head(schema, role):
    parts = [f"人物：{role}"]
    if schema.get("global_summary"):
        parts.append("概述：" + schema["global_summary"])
    ct = R._traits_of(schema)
    if ct:
        parts.append("核心特质：")
        for t in ct:
            parts.append(f"  - {t.get('trait','')}：{t.get('desc','')}")
    return parts

def facets_system(schema, facets, role):
    """LTM = core_traits + the single selected facet body, in the paper's Magic-If (Fig.19)."""
    parts = _persona_head(schema, role)
    parts.append("当前情境facet：")
    for f in facets:
        parts.append(f"  [{f.get('title','facet')}]")
        parts.append(R._facet_block(f, drop_keys=False))
    return F.magicif_system(role, "\n".join(parts))

def main():
    insts = {it["id"]: it for it in (json.loads(l) for l in open(INSTANCES))}
    routes = {arm: {r["id"]: r for r in (json.loads(l) for l in open(p))}
              for arm, p in ARMS.items()}

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
    for arm in ARMS:
        for it in insts.values():
            if (str(it["id"]), arm) in done:
                continue
            rec = routes[arm].get(it["id"])
            if rec is None:
                continue
            sel = rec["sel"]
            facets = facets_of(it["schema"])
            if not (0 <= sel < len(facets)):     # guard; no -1 in current data
                continue
            role = it["role"]
            system_text = facets_system(it["schema"], [facets[sel]], role)
            msgs = R.build_messages(system_text, it["stm"], role)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=True,
                                   temperature=0.7, top_p=0.8, pad_token_id=tok.eos_token_id)
            resp = tok.decode(g[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
            out.write(json.dumps({"id": it["id"], "role": role, "condition": arm,
                                  "cued_index": rec["cued_index"], "sel": sel,
                                  "hit": rec["hit"], "response": resp}, ensure_ascii=False) + "\n")
            out.flush()
        print(f"[arm {arm} done]")
    out.close()
    print("done:", OUT)

if __name__ == "__main__":
    main()
