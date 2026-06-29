# -*- coding: utf-8 -*-
"""
extcue_gen2.py -- Phase 3 generation: ceiling + richer-key retrieve-then-generate.

Conditions (all thinking-OFF, max_new=1024, temp 0.7 / top_p 0.8, like mrprompt-repro):
  oracle             : inject the TRUE (cued) facet body directly        -> ceiling
  extcue_body_top1   : inject the top-1 facet by the BODY key (best R@1=0.35)
  extcue_cuesitu_top3: inject the top-3 facets by the CUE_SITU key (best R@3=0.70), model picks

Key x k interaction (Phase-3 routing): R@1 best = body, R@3 best = cue_situ. Thin keys lose
their causal real-vs-wrong gap at k=3, so top-3 must pair with a richer key -- hence cue_situ
for the top-3 arm here.

Baselines reused from mrprompt-repro (same 100 instances, same rubric):
  allctx = adh[mrprompt] (all facets in context)   base = adh[base]
Reads data/extcue_selections_v2.jsonl (body.top1 / body.top3) + instances.
Writes data/generations_extcue2.jsonl (resumable).
"""
import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faithful_render as R
import faithful_prompts as F

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES = f"{BASE}/data/instances_faithful.jsonl"
SELECTIONS = f"{BASE}/data/extcue_selections_v2.jsonl"
OUT = f"{BASE}/data/generations_extcue2.jsonl"
GEN = "Qwen/Qwen3-8B"
MAX_NEW = 1024

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
    """LTM = core_traits + the given facet bodies, wrapped in the paper's Magic-If (Fig.19)."""
    parts = _persona_head(schema, role)
    label = "当前情境facet：" if len(facets) == 1 else "候选情境facet（请按对话线索选择最匹配的一个）："
    parts.append(label)
    for f in facets:
        parts.append(f"  [{f.get('title','facet')}]")
        parts.append(R._facet_block(f, drop_keys=False))
    return F.magicif_system(role, "\n".join(parts))

CONDS = ("oracle", "extcue_body_top1", "extcue_cuesitu_top3")

def chosen_indices(cond, sel):
    if cond == "oracle":
        return [sel["cued_index"]]
    if cond == "extcue_body_top1":
        return [sel["body"]["top1"]]
    return list(sel["cue_situ"]["top3"])

def main():
    insts = [json.loads(l) for l in open(INSTANCES)]
    sel = {s["id"]: s for s in (json.loads(l) for l in open(SELECTIONS))}

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
    # condition-outer so 'oracle' (the ceiling) completes first and can be judged early
    for cond in CONDS:
        for n, it in enumerate(insts):
            if (str(it["id"]), cond) in done:
                continue
            facets = facets_of(it["schema"]); role = it["role"]; s = sel[it["id"]]
            idxs = chosen_indices(cond, s)
            fs = [facets[j] for j in idxs]
            system_text = facets_system(it["schema"], fs, role)
            msgs = R.build_messages(system_text, it["stm"], role)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=True,
                                   temperature=0.7, top_p=0.8, pad_token_id=tok.eos_token_id)
            resp = tok.decode(g[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
            hit = int(idxs[0] == s["cued_index"]) if cond == "extcue_body_top1" else None
            hit3 = int(s["cued_index"] in idxs) if cond == "extcue_cuesitu_top3" else None
            out.write(json.dumps({"id": it["id"], "role": role, "condition": cond,
                                  "cued_index": s["cued_index"], "sel": idxs,
                                  "hit": hit, "hit3": hit3, "response": resp}, ensure_ascii=False) + "\n")
            out.flush()
        print(f"[cond {cond} done]")
    out.close()
    print("done:", OUT)

if __name__ == "__main__":
    main()
