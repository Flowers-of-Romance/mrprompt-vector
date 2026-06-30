# -*- coding: utf-8 -*-
"""
control_gen_multi.py -- power boost for the deconfound (control_gen.py).

The single-sample run gave the right direction (ceiling +0.41 = length +0.15 + near-distractor
+0.26) but each sub-effect was individually n.s. at n=100. Per-instance adherence is a noisy
integer (temp 0.7 sampling), so we average N_SAMPLES draws per (instance, condition) to shrink
the sampling-noise part of the paired SEM.

Same conditions / model / settings as control_gen.py. The existing 400 records (no "sample"
field) are treated as sample 0; this script tops every (id, condition) up to N_SAMPLES,
writing a "sample" index and a reproducible per-draw torch seed. Resumable.

Writes data/generations_control.jsonl (append). Run with ~/comfy-rocm/bin/python.
"""
import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faithful_render as R
import faithful_prompts as F
import control_gen as C   # reuse schema_for / build_pool / system_text_for / facets_of

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTANCES = f"{BASE}/data/instances_faithful.jsonl"
OUT = f"{BASE}/data/generations_control.jsonl"
GEN = "Qwen/Qwen3-8B"
MAX_NEW = 1024
N_SAMPLES = int(os.environ.get("N_SAMPLES", "5"))
COND_IDX = {c: i for i, c in enumerate(C.CONDS)}

def main():
    insts = [json.loads(l) for l in open(INSTANCES)]
    pool = C.build_pool(insts)

    # count existing samples per (id, cond); records without "sample" count as sample 0
    have = {}
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                d = json.loads(l)
            except Exception:
                continue
            have.setdefault((str(d["id"]), d["condition"]), set()).add(int(d.get("sample", 0)))

    todo = sum(max(0, N_SAMPLES - len(have.get((str(it["id"]), c), set())))
               for c in C.CONDS for it in insts)
    print(f"N_SAMPLES={N_SAMPLES}  draws to generate: {todo}")
    if todo == 0:
        print("nothing to do"); return

    tok = AutoTokenizer.from_pretrained(GEN)
    model = AutoModelForCausalLM.from_pretrained(GEN, dtype=torch.bfloat16, device_map="auto").eval()
    out = open(OUT, "a")
    for cond in C.CONDS:
        n_made = 0
        for it in insts:
            key = (str(it["id"]), cond)
            present = have.get(key, set())
            role = it["role"]
            schema = C.schema_for(cond, it, pool)
            system_text = C.system_text_for(schema, role)
            msgs = R.build_messages(system_text, it["stm"], role)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt").to(model.device)
            for s in range(N_SAMPLES):
                if s in present:
                    continue
                torch.manual_seed(int(it["id"]) * 1000 + COND_IDX[cond] * 100 + s)
                with torch.no_grad():
                    g = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=True,
                                       temperature=0.7, top_p=0.8, pad_token_id=tok.eos_token_id)
                resp = tok.decode(g[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
                out.write(json.dumps({"id": it["id"], "role": role, "condition": cond,
                                      "sample": s, "cued_index": it["cued_index"],
                                      "n_facets": len(C.facets_of(schema)),
                                      "prompt_tokens": int(ids.input_ids.shape[1]),
                                      "response": resp}, ensure_ascii=False) + "\n")
                out.flush(); n_made += 1
        print(f"[cond {cond}] generated {n_made} new draws")
    out.close()
    print("done:", OUT)

if __name__ == "__main__":
    main()
