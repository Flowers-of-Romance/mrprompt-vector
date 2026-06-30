# -*- coding: utf-8 -*-
"""
llm_route.py -- LLM router via the authenticated `claude` CLI (Claude Opus 4.6),
no API key needed. Two methods:

  router    : pick 1 of all N facets given the STM (the routing ceiling; uses a DIFFERENT
              model than the GPT-4.1 labeler, so it is not circular with the labels).
  twostage  : pick 1 of the top-3 candidates from the cue_situ retriever (cheap RAG).

Each instance is one `claude --model claude-opus-4-6 -p ... --output-format json` call.
Writes data/llm_route_<method>.jsonl (resumable). Routing accuracy is computed by route_report.
"""
import os, sys, json, subprocess
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHOD = sys.argv[1] if len(sys.argv) > 1 else "router"
MODEL = "claude-opus-4-6"
OUT = f"{BASE}/data/llm_route_{METHOD}.jsonl"

def facets_of(s): return s.get("scene_facets") or s.get("Personality", {}).get("scene_facets", [])
def facet_desc(f):
    return (f"标题:{f.get('title','')} | 情境:{f.get('situation','')} | 社会角色:{f.get('social_role','')} | "
            f"情绪:{f.get('emotional_state','')} | 行为:{f.get('behavior_pattern','')} | 思维:{f.get('thinking_pattern','')}")

insts = {it["id"]: it for it in (json.loads(l) for l in open(f"{BASE}/data/instances_faithful.jsonl"))}
sel = {s["id"]: s for s in (json.loads(l) for l in open(f"{BASE}/data/extcue_selections_v2.jsonl"))}

def candidates(it):
    facets = facets_of(it["schema"])
    if METHOD == "twostage":
        cand = sel[it["id"]]["cue_situ"]["top3"]          # original indices of the 3 candidates
        return cand, [facets[j] for j in cand]
    return list(range(len(facets))), facets

def build_prompt(it):
    orig_idx, facets = candidates(it)
    lines = [f"[{k}] {facet_desc(f)}" for k, f in enumerate(facets)]
    return ("你是角色扮演的情境路由器。给定最近的对话(STM)和角色的若干情境切面(facet)，"
            "选出最符合当前对话、角色接下来应据此回应的那一个切面。\n\n"
            f"对话(STM):\n{it['stm']}\n\n候选切面:\n" + "\n".join(lines) +
            '\n\n只输出JSON，不要解释：{"pick": <切面编号整数>}'), orig_idx

def call(it):
    prompt, orig_idx = build_prompt(it)
    try:
        p = subprocess.run([
            "claude", "--model", MODEL, "-p", prompt, "--output-format", "json"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
        obj = json.loads(p.stdout)
        txt = obj.get("result", "")
        s = txt[txt.find("{"): txt.rfind("}")+1]
        pick_local = int(json.loads(s)["pick"])
        sel_orig = orig_idx[pick_local] if 0 <= pick_local < len(orig_idx) else -1
    except Exception as e:
        sel_orig = -1
    return {"id": it["id"], "cued_index": it["cued_index"], "sel": sel_orig,
            "hit": int(sel_orig == it["cued_index"])}

def main():
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["id"] for l in open(OUT)}
    todo = [it for it in insts.values() if it["id"] not in done]
    print(f"method={METHOD} model={MODEL} todo={len(todo)}")
    out = open(OUT, "a")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for n, rec in enumerate(ex.map(call, todo)):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            if (n+1) % 10 == 0: print(f"  {n+1}/{len(todo)}")
    out.close()
    recs = [json.loads(l) for l in open(OUT)]
    acc = sum(r["hit"] for r in recs) / len(recs)
    print(f"{METHOD} routing R@1 = {acc:.3f}  (n={len(recs)})")

if __name__ == "__main__":
    main()
