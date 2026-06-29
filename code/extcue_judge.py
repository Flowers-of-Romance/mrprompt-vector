# -*- coding: utf-8 -*-
"""
extcue_judge.py -- Phase 2 scoring for the external-cue method.

Single-response adherence (1-10) to the TRUE (cued) facet, using the SAME rubric as
mrprompt-repro (judge_faithful.ADH_SYS, GPT-4.1-mini, temp 0). So adherence is directly
comparable to the in-context numbers in scores_faithful.jsonl.

Reads  data/generations_extcue.jsonl + data/instances_faithful.jsonl
Writes data/scores_extcue.jsonl  (one record per id: adh per condition + retrieval hit)
"""
import os, json
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

client = OpenAI()
JUDGE = "gpt-4.1-mini-2025-04-14"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ADH_SYS = """你是严格的角色扮演评测员。给定一个"目标情境facet"（其 social_role/emotional_state/behavior_pattern/thinking_pattern 代表角色在该情境下应有的表现）、对话上下文(STM)、以及角色生成的一句回应。
评估：该回应在多大程度上体现并契合"目标facet"所描述的社会姿态、情绪、行为方式与思维倾向？
打分 1-10：10=高度契合该facet；5=中性/不明显；1=完全不契合甚至体现相反倾向。
只依据是否契合该facet评分，不评价文采或流畅度。只输出 JSON：{"score":1到10的整数,"reason":"简短理由"}"""

def facets_of(schema):
    if "scene_facets" in schema:
        return schema.get("scene_facets", [])
    return schema.get("Personality", {}).get("scene_facets", [])

def facet_desc(f):
    return {k: f.get(k) for k in ("title", "situation", "social_role",
            "emotional_state", "behavior_pattern", "thinking_pattern")}

def adherence(facet, stm, response):
    if not response or not response.strip():
        return None
    user = f"目标facet:\n{json.dumps(facet_desc(facet), ensure_ascii=False)}\n\nSTM:\n{stm}\n\n角色回应:\n{response}"
    r = client.chat.completions.create(
        model=JUDGE, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": ADH_SYS}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content).get("score")

def hit_of(g):
    """top-1 conditions store 'hit'; top-3 stores 'hit3'; oracle has neither."""
    return g.get("hit", g.get("hit3"))

def main():
    insts = {it["id"]: it for it in (json.loads(l) for l in open(f"{BASE}/data/instances_faithful.jsonl"))}
    gens = []
    for fn in ("generations_extcue.jsonl", "generations_extcue2.jsonl"):
        p = f"{BASE}/data/{fn}"
        if os.path.exists(p):
            gens += [json.loads(l) for l in open(p)]

    def score_one(g):
        it = insts[g["id"]]
        true_facet = facets_of(it["schema"])[it["cued_index"]]
        return g["id"], g["condition"], hit_of(g), adherence(true_facet, it["stm"], g["response"])

    rows = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for gid, cond, hit, sc in ex.map(score_one, gens):
            rows.setdefault(gid, {"id": gid, "adh": {}, "hit": {}})
            rows[gid]["adh"][cond] = sc
            rows[gid]["hit"][cond] = hit

    with open(f"{BASE}/data/scores_extcue.jsonl", "w") as fo:
        for gid in rows:
            fo.write(json.dumps(rows[gid], ensure_ascii=False) + "\n")
    print(f"wrote scores for {len(rows)} instances")

if __name__ == "__main__":
    main()
