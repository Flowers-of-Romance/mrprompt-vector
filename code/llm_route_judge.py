# -*- coding: utf-8 -*-
"""
llm_route_judge.py -- Phase 4 scoring for the two LLM-router arms.

Same rubric / judge as extcue_judge (ADH_SYS, GPT-4.1-mini, temp 0), so the adherence is
directly comparable to oracle / allctx / body_top1 in scores_extcue.jsonl.

Reads  data/generations_llmroute.jsonl + data/instances_faithful.jsonl
Writes data/scores_llmroute.jsonl (one record per id: adh + hit per arm) and prints the
per-arm mean adherence and the hit/miss split.
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

def main():
    insts = {it["id"]: it for it in (json.loads(l) for l in open(f"{BASE}/data/instances_faithful.jsonl"))}
    gens = [json.loads(l) for l in open(f"{BASE}/data/generations_llmroute.jsonl")]

    def score_one(g):
        it = insts[g["id"]]
        true_facet = facets_of(it["schema"])[it["cued_index"]]
        return g["id"], g["condition"], g.get("hit"), adherence(true_facet, it["stm"], g["response"])

    rows = {}
    arms = {}   # cond -> list of (hit, score)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for gid, cond, hit, sc in ex.map(score_one, gens):
            rows.setdefault(gid, {"id": gid, "adh": {}, "hit": {}})
            rows[gid]["adh"][cond] = sc
            rows[gid]["hit"][cond] = hit
            arms.setdefault(cond, []).append((hit, sc))

    with open(f"{BASE}/data/scores_llmroute.jsonl", "w") as fo:
        for gid in rows:
            fo.write(json.dumps(rows[gid], ensure_ascii=False) + "\n")
    print(f"wrote scores for {len(rows)} instances")
    for cond, vals in arms.items():
        ss = [s for _, s in vals if s is not None]
        mean = sum(ss) / len(ss)
        hit = [s for h, s in vals if h == 1 and s is not None]
        miss = [s for h, s in vals if h == 0 and s is not None]
        hm = sum(hit) / len(hit) if hit else float("nan")
        mm = sum(miss) / len(miss) if miss else float("nan")
        print(f"{cond:14s} mean_adh={mean:.3f} (n={len(ss)})  hit={hm:.3f} (n={len(hit)})  miss={mm:.3f} (n={len(miss)})")

if __name__ == "__main__":
    main()
