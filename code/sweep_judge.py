# -*- coding: utf-8 -*-
"""
sweep_judge.py -- score the measured embedding-retriever methods (sweep_gen.py).

Same rubric / judge / temperature as control_judge / llm_route_judge (ADH_SYS, GPT-4.1-mini,
temp 0), scoring against the cued facet, so the result is comparable to body_top1 / all-facets.
Resumable.

Reads  data/generations_sweep.jsonl + data/instances_faithful.jsonl
Writes data/scores_sweep.jsonl (one row per draw: id, method, hit, adh)
Run with ~/mdrp-repro/venv/bin/python  (source ~/.openai_env first).
"""
import os, json
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

client = OpenAI()
JUDGE = "gpt-4.1-mini-2025-04-14"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{BASE}/data/scores_sweep.jsonl"

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
    gens = [json.loads(l) for l in open(f"{BASE}/data/generations_sweep.jsonl")]

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            try:
                d = json.loads(l); done.add((str(d["id"]), d["method"]))
            except Exception:
                pass
    todo = [g for g in gens if (str(g["id"]), g["method"]) not in done]
    print(f"{len(gens)} draws, {len(todo)} to score")

    def score_one(g):
        it = insts[g["id"]]
        true_facet = facets_of(it["schema"])[it["cued_index"]]
        return g["id"], g["method"], g.get("hit"), adherence(true_facet, it["stm"], g["response"])

    out = open(OUT, "a"); n = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for gid, method, hit, sc in ex.map(score_one, todo):
            out.write(json.dumps({"id": gid, "method": method, "hit": hit, "adh": sc},
                                 ensure_ascii=False) + "\n")
            out.flush(); n += 1
    out.close()
    print(f"scored {n} new -> {OUT}")

if __name__ == "__main__":
    main()
