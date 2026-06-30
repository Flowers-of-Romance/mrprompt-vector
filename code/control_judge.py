# -*- coding: utf-8 -*-
"""
control_judge.py -- score the deconfound run (control_gen.py).

Same rubric / judge / temperature as llm_route_judge & extcue_judge (ADH_SYS, GPT-4.1-mini,
temp 0), scoring each response against the TRUE (cued) facet, so adherence is directly
comparable to oracle / allctx / far_c within this run and to the rest of the project.

Reads  data/generations_control.jsonl + data/instances_faithful.jsonl
Writes data/scores_control.jsonl (one record per id: adh per condition)
Run with ~/mdrp-repro/venv/bin/python  (openai installed; source ~/.openai_env first).
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
    gens = [json.loads(l) for l in open(f"{BASE}/data/generations_control.jsonl")]

    def score_one(g):
        it = insts[g["id"]]
        true_facet = facets_of(it["schema"])[it["cued_index"]]
        return g["id"], g["condition"], adherence(true_facet, it["stm"], g["response"])

    rows = {}
    arms = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for gid, cond, sc in ex.map(score_one, gens):
            rows.setdefault(gid, {"id": gid, "adh": {}})
            rows[gid]["adh"][cond] = sc
            arms.setdefault(cond, []).append(sc)

    with open(f"{BASE}/data/scores_control.jsonl", "w") as fo:
        for gid in rows:
            fo.write(json.dumps(rows[gid], ensure_ascii=False) + "\n")
    print(f"wrote scores for {len(rows)} instances")
    for cond, vals in arms.items():
        ss = [s for s in vals if s is not None]
        print(f"{cond:10s} mean_adh={sum(ss)/len(ss):.3f} (n={len(ss)})")

if __name__ == "__main__":
    main()
