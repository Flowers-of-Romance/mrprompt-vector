# -*- coding: utf-8 -*-
"""
faithful_render.py -- Stage C rendering: wrap each condition's LTM text in the paper's
VERBATIM generation system prompt (Fig.18 plain / Fig.19 Magic-If). Replaces the
self-made MAGIC_IF protocol in renderers.py.

Condition -> (LTM source, generation template):
  base               : raw narrative          + Fig.18 (plain)
  card               : Card-LTM   (Fig.14)     + Fig.18 (plain)
  mrprompt           : facet-LTM  (Fig.15)     + Fig.19 (Magic-If)
  mrprompt_anti      : facet-LTM, cued facet replaced by inverted facet  + Fig.19
  mrprompt_noscene   : facet-LTM with scene_facets removed               + Fig.19
  mrprompt_nokey     : facet-LTM with situation/cue_phrases dropped       + Fig.19
  mrprompt_wrongkey  : facet-LTM, cued facet's keys swapped with a neighbour + Fig.19

The variant logic (anti/noscene/nokey/wrongkey) is identical to renderers.py so the
cue-key ablation stays comparable; only the protocol wrapping changed.
"""
import copy, json
import faithful_prompts as F

FACET_FIELDS = ["situation", "social_role", "emotional_state",
                "behavior_pattern", "thinking_pattern", "cue_phrases"]

def _facets_of(schema):
    """schema may be flat or nested under Personality."""
    if "scene_facets" in schema:
        return schema.get("scene_facets", [])
    return schema.get("Personality", {}).get("scene_facets", [])

def _traits_of(schema):
    if "core_traits" in schema:
        return schema.get("core_traits", [])
    return schema.get("Personality", {}).get("core_traits", [])

def _facet_block(f, drop_keys=False):
    out = []
    for k in FACET_FIELDS:
        if drop_keys and k in ("situation", "cue_phrases"):
            continue
        v = f.get(k)
        if v:
            out.append(f"    {k}: {v if not isinstance(v, list) else '、'.join(map(str, v))}")
    return "\n".join(out)

def _variant_facets(schema, mode, cued_index=None, anti=None):
    facets = copy.deepcopy(_facets_of(schema))
    if mode == "no_scene":
        return []
    if mode == "anti" and cued_index is not None and anti is not None and 0 <= cued_index < len(facets):
        facets[cued_index] = {**facets[cued_index], **anti}
    if mode == "wrongkey" and cued_index is not None and len(facets) > 1:
        other = (cued_index + 1) % len(facets)
        facets[cued_index] = {**facets[cued_index],
                              "situation": facets[other].get("situation", ""),
                              "cue_phrases": facets[other].get("cue_phrases", [])}
    return facets

# ---------------------------------------------------------------- LTM text (role_information)
def facet_ltm_text(schema, mode="full", cued_index=None, anti=None):
    """Serialize the MRPrompt facet-LTM (Fig.15 product) to text. NO protocol here."""
    name = schema.get("name", "")
    parts = [f"人物：{name}"]
    if schema.get("global_summary"):
        parts.append("概述：" + schema["global_summary"])
    ct = _traits_of(schema)
    if ct:
        parts.append("核心特质：")
        for t in ct:
            parts.append(f"  - {t.get('trait','')}：{t.get('desc','')}")
    facets = _variant_facets(schema, mode, cued_index, anti)
    if facets:
        parts.append("情境facet（可按对话线索检索）：")
        drop = (mode == "nokey")
        for i, f in enumerate(facets):
            parts.append(f"  [{f.get('title','facet'+str(i))}]")
            parts.append(_facet_block(f, drop_keys=drop))
    return "\n".join(parts)

def card_ltm_text(card_schema):
    """Serialize the Card-LTM (Fig.14 product) to text."""
    name = card_schema.get("name", "")
    p = card_schema.get("Personality", {})
    obj = {
        "name": name,
        "Nickname": card_schema.get("Nickname", ""),
        "Relationships": card_schema.get("Relationships", []),
        "global_summary": card_schema.get("global_summary", ""),
        "core_traits": [t.get("trait", "") for t in p.get("core_traits", [])],
        "scene_facets": p.get("scene_facets", []),
    }
    return "【人物卡】\n" + json.dumps(obj, ensure_ascii=False, indent=2)

def base_narrative_text(schema):
    """Plain-prose role_information for the base condition."""
    name = schema.get("name", "")
    parts = [f"【人物】{name}"]
    if schema.get("global_summary"):
        parts.append(schema["global_summary"])
    traits = "、".join(t.get("trait", "") for t in _traits_of(schema))
    if traits:
        parts.append(f"性格上，他/她{traits}。")
    return "\n".join(parts)

# ---------------------------------------------------------------- system prompt per condition
def system_for(condition, schema, card_schema=None, cued_index=None, anti=None, role=None):
    # {role} must match the CharacterEval role key (= STM speaker label), which can differ
    # from the full name the constructor LLM wrote into the LTM's name field.
    role = role or schema.get("name", "")
    if condition == "base":
        return F.plain_system(role, base_narrative_text(schema))
    if condition == "card":
        info = card_ltm_text(card_schema) if card_schema else card_ltm_text({"name": role, **schema})
        return F.plain_system(role, info)
    mode = {"mrprompt": "full", "mrprompt_anti": "anti", "mrprompt_noscene": "no_scene",
            "mrprompt_nokey": "nokey", "mrprompt_wrongkey": "wrongkey"}.get(condition)
    if mode is None:
        raise ValueError(condition)
    info = facet_ltm_text(schema, mode, cued_index, anti)
    return F.magicif_system(role, info)

def build_messages(system_text, stm, role):
    """System carries persona+rules (Fig.18/19). User presents the STM dialogue turn."""
    user = (f"【对话上下文（STM）】最后一句是对话者的发言，接下来轮到你（{role}）回应：\n\n"
            f"{stm}\n\n请只输出{role}接下来要说的一句话。")
    return [{"role": "system", "content": system_text},
            {"role": "user", "content": user}]
