# mrprompt-vector

A follow-up to [mrprompt-repro](https://github.com/Flowers-of-Romance/mrprompt-repro):
does making the cue **external** turn MRPrompt's "cue-addressable" recall into a real,
load-bearing mechanism?

## Background

mrprompt-repro found that MRPrompt's advertised mechanism — *cue-addressable* facet recall
(matching dialogue cues to facet cue-keys) — is **not supported in-context**: deleting or
scrambling the cue keys does not change the output, because every facet body is already in
the prompt, so the model matches on the body content and the short cue is bypassed.

The natural question (raised in §4.2 of that article): an address only means something when
its target is otherwise unreachable. So **take the facet bodies out of the prompt and make
the cue the only retrieval path** — store facets in an external memory keyed by the cue, and
inject only the matched body. Does the cue then carry weight?

This repo runs that experiment on the **same 100 instances** (same characters, same STM, same
Qwen3-8B, same adherence rubric), so the external numbers are directly comparable.

## Method

1. **Routing (Phase 1, `extcue_route.py`)** — embed the STM and each facet's cue with
   `BAAI/bge-m3`; retrieve the facet whose cue is nearest. Measure top-1 accuracy against the
   facet the constructor marked as cued, vs a `wrongkey` condition (the cued facet's cue is
   overwritten by a neighbour's, mirroring mrprompt-repro), vs chance (~1/7.5).

2. **Generation (Phase 2, `extcue_gen.py`)** — inject only the retrieved facet's body
   (with `core_traits`) into the paper's Magic-If prompt (Fig.19) and generate with Qwen3-8B
   (thinking-OFF, `max_new_tokens=1024`, temp 0.7 / top_p 0.8 — identical to mrprompt-repro).
   Conditions: `extcue` (real cue) and `extcue_wrongkey` (corrupted cue).

3. **Scoring (`extcue_judge.py`)** — single-response adherence (1–10) to the **true (cued)**
   facet, using mrprompt-repro's verbatim ADH rubric (GPT-4.1-mini, temp 0).

4. **Analysis (`extcue_analyze.py`)** — external `extcue − wrongkey` vs in-context
   `mrprompt − wrongkey`; adherence split by whether retrieval hit.

## Layout

```
code/   extcue_route.py  extcue_gen.py  extcue_judge.py  extcue_analyze.py
        faithful_render.py  faithful_prompts.py   (copied from mrprompt-repro)
data/   instances_faithful.jsonl              (100 instances, from mrprompt-repro)
        scores_faithful_incontext.jsonl       (in-context baseline)
        extcue_selections.jsonl  generations_extcue.jsonl  scores_extcue.jsonl  chart_data.json
```

## Run

```bash
PY=~/comfy-rocm/bin/python          # ROCm torch + transformers
$PY code/extcue_route.py            # Phase 1: routing accuracy + selections
$PY code/extcue_gen.py              # Phase 2: generation (Qwen3-8B)
OPENAI_API_KEY=... $PY code/extcue_judge.py
$PY code/extcue_analyze.py          # final numbers + chart data
```

## Results

See `article/` (JA/EN). Headline numbers in `extcue_results.json`.
