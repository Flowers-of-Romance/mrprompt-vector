# mrprompt-vector

A follow-up to [mrprompt-repro](https://github.com/Flowers-of-Romance/mrprompt-repro):
does making the cue **external** turn MRPrompt's "cue-addressable" recall into a real,
load-bearing mechanism — and if selection has value, can retrieval realize it?

Full writeup (abstract + charts):
[JA](https://flowers-of-romance.github.io/poptones/posts/mrprompt-vector/) /
[EN](https://flowers-of-romance.github.io/poptones/posts/en/mrprompt-vector/).

## Background

mrprompt-repro found that MRPrompt's advertised mechanism — *cue-addressable* facet recall
(matching dialogue cues to facet cue-keys) — is **not supported in-context**: deleting or
scrambling the cue keys does not change the output, because every facet body is already in
the prompt, so the model matches on the body content and the short cue is bypassed.

An address only means something when its target is otherwise unreachable. So **take the facet
bodies out of the prompt and make the cue the only retrieval path** — store facets in an
external memory keyed by the cue, inject only the matched body, and ask: does the cue then
carry weight, and does selecting one facet beat dumping all of them?

## Setup

Same 100 instances as mrprompt-repro (same characters, STM, cued facet), same generator
(Qwen3-8B, thinking-OFF, `max_new_tokens=1024`, temp 0.7 / top_p 0.8) and same adherence
rubric (ADH, GPT-4.1-mini, temp 0), so every number is directly comparable to the in-context
baseline.

Retrieval: the query is the **STM** (recent dialogue); the documents are each facet's **key**.
Embed both with `BAAI/bge-m3` (CLS), take the cosine nearest neighbour. The key has three
richness levels: `cue_only` (cue_phrases), `cue_situ` (cue_phrases + situation), `body`
(situation + emotional_state + behavior_pattern + thinking_pattern).

## Experiments & headline results

1. **Routing — causal but weak** (`extcue_route` / `extcue_gen` / `extcue_judge` /
   `extcue_analyze`). real > wrong for every key, so the external cue is causal (unlike
   in-context). But top-1 caps at R@1 = 0.35 (body key): one character's facets are
   semantically close, so the index is weak.

2. **top-k stays "addressing" only with a rich key** (`extcue_route2`). recall@3 = 0.65–0.70;
   the causal real − wrong gap at k=3 survives only for `cue_situ` / `body`, not `cue_only`.

3. **Final task — retrieval ties all-facets** (`extcue_gen2`). oracle − allctx = +0.45
   (single sample): injecting only the correct facet beats all seven, so selection has value.
   But body_top1 / cuesitu_top3 − allctx ≈ 0 (null): retrieval cannot realize it. Adherence is
   bimodal (hit ≈ oracle, miss ≈ base); the bottleneck is routing accuracy.

4. **Raising routing accuracy** (`route_sweep` + `sweep_gen` / `sweep_judge` / `sweep_report`;
   `llm_route*`). 11 embedding retrievers, measured (not projected): R@1 caps at 0.40 and none
   beats all-facets (all |z| < 2). An LLM two-stage router (retrieve `cue_situ` top-3 → Claude
   Opus picks one) beats all-facets by +0.36 (z = 2.5) and matches the oracle ceiling; the
   pick-one-of-all router is +0.19 (n.s.) — a higher miss floor sinks it.

5. **Deconfounding the ceiling** (`control_gen*` / `control_judge*` / `control_analyze*`). At
   5 samples/cell the oracle − allctx ceiling is +0.27 (smaller than the single-sample +0.45).
   A length-matched, distractor-free control (`oracle_dup`: cued facet repeated to allctx
   length) is indistinguishable from oracle (length contribution +0.00), so the whole ceiling
   is distractor cost (+0.27, z = 3.4) — not context length. Near vs far distractors make no
   difference: what matters is whether competing facets are present, not their proximity.

## Layout

```
code/
  faithful_render.py  faithful_prompts.py            prompt rendering (from mrprompt-repro)
  extcue_route.py  extcue_route2.py                  routing (bge-m3) + selections
  extcue_gen.py  extcue_gen2.py                      generation (Qwen3-8B)
  extcue_judge.py  extcue_analyze.py                 ADH scoring + analysis
  route_sweep.py                                     11-retriever routing sweep + selections
  sweep_gen.py  sweep_judge.py  sweep_report.py      measure the embedding retrievers
  llm_route.py  llm_route_gen.py  llm_route_judge.py LLM router / two-stage (Claude Opus CLI)
  control_gen.py  control_judge.py  control_analyze.py            ceiling deconfound (1 sample)
  control_gen_multi.py  control_judge_multi.py  control_analyze_multi.py   (5 samples/cell)
data/
  instances_faithful.jsonl  scores_faithful_incontext.jsonl       from mrprompt-repro
  extcue_selections*.jsonl  generations_extcue*.jsonl  scores_extcue.jsonl
  route_sweep.json  route_sweep_selections.jsonl                  (sweep R@1/R@3 + picks)
  generations_sweep.jsonl  scores_sweep.jsonl  sweep_measured.json
  llm_route_*.jsonl  generations_llmroute.jsonl  scores_llmroute.jsonl  phase4_*.json
  generations_control.jsonl  scores_control*.jsonl  control_decomp*.json
```

## Run

```bash
PY=~/comfy-rocm/bin/python                 # ROCm torch + transformers (gen, embeddings)
JUDGE=~/mdrp-repro/venv/bin/python         # OpenAI judge; first: source ~/.openai_env

# Results 1–3: routing, top-k, final-task ceiling
$PY code/extcue_route.py; $PY code/extcue_gen.py; $JUDGE code/extcue_judge.py; $PY code/extcue_analyze.py
$PY code/extcue_route2.py; $PY code/extcue_gen2.py

# Result 4: embedding sweep (measured) + LLM router
$PY code/route_sweep.py; $PY code/sweep_gen.py; $JUDGE code/sweep_judge.py; $JUDGE code/sweep_report.py
$PY code/llm_route.py; $PY code/llm_route_gen.py; $JUDGE code/llm_route_judge.py

# Result 5: ceiling deconfound, 5 samples/cell
N_SAMPLES=5 $PY code/control_gen_multi.py; $JUDGE code/control_judge_multi.py; $JUDGE code/control_analyze_multi.py
```

The LLM router (`llm_route.py`) calls `claude --model claude-opus-4-6 -p ... --output-format json`
(the authenticated CLI is the only Claude path here). All scripts are location-independent
(`BASE = dirname(dirname(__file__))`).
