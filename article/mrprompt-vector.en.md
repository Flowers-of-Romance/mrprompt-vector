# Making MRPrompt's cue external

## Starting point

The earlier study (mrprompt-repro) showed that MRPrompt's "cue-addressable facet recall" is unsupported in-context. Deleting or scrambling the cue keys (cue_phrases) did not move the output, and attention landed on the facet body, not the keys. The reason is simple: because every facet body is in the prompt, the model can pick the needed facet from the body content, and the short key is bypassed. An address only means something when its target is otherwise unreachable.

If that is right, then taking the facet bodies out of the prompt and making the cue the only retrieval path should let the cue-matched facet be recalled. Store facets in an external memory keyed by the cue, and inject only the body whose key matches the dialogue — a retrieval (RAG) setup. We implement this on the same 100 instances, the same Qwen3-8B, and the same scoring as mrprompt-repro, and measure two things.

1. Does the cue, made external, work as an address (did we retrieve the right facet)?
2. Does that retrieve-then-generate beat putting all facets in context (in-context, the competitor the reproduction actually validated) on the final task?

Question 1 is retrieval accuracy; question 2 is task quality. They are different, and the bar for 2 is high. At this scale each character has ~7 facets, which fit in context. Since they fit, handing over a few retrieved facets can be a noisier reduction than handing over all of them.

## Method

- Data: mrprompt-repro's instances_faithful.jsonl (100 instances, 7–8 facets per character, with the cued facet and STM) used as is.
- Retrieval: embed the STM and each facet's key with `BAAI/bge-m3` (multilingual, CLS pooling, bf16/iGPU) and take the cosine nearest neighbour. Three key-richness levels:
  - `cue_only`: cue_phrases only (closest to the paper's "cue keys")
  - `cue_situ`: cue_phrases + situation
  - `body`: situation + emotional_state + behavior_pattern + thinking_pattern (the facet content)
- Generation: inject the retrieved facet body (+ core_traits) into the paper's Magic-If (Fig.19) and generate with Qwen3-8B (thinking-OFF, max_new_tokens=1024, temperature 0.7 / top_p 0.8). All identical to mrprompt-repro, thinking held OFF (so it doesn't mix with the CoT main effect).
- Scoring: single-response adherence (1–10) to the true (cued) facet, with mrprompt-repro's ADH rubric (GPT-4.1-mini, temp 0).
- Baselines (same instances, reused from mrprompt-repro): `allctx` = all facets in context (= the reproduction's mrprompt, adherence 7.43), `base` = prose without facets (7.23).

## Result 1: external, the cue becomes causal — but it is a weak index

Using the STM as query, we retrieve by each facet's key and measure whether the top-1 is the cued facet (n=100, chance@1=0.139 ≈ 7 facets). We compare against a wrongkey condition (the cued facet's key overwritten by a neighbour's, the same operation as mrprompt-repro).

| key | R@1 real | R@1 wrong | R@3 real | R@3 wrong |
|---|---|---|---|---|
| cue_only | 0.300 | 0.180 | 0.650 | 0.630 |
| cue_situ | 0.330 | 0.220 | 0.700 | 0.530 |
| body | 0.350 | 0.160 | 0.670 | 0.470 |

(chance@1=0.139, chance@3=0.418)

- real > wrong for every key. In contrast to in-context, where wrongkey had no effect (mrprompt − wrongkey = +0.03, null), externalizing makes the cue causal: the right key pulls the right body, a wrong key pulls a wrong one.
- But top-1 tops out at 0.350 even with the best key (body). Narrowing the query to the last utterance does not help (the full STM is best). This is not the query choice but the fact that one character's facets are semantically close (chance 0.139 over 7), so discrimination is intrinsically hard. Even using the body itself as the key, only 35% of top-1 are correct.
- Richer keys raise R@1 (cue_only 0.300 → cue_situ 0.330 → body 0.350). That is roughly the top-1 ceiling of retrieval-MRPrompt.

## Result 2: top-k stays "addressing" only paired with a rich key

recall@3 (cued present in the top-3) is 0.65–0.70, clearly above chance@3=0.418: widen to three and the cued facet is in there ~70% of the time. But the causal gap at k=3 (R@3 real − wrong) depends on key richness.

| key | R@3 real − wrong |
|---|---|
| cue_only | +0.02 (≈ null) |
| cue_situ | +0.17 |
| body | +0.20 |

With cue_only at top-3, whether the cue is correct barely matters (a wrong key still retrieves 0.63). The moment k widens, a thin key degrades from "addressing by cue" to "just grabbing three semantically near facets." To use top-3 while keeping the mechanism under test (addressing by cue), the key must be richer.

There is also a key × k interaction: the best R@1 is body (0.350), but the best R@3 is cue_situ (0.700; body 0.670). The optimal key flips with k. In the task evaluation below we use body for top-1 and cue_situ for top-3.

## Result 3: on the final task — can retrieve-then-generate beat all-facets-in-context?

So far this is "did we retrieve." The verdict is the final-task adherence (1–10 to the cued facet), on the same 100 instances, thinking OFF, output budget 1024. First the maximum uplift perfect routing could buy: oracle (the true facet injected directly) minus allctx (all facets). If this is near 0, then retrieving correctly equals putting everything in — routing has no value at this scale — settled before we even ask about retriever accuracy.

| condition | adherence |
|---|---|
| base (prose, no facets) | 7.23 |
| allctx (all facets, in-context) | 7.43 |
| oracle (cued facet only) | 7.88 |
| body_top1 (top-1 retrieval, body key) | 7.46 |
| cuesitu_top3 (top-3 retrieval, cue_situ key) | 7.50 |

- oracle − allctx = +0.45 ±0.14 (n=100, ~3.3 SEM). Against the prediction, it is not 0. Injecting only the correct facet beats injecting all seven. In all-facets, the other six act as faint distractors and blunt the alignment. So selection does have value at this scale — perfect routing leaves +0.45 on the table.
- But retrieval does not beat allctx. body_top1 − allctx = +0.03 ±0.16, cuesitu_top3 − allctx = +0.07 ±0.17, both null. It stops short of the ceiling (+0.45).
- The reason is routing accuracy. Split adherence by whether retrieval was correct and it is bimodal:

| condition | when hit | when miss |
|---|---|---|
| body_top1 | 7.74 (n=35) | 7.31 (n=65) |
| cuesitu_top3 | 7.61 (n=70) | 7.23 (n=30) |

On a hit it lands near oracle (7.6–7.7), on a miss near base (7.2–7.3). The average ties allctx because the hit/miss weighting (35–70% hit) lands exactly there. The bottleneck is routing accuracy, not the value of the facet: with only 35% top-1 / 70% top-3 correct (one character's facets are semantically close), the +0.45 ceiling is out of reach.

Finally, the causality of the cue at the adherence level. Comparing cue_only top-1 with its wrongkey: extcue − wrongkey = +0.23 ±0.14, against in-context's mrprompt − wrongkey = +0.03 (null), with real(0.30) > wrong(0.18) at the routing level too. Weak, but external, the right key pulls the right body.

## Result 4: how far does the final task move once routing accuracy is raised?

Result 3 located the bottleneck at routing accuracy. So how far does adherence move once a means of raising it is added? We test two. One swaps the embedding retriever to span a range of top-1 accuracy (R@1). The other is an LLM router that reads the facet bodies and selects. For the first we do not re-measure generation/scoring per method; we convert R@1 to adherence via body_top1's measured hit/miss means (7.74 / 7.31) — a projection. The second is the decision point, so we actually inject the selected facet into Qwen3-8B, generate and score it — measured. The measured anchors are oracle (true facet injected directly) = 7.88 and all-facets = 7.43.

### The span of embedding retrievers (projection from R@1)

Eleven embedding-based methods on the same 100 instances: random, bm25 (jieba), bge-m3 dense (keys cue / cue_situ / body), a colbert approximation over bge-m3 (token-level max-sim), hybrid RRF (bm25 + bge-m3 body), bge-large-zh-v1.5 (Chinese-specialized, body key), difference-vector (a facet's body minus the mean of the same character's other facet bodies), bge-reranker-v2-m3 (cross-encoder), and mean-pooled last hidden states of Qwen3-8B (body key).

| method | R@1 | R@3 | projected adherence |
|---|---|---|---|
| bge-large-zh body | 0.40 | 0.66 | 7.48 |
| difference-vector | 0.38 | 0.66 | 7.47 |
| bge-m3 colbert-approx | 0.38 | 0.71 | 7.47 |
| bge-m3 dense body | 0.35 | 0.67 | 7.46 |
| bge-m3 dense cue_situ | 0.33 | 0.70 | 7.45 |
| bge-m3 dense cue | 0.30 | 0.65 | 7.44 |
| hybrid RRF(bm25+dense) | 0.28 | 0.65 | 7.43 |
| bge-reranker-v2-m3 | 0.27 | 0.59 | 7.43 |
| qwen3-8b hidden body | 0.27 | 0.63 | 7.43 |
| bm25 (jieba) | 0.19 | 0.45 | 7.39 |
| random | 0.14 | 0.42 | 7.37 |

R@1 falls in 0.19–0.40, concentrated in 0.27–0.40 excluding bm25. Chinese-specialized (0.40), the cross-encoder (0.27), and the generator's hidden states (0.27) do not exceed 0.40. Projected adherence stays within ±0.05 of all-facets (7.43). The Result 3 null does not move with the choice of embedding method; the semantic proximity of one character's facets does not shrink by swapping the embedding.

This projection has an assumption: the hit/miss means (7.74 / 7.31) are taken from body_top1 and held constant across methods. If the hit population differs by method, the conversion is off. The LLM router below is exactly that case.

### The LLM router, measured

Claude Opus 4.6 (via CLI, a different family from the Qwen3-8B generator) selects a facet, and the one it picks is injected into Qwen3-8B, generated and scored. router picks one of all facets; two-stage narrows to the top-3 of cue_situ retrieval, then picks one.

| condition | R@1 | measured adherence | vs all-facets |
|---|---|---|---|
| oracle (cued facet) | 1.00 | 7.88 | +0.45 ±0.14 (z=3.3) |
| llm_twostage (top-3→1) | 0.51 | 7.79 | +0.36 ±0.14 (z=2.5) |
| llm_router (all→1) | 0.57 | 7.62 | +0.19 ±0.15 (z=1.3) |
| cuesitu_top3 | — | 7.50 | +0.07 (null) |
| body_top1 | 0.35 | 7.46 | +0.03 (null) |
| all-facets | — | 7.43 | — |

- two-stage beats all-facets by +0.36 ±0.14 (z=2.5), and its gap to oracle is −0.09 ±0.11 (null): nearly indistinguishable from the +0.45 ceiling. The uplift embedding retrieval could not reach (body_top1 − allctx = +0.03, null), two-stage takes almost in full.
- router alone is +0.19 ±0.15 (z=1.3): same direction, but not significant against all-facets. two-stage minus router is +0.17 ±0.12 (null) — router has the higher R@1 (0.57), yet two-stage has the higher adherence (7.79).
- The reason is the floor on a miss. Split by hit/miss: router scores 8.11 on a hit, 6.98 on a miss; two-stage 8.26 and 7.31. The ceiling is high for both (8.1–8.3); the difference is the miss. router picks from all facets, so a miss drops below base (7.23). two-stage picks within cue_situ's top-3, so even a miss stays on a conversation-near facet and bottoms out at 7.31. Narrowing with top-k retrieval to hold the floor and letting the LLM take the ceiling is what moves the mean significantly.
- Against the projection: projection had router 7.56 > two-stage 7.53, the reverse of the measured order (two-stage 7.79 > router 7.62). Projection represented the hit group by body_top1's 7.74, whereas the LLM router's hit group scored 8.1–8.3. The method-dependence of the hit population was large enough to flip the order — which is why the LLM router was measured, not projected.

Raising routing accuracy moves the very bottleneck Result 3 identified: swapping the embedding caps R@1 at 0.40 and ties all-facets, but an LLM two-stage router that reads the facet bodies beats all-facets significantly and matches the oracle ceiling. The cost is ~8 s and ~$0.08 per LLM-router call, two orders above embedding retrieval.

Caveat: only the two LLM-router arms are measured; the eleven embedding methods remain a projection from body_top1's hit/miss. The GPT-4.1-mini judge bias is common to all conditions.

## Takeaways

1. In-context (mrprompt-repro): the cue is inert. With the body fully in context, it is bypassed.
2. External (this study): the cue becomes causal (extcue − wrongkey = +0.23; real > wrong in routing). And injecting only the correct facet beats all-facets by +0.45 — selection itself has value. Fitting in context does not mean dumping everything is best.
3. But current retrieval cannot realize that value. One character's facets are semantically close, so even with the body as the key, top-1 caps at 35% and top-3 at 70%. Adherence splits into near-oracle on a hit and near-base on a miss, and the average ties all-facets (null). The bottleneck is routing accuracy, not the value of selection.

4. Routing accuracy can be raised. Swapping the embedding caps R@1 at 0.40, but an LLM two-stage router (narrow to cue_situ's top-3, then pick one) beats all-facets by +0.36 ±0.14 (z=2.5), with the gap to the oracle ceiling at −0.09 (null). The value of selection (+0.45) becomes reachable once you change the router.

The resolution: adding a top-1 embedding retrieval only ties all-facets and routing is hard to get right, but the value of selection (+0.45) is real and there are two ways to it. (a) Make the router an LLM two-stage: narrowing to cue_situ's top-3 and letting the LLM pick one beat all-facets significantly and matched the oracle ceiling here, where an embedding top-1 does not reach. (b) Facets no longer fit in context, where all-facets is off the table and retrieval, however coarse, becomes necessary. The LLM two-stage costs ~8 s and ~$0.08 per call, so when facets fit, all-facets + CoT is still the cheaper default.

Caveat: allctx (7 facets) has a longer input than body_top1/cuesitu_top3, so a length asymmetry could enter the quality gap. But oracle (1 facet) beats allctx (7 facets) by +0.45 with a shorter context, so length works against the shorter conditions; the ceiling is not explained by context length. Scoring is GPT-4.1-mini adherence and inherits judge bias. Single model (Qwen3-8B), 100 instances, Chinese characters — preliminary.

---

Data and code in this repository (mrprompt-vector). The MS-FA / adherence metric is borrowed verbatim from mrprompt-repro.
