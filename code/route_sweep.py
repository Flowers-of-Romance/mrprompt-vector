# -*- coding: utf-8 -*-
"""
route_sweep.py -- Phase 4: how far can the RETRIEVER go? Map routing accuracy across a
spectrum of retrieval methods (lexical -> dense -> specialized -> late-interaction ->
cross-encoder), on the same 100 instances. Projects task adherence from routing accuracy
via the measured hit/miss means (body_top1: hit 7.74 / miss 7.31; top3: 7.61 / 7.23).

Methods (local; the LLM router / two-stage run separately via the claude CLI):
  random, bm25, bge-m3 dense {cue, cue_situ, body}, bge-m3 colbert-approx,
  hybrid RRF(bm25 + bge-m3 body), bge-large-zh-v1.5 body, difference-vector (bge-m3 body),
  bge-reranker-v2-m3 (cross-encoder), Qwen3-8B hidden-state body.
  (bge-m3 native sparse needs FlagEmbedding -> skipped; hybrid uses RRF instead.)

Query = full STM. Reports R@1, R@3, and projected top-1 adherence. Writes route_sweep.json.
"""
import os, json, math
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INST = f"{BASE}/data/instances_faithful.jsonl"
HIT1, MISS1 = 7.74, 7.31      # body_top1 hit/miss adherence (measured)

def facets_of(s): return s.get("scene_facets") or s.get("Personality", {}).get("scene_facets", [])
def cue_text(f): return "；".join(f.get("cue_phrases") or []) or f.get("title", "")
def situ_text(f): return (("；".join(f.get("cue_phrases") or [])) + "。" + (f.get("situation") or "")).strip("。")
def body_text(f):
    return "。".join(str(f.get(k) or "") for k in ("situation","emotional_state","behavior_pattern","thinking_pattern")).strip("。")

insts = [json.loads(l) for l in open(INST)]
N = len(insts)
results = {}   # name -> {"R@1":x, "R@3":y}

def record(name, picks_topk):
    # picks_topk: list over instances of (ranked index list)
    r1 = sum(1.0 for i, it in enumerate(insts) if picks_topk[i][0] == it["cued_index"]) / N
    r3 = sum(1.0 for i, it in enumerate(insts) if it["cued_index"] in picks_topk[i][:3]) / N
    results[name] = {"R@1": round(r1, 3), "R@3": round(r3, 3),
                     "proj_adh_top1": round(r1*HIT1 + (1-r1)*MISS1, 3)}
    print(f"{name:26s} R@1={r1:.3f}  R@3={r3:.3f}  proj_adh={r1*HIT1+(1-r1)*MISS1:.3f}")

# ---- random (analytic) ----
import statistics
results["random"] = {"R@1": round(statistics.mean(1/len(facets_of(it['schema'])) for it in insts), 3),
                     "R@3": round(statistics.mean(min(3,len(facets_of(it['schema'])))/len(facets_of(it['schema'])) for it in insts), 3),
                     "proj_adh_top1": None}
print(f"{'random':26s} R@1={results['random']['R@1']:.3f}  R@3={results['random']['R@3']:.3f}")

# ---- BM25 (jieba) ----
import jieba
from rank_bm25 import BM25Okapi
def tok(s): return [w for w in jieba.cut(s) if w.strip()]
bm25_topk = []
for it in insts:
    facets = facets_of(it["schema"])
    corpus = [tok(body_text(f)) for f in facets]
    bm = BM25Okapi(corpus)
    scores = bm.get_scores(tok(it["stm"]))
    bm25_topk.append(sorted(range(len(facets)), key=lambda j: -scores[j]))
record("bm25", bm25_topk)
bm25_rank = bm25_topk  # reuse for hybrid

# ---- bge-m3 dense + colbert-approx + difference ----
class BGE:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
    @torch.no_grad()
    def cls(self, texts, bs=16):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            h = self.m(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
        return torch.cat(out)
    @torch.no_grad()
    def tokens(self, text):
        enc = self.tok([text], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        h = self.m(**enc).last_hidden_state[0]
        mask = enc["attention_mask"][0].bool()
        return torch.nn.functional.normalize(h[mask].float(), dim=-1).cpu()

bge = BGE("BAAI/bge-m3")
Qd = bge.cls([it["stm"] for it in insts])
def dense_topk(keyfn):
    out = []
    for i, it in enumerate(insts):
        facets = facets_of(it["schema"])
        K = bge.cls([keyfn(f) for f in facets])
        out.append((K @ Qd[i]).argsort(descending=True).tolist())
    return out
record("bge-m3 dense cue", dense_topk(cue_text))
record("bge-m3 dense cue_situ", dense_topk(situ_text))
bge_body = dense_topk(body_text); record("bge-m3 dense body", bge_body)

# difference-vector: facet body minus mean of the other facets' bodies, match query
diff_topk = []
for i, it in enumerate(insts):
    facets = facets_of(it["schema"]); n = len(facets)
    K = bge.cls([body_text(f) for f in facets])
    D = torch.nn.functional.normalize(K - (K.sum(0, keepdim=True) - K) / max(1, n-1), dim=-1)
    diff_topk.append((D @ Qd[i]).argsort(descending=True).tolist())
record("difference-vector", diff_topk)

# colbert-approx: token-level max-sim (query tokens vs facet-body tokens) from bge-m3
col_topk = []
Qtok = [bge.tokens(it["stm"]) for it in insts]
for i, it in enumerate(insts):
    facets = facets_of(it["schema"]); q = Qtok[i]
    sc = []
    for f in facets:
        d = bge.tokens(body_text(f))
        sc.append((q @ d.T).max(dim=1).values.sum().item())
    col_topk.append(sorted(range(len(facets)), key=lambda j: -sc[j]))
record("bge-m3 colbert-approx", col_topk)

# hybrid: RRF(bm25, bge-m3 body)
def rrf(ranks_a, ranks_b, k=60):
    out = []
    for i in range(N):
        n = len(facets_of(insts[i]["schema"]))
        ra = {idx: r for r, idx in enumerate(ranks_a[i])}
        rb = {idx: r for r, idx in enumerate(ranks_b[i])}
        score = {j: 1/(k+ra[j]) + 1/(k+rb[j]) for j in range(n)}
        out.append(sorted(range(n), key=lambda j: -score[j]))
    return out
record("hybrid RRF(bm25+dense)", rrf(bm25_rank, bge_body))

del bge.m; bge = None; torch.cuda.empty_cache()

# ---- bge-large-zh-v1.5 (chinese-specialized, CLS, query instruction) ----
QINSTR = "为这个句子生成表示以用于检索相关文章："
zh = BGE("BAAI/bge-large-zh-v1.5")
Qz = zh.cls([QINSTR + it["stm"] for it in insts])
zh_topk = []
for i, it in enumerate(insts):
    facets = facets_of(it["schema"])
    K = zh.cls([body_text(f) for f in facets])
    zh_topk.append((K @ Qz[i]).argsort(descending=True).tolist())
record("bge-large-zh body", zh_topk)
del zh.m; zh = None; torch.cuda.empty_cache()

# ---- bge-reranker-v2-m3 (cross-encoder) ----
rtok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
rmodel = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", dtype=torch.bfloat16).cuda().eval()
@torch.no_grad()
def rerank(query, docs):
    pairs = [[query, d] for d in docs]
    enc = rtok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    return rmodel(**enc).logits.view(-1).float().cpu()
rr_topk = []
for it in insts:
    facets = facets_of(it["schema"])
    s = rerank(it["stm"], [body_text(f) for f in facets])
    rr_topk.append(s.argsort(descending=True).tolist())
record("bge-reranker-v2-m3", rr_topk)
del rmodel; torch.cuda.empty_cache()

# ---- Qwen3-8B hidden states (mean-pool last hidden, body key) ----
qtokz = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
qmodel = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map="auto", output_hidden_states=True).eval()
@torch.no_grad()
def qemb(texts, bs=8):
    out = []
    for i in range(0, len(texts), bs):
        enc = qtokz(texts[i:i+bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to(qmodel.device)
        h = qmodel(**enc).hidden_states[-1]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (h*mask).sum(1)/mask.sum(1)
        out.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(out)
Qq = qemb([it["stm"] for it in insts])
qw_topk = []
for i, it in enumerate(insts):
    facets = facets_of(it["schema"])
    K = qemb([body_text(f) for f in facets])
    qw_topk.append((K @ Qq[i]).argsort(descending=True).tolist())
record("qwen3-8b hidden body", qw_topk)

print("\n=== summary (sorted by R@1) ===")
for name, v in sorted(results.items(), key=lambda kv: -(kv[1]["R@1"])):
    print(f"{name:26s} R@1={v['R@1']:.3f} R@3={v['R@3']:.3f} proj_adh={v['proj_adh_top1']}")
json.dump(results, open(f"{BASE}/route_sweep.json","w"), ensure_ascii=False, indent=2)
print("wrote route_sweep.json")
