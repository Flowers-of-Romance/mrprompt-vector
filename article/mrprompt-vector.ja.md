# MRPrompt の cue を外付けにする

## 出発点

先の検証（[mrprompt-repro](https://github.com/Flowers-of-Romance/mrprompt-repro)）で、MRPrompt の「cue-addressable な facet 想起」は in-context では支持されないと示した。手がかりキー（cue_phrases）を消しても壊しても出力は動かず、注意は本体に集まりキーには向かなかった。理由は単純で、facet 本体がプロンプトに全部入っているため、モデルは本体の内容で必要な facet を選べてしまい、短いキーは迂回されるからである。アドレスが意味を持つのは、指す先が他の経路で得られないときに限る。

上記が正しいならば、facet 本体をプロンプトから外し、cue を唯一の取り出し経路に置けば、cue に照合する facet が想起される筈である。facet を外部メモリにキー付きで格納し、対話とキーの照合で一致した本体だけを注入する。いわゆる retrieval（RAG）の構成にする。本稿はこれを mrprompt-repro と同一の100インスタンス・同一の Qwen3-8B・同一の採点で実装し、二つを測る。

1. cue は外付けにすると宛先として機能するか（引けたか）。
2. その retrieve-then-generate は、全 facet をコンテキストに入れる方式（in-context、再現で実際に効いていた対抗馬）に最終タスクで勝てるか。

問1は retrieval の精度、問2はタスク品質である。両者は別物で、問2のバーは高い。この規模ではキャラあたり facet は約7面でそもそもコンテキストに収まる。収まる以上、retrieve した数件を渡すのは全 facet を渡すよりノイズの多い縮約になりうる。

## 方法

- データ：mrprompt-repro の instances_faithful.jsonl（100インスタンス、各キャラ7–8 facet、cued facet と STM 付き）をそのまま使用。
- 検索：`BAAI/bge-m3`（多言語・CLSプーリング、bf16/iGPU）で STM と各 facet の鍵を埋め込み、コサイン最近傍を引く。鍵の厚さを3段で比較する。
  - `cue_only`：cue_phrases のみ（論文の手がかりキーに最も忠実）
  - `cue_situ`：cue_phrases ＋ situation
  - `body`：situation ＋ emotional_state ＋ behavior_pattern ＋ thinking_pattern（facet 本体）
- 生成：引いた facet 本体（＋core_traits）を論文の Magic-If（Fig.19）に載せ、Qwen3-8B（thinking-OFF、max_new_tokens=1024、temperature 0.7／top_p 0.8）。すべて mrprompt-repro と同一設定で、思考は OFF に統一（CoT 主効果と混ざらないため）。
- 採点：応答が正解（cued）facet にどれだけ即しているかの適合度（1–10）を、mrprompt-repro と同じ ADH ルーブリック（GPT-4.1-mini、temp 0）で採点。
- ベースライン（同一インスタンス、mrprompt-repro から流用）：`allctx`＝全 facet を入れる方式（＝再現の mrprompt、適合度 7.43）、`base`＝facet なし散文（7.23）。

## 結果1：cue は外付けにすると因果的になるが、弱い索引である

STM を query に、各 facet の鍵で最近傍を引き、引けた1枚が cued facet かを top-1 精度（R@1）で測った（n=100。chance@k はランダム検索時に cued が top-k に入る確率で、facet 数 n に対し「k/n」をインスタンス平均した値。ここでは chance@1=0.139）。鍵を壊す wrongkey（cued facet の鍵を隣の facet のもので上書き、mrprompt-repro と同じ操作）と比較する。

| 鍵 | R@1 real | R@1 wrong | R@3 real | R@3 wrong |
|---|---|---|---|---|
| cue_only | 0.300 | 0.180 | 0.650 | 0.630 |
| cue_situ | 0.330 | 0.220 | 0.700 | 0.530 |
| body | 0.350 | 0.160 | 0.670 | 0.470 |

（chance@1=0.139、chance@3=0.418）

- real > wrong がすべての鍵で出る。in-context では wrongkey が無効だった（mrprompt − wrongkey = +0.03、null）のと対照的に、外付けにすると cue は因果的になる。正しいキーが正しい本体を引き、誤キーが誤った本体を引く。
- ただし top-1 は最良の鍵（body）でも 0.350 止まり。query を直近発話に絞っても改善しない（全 STM が最良）。これは query の選び方ではなく、同一キャラの facet 同士が意味的に近く（7面で chance 0.139）、弁別が本質的に難しいことによる。本体そのものを鍵にしても3割しか top-1 を当てられない。
- 鍵を厚くすると R@1 は上がる（cue_only 0.300 → cue_situ 0.330 → body 0.350）。retrieval-MRPrompt の top-1 上限がこのあたり。

## 結果2：top-k は厚い鍵とセットでのみ「宛先指定」であり続ける

recall@3（cued が top-3 に含まれる率）は 0.65–0.70 で、chance@3=0.418 を明確に上回る。引くのを3件に広げれば、cued は7割方含まれる。ここで k を広げたときの因果差（R@3 の real − wrong）を見ると、鍵の厚さが効く。

| 鍵 | R@3 real − wrong |
|---|---|
| cue_only | +0.02（ほぼ null） |
| cue_situ | +0.17 |
| body | +0.20 |

cue_only を top-3 にすると、cue が正しいかどうかがほとんど効かなくなる（誤鍵でも 0.63 引けてしまう）。k を広げた瞬間、薄い鍵は「cue で宛先指定」ではなく「意味の近い facet を3つ拾っているだけ」に退化する。top-3 を使うなら、検証対象の機構（cue による宛先指定）を保つために鍵を厚くしなければならない。

鍵 × k には交互作用もある。R@1 の最良は body（0.350）だが、R@3 の最良は cue_situ（0.700、body は 0.670）。運用する k によって最適な鍵が入れ替わる。以下のタスク評価では、top-1 は body、top-3 は cue_situ を用いる。

## 結果3：最終タスク — retrieve-then-generate は全 facet 投入に勝てるか

ここまでは「引けたか」である。決着は最終タスクの適合度（正解 facet への 1–10）で測る。同一100インスタンス、思考OFF、出力予算1024で揃えた。まず完璧な routing が許す最大の上積みを見る：oracle（正解 facet を直接注入）と allctx（全 facet）の差。これが 0 に近ければ、正しく引いても全部入れても同じ＝この規模では routing に価値がないことが、retriever の精度を問う前に確定する。

| 条件 | 適合度 |
|---|---|
| base（facet なし散文） | 7.23 |
| allctx（全 facet、in-context） | 7.43 |
| oracle（正解 facet のみ） | 7.88 |
| body_top1（top-1 検索、body 鍵） | 7.46 |
| cuesitu_top3（top-3 検索、cue_situ 鍵） | 7.50 |

- oracle − allctx = +0.45 ±0.14（n=100、約3.3 SEM）。予想に反して 0 ではない。正解 facet だけを入れる方が、全7面を入れるより良い。全 facet 投入では残り6面が distractor として薄く効き、適合を鈍らせている。つまりこの規模でも選別には価値がある。完璧に引ければ +0.45 の余地がある。
- しかし retrieval は allctx を抜けない。body_top1 − allctx = +0.03 ±0.16、cuesitu_top3 − allctx = +0.07 ±0.17、どちらも null。天井（+0.45）の手前で止まる。
- 理由は routing 精度。retrieval の正誤で適合度を分けると二峰になる。

| 条件 | 当たり時 | 外し時 |
|---|---|---|
| body_top1 | 7.74（n=35） | 7.31（n=65） |
| cuesitu_top3 | 7.61（n=70） | 7.23（n=30） |

当たれば oracle 近傍（7.6–7.7）、外せば base 近傍（7.2–7.3）。平均が allctx と並ぶのは、当たり 35–70%・外し 65–30% の加重がちょうど allctx に落ちるからである。律速は facet の価値ではなく routing の精度になる。top-1 で35%、top-3 で70%しか当てられない（同一キャラの facet が意味的に近い）ため、+0.45 の天井を取りに行けない。

最後に、cue が宛先として因果的かを適合度レベルでも確認する。cue_only top-1 とその wrongkey（cued facet の鍵を隣の facet のもので上書き）を比べると、extcue − wrongkey = +0.23 ±0.14。in-context の mrprompt − wrongkey = +0.03（null）と対照的で、routing レベルでも real(0.30) > wrong(0.18)。弱いが、外付けにすると正しい鍵が正しい本体を引く向きに効く。

## 結果4：routing 精度を上げると最終タスクはどこまで動くか

結果3で律速は routing 精度だと分かった。では routing 精度を上げる手段を入れたとき、適合度はどこまで動くか。二系統で測る。一つは埋め込み検索器を取り替えて R@1 の幅を出す。もう一つは facet 本体を読んで選ぶ LLM ルーターである。いずれも各手法の top-1 facet を実際に Qwen3-8B に注入して生成・採点する（実測。各手法×100インスタンス、単サンプル、body_top1 と同一の単一 facet 注入）。アンカーは oracle（正解 facet を直接注入）＝7.88、all-facets＝7.43。

### 埋め込み検索器の幅（実測）

埋め込みベースの11手法を同一100インスタンスで測った。random、bm25（jieba）、bge-m3 dense（鍵 cue／cue_situ／body）、bge-m3 の colbert 近似（トークン単位 max-sim）、hybrid RRF（bm25＋bge-m3 body）、bge-large-zh-v1.5（中国語特化、body 鍵）、difference-vector（facet 本体から同キャラ他 facet の平均を引いた差分）、bge-reranker-v2-m3（cross-encoder）、Qwen3-8B の最終隠れ状態の平均プーリング（body 鍵）。

| 手法 | R@1 | R@3 | 実測適合度 |
|---|---|---|---|
| bge-large-zh body | 0.40 | 0.66 | 7.62 |
| difference-vector | 0.38 | 0.66 | 7.34 |
| bge-m3 colbert 近似 | 0.38 | 0.71 | 7.55 |
| bge-m3 dense body | 0.35 | 0.67 | 7.46 |
| bge-m3 dense cue_situ | 0.33 | 0.70 | 7.45 |
| bge-m3 dense cue | 0.30 | 0.65 | 7.67 |
| hybrid RRF（bm25＋dense） | 0.28 | 0.65 | 7.26 |
| bge-reranker-v2-m3 | 0.27 | 0.59 | 7.40 |
| Qwen3-8B hidden body | 0.27 | 0.63 | 7.34 |
| bm25（jieba） | 0.19 | 0.45 | 7.46 |
| random | 0.14 | 0.42 | — |

R@1 は 0.19–0.40 に収まり、bm25 を除けば 0.27–0.40 に集中する。中国語特化（0.40）、cross-encoder（0.27）、生成モデルの隠れ状態（0.27）を入れても 0.40 を超えない。実測適合度は 7.26–7.67 に散らばるが、all-facets（7.43）と有意に異なる手法はない（最大の bge-large-zh body で +0.19、z=1.1。全手法 |z|<2。random は chance 下限のため生成せず）。結果3の null は、埋め込み手法を換算でなく実測しても動かない。同一キャラ facet の意味的近接は、埋め込みの差し替えでは縮まらない。単サンプルなので手法間の細かい順位はノイズ（SEM≈0.17–0.19）込みで読む。

R@1 からの換算（R@1×7.74＋(1−R@1)×7.31）で済ませず実測したのは、その換算が当たり群の平均を body_top1 の 7.74 で代表させ全手法一定とみなすからである。実測すると当たり群の平均は手法ごとに 7.74–8.00 と動いた。換算は値を 7.39–7.48 に圧縮していたが、実測は 7.26–7.67 に広がり順位も入れ替わる。定性的な結論（all-facets と並ぶ）は両者で一致するが、数値は換算では当てにならない。次の LLM ルーターも同じ理由で実測する。

### LLM ルーターの実測

Claude Opus 4.6（CLI 経由、生成用 Qwen3-8B とは別系統）に facet を選ばせ、選んだ1枚を Qwen3-8B に注入して生成・採点した。router は全 facet から1つを選ぶ。two-stage は cue_situ 検索の top-3 に候補を絞ってから1つを選ぶ。

| 条件 | R@1 | 実測適合度 | all-facets との差 |
|---|---|---|---|
| oracle（正解 facet） | 1.00 | 7.88 | +0.45 ±0.14（z=3.3） |
| llm_twostage（top-3→1つ） | 0.51 | 7.79 | +0.36 ±0.14（z=2.5） |
| llm_router（全 facet→1つ） | 0.57 | 7.62 | +0.19 ±0.15（z=1.3） |
| cuesitu_top3 | — | 7.50 | +0.07（null） |
| body_top1 | 0.35 | 7.46 | +0.03（null） |
| all-facets | — | 7.43 | — |

- two-stage は all-facets を +0.36 ±0.14（z=2.5）上回り、oracle との差は −0.09 ±0.11（null）。+0.45 の天井とほぼ区別がつかない。埋め込み検索（body_top1 − allctx = +0.03、null）では届かなかった上積みを、two-stage はほぼ取り切る。
- router 単独は +0.19 ±0.15（z=1.3）で、方向は同じだが all-facets との差は有意でない。two-stage と router の差は +0.17 ±0.12（null）で、R@1 は router（0.57）が上なのに適合度は two-stage（7.79）が上回る。
- 理由は外し時の下限。当たり/外しで割ると、router は当たり 8.11／外し 6.98、two-stage は当たり 8.26／外し 7.31。上限は両者とも高い（8.1–8.3）。差は外し時で、router は全 facet から選ぶため外すと base（7.23）以下まで落ちる。two-stage は cue_situ の top-3 に絞ってから選ぶので、外しても会話に近い facet にとどまり 7.31 で下げ止まる。top-k 検索で下限を守り、LLM で上限を取る二段が、平均を有意に押し上げる。
- なぜ換算で済まさず実測したか。当たり群の母集団は手法依存で、LLM ルーターの当たり群は 8.1–8.3、body_top1 は 7.74 と差がある。R@1 から固定の当たり/外し平均で換算すると router 7.56 ＞ two-stage 7.53 となり、実測の順位（two-stage 7.79 ＞ router 7.62）と逆になる。だから換算で済まさず生成・採点した。

routing 精度を上げる手段は、結果3が示した律速をそのまま動かす。埋め込みの差し替えでは R@1 が 0.40 で頭打ちになり all-facets と並ぶが、facet 本体を読む LLM 二段ルーターは all-facets を有意に上回り、oracle 天井に並ぶ。コストは LLM ルーターが1コール約8秒・約$0.08 で、埋め込み検索より2桁高い。

限界：埋め込み手法・LLM ルーターとも単サンプルの実測で、手法間の細かい順位はノイズ込み（SEM≈0.17–0.19）。LLM 採点（GPT-4.1-mini）の判定器バイアスは全条件に共通で残る。

## 結果5：天井の分解 — 選別の価値か、文脈長か

結果3の +0.45（oracle − allctx）には交絡がある。残り6面を外す操作は、distractor を消すと同時に入力を短くする。oracle と allctx の2条件だけでは、上積みが選別（distractor 除去）によるものか、単に文脈が短いことによるものか分離できない。そこで長さ・facet 数・正解 facet の位置をそろえ、distractor の性質だけを変える統制を加える。描画の枠は固定し、並ぶ facet の中身だけを差し替えた。同一100インスタンス、各セル5サンプルの平均で測った。

| 条件 | facet | 入力token | 適合度 |
|---|---|---|---|
| oracle_c（正解1面のみ） | 1 | 1160 | 7.66 |
| oracle_dup（正解面を反復、distractor なし） | 7 | 1924 | 7.66 |
| allctx_c（同キャラ全面、近い distractor） | 7 | 1883 | 7.39 |
| far_c（正解＋他キャラの面、遠い distractor） | 7 | 1889 | 7.33 |

oracle_dup は正解 facet を反復して allctx と同じ長さに伸ばした、distractor を含まない長さ統制。far_c は正解 facet を元の位置に残し、他の枠を他キャラの facet で置き換えた、同じ長さ・数・位置で distractor の意味的近さだけを変えた条件。

- 天井は再現する。oracle_c − allctx_c = +0.27 ±0.09（z=3.1）。
- 長さの寄与はゼロ。oracle_dup − oracle_c = +0.00 ±0.06（z=0.0）。distractor を入れずに allctx と同じ長さまで伸ばしても適合度は変わらない。
- 天井のほぼ全量が distractor による。allctx_c − oracle_dup = −0.27 ±0.08（z=3.4）。同じ長さで competing facet を加えると、天井ぶんだけ下がる。
- 近いか遠いかは差を生まない。far_c − allctx_c = −0.06 ±0.09（n.s.）。distractor が同キャラの近接 facet でも他キャラの遠い facet でも適合度は変わらない。問題は competing facet が在るか否かであり、意味的な近さではない。

分解すると、天井 +0.27 = 長さ +0.00 + distractor +0.27。上積みは入力が短いことの副作用ではなく、distractor を除いたこと（選別）そのものによる。

注：本節は同じ oracle − allctx を5サンプル平均で測り直した。信頼できる天井は +0.27 で、結果3・4で単サンプルから出した +0.45 より小さい（単サンプルの oracle が高めの1ドローを含むため）。方向と有意性は一致する。本節以外の +0.45 は単サンプル推定として読まれたい。

## まとめ

1. in-context（mrprompt-repro）：cue は不活性。本体が文脈に全部あるので迂回される。
2. 外付け（本実験）：cue は因果的になる（extcue − wrongkey = +0.23、routing でも real > wrong）。さらに、正解 facet だけを入れる oracle は全 facet 投入を +0.45 上回る。選別そのものには価値がある。コンテキストに収まるからといって全部入れるのが最善とは限らない。
3. だが現状の retrieval はその価値を実現できない。同一キャラの facet は意味的に近く、cue でも本体でも top-1 35%・top-3 70%が限界。当たれば oracle 近傍、外せば base 近傍に割れ、平均は all-facets と並ぶ（null）。律速は選別の価値ではなく routing の精度である。

4. routing 精度は上げられる。埋め込みの差し替えでは R@1 が 0.40 で頭打ちだが、facet 本体を読む LLM 二段ルーター（cue_situ の top-3 に絞って1つ選ぶ）は all-facets を +0.36 ±0.14 上回り（z=2.5）、oracle 天井との差は −0.09（null）。選別の価値（+0.45）は、ルーターを替えれば実際に取りに行ける。

落とし所はこうなる。埋め込み検索を top-1 で足すだけなら all-facets と並ぶだけで、routing も当てにくい。だが選別の価値（+0.45）は実在し、取りに行く経路は二つある。(a) routing を LLM 二段ルーターにする。本実験では cue_situ の top-3 に絞って LLM に1つ選ばせると、all-facets を有意に上回り oracle 天井に並んだ。埋め込みの top-1 では届かない。(b) facet がコンテキストに収まらない規模。そこでは all-facets が選択肢から外れ、多少粗くても retrieval が必須になる。コストは LLM 二段が1コール約8秒・約$0.08 で、収まる規模なら all-facets ＋ CoT が依然として安い既定値である。

限界：天井（oracle − allctx）は文脈長と交絡しうるが、結果5の長さ統制（distractor を含まず allctx と同じ長さの oracle_dup）が oracle と区別できない（+0.00）ため、天井は文脈長では説明されない。天井の大きさは単サンプル（+0.45）と5サンプル（+0.27）で差があり、本文の単サンプル値はやや過大である。採点は GPT-4.1-mini の adherence で判定器バイアスは残る。単一モデル（Qwen3-8B）・100インスタンス・中国語キャラの予備的結果である。

## 予測：件数が膨大なとき

本実験は1キャラ約7面で、コンテキストに収まる規模である。facet ストアが万件規模（収まらない）になると、結論の構造が変わると予測する。以下はデータの外挿であって測定ではない。

- all-facets が選択肢から消える。全部入れが不可能になり、「retrieval は all-facets と並ぶだけ」という結果3の null の比較そのものが成立しなくなる。retrieval は任意でなく必須になる。結果3は、収まる規模の上限近くのスナップショットと読むべきである。
- 全 facet を読む LLM ルーターも不可能になる。万件を読ませて1つ選ぶ方式は使えず、生き残るのは二段（検索で top-k に絞って LLM が選ぶ）だけになる。本実験で two-stage を推した結論は、規模が上がるほど唯一の選択肢として残る方向に向かう。ただしその品質は一段目の recall で頭打ちになる。
- routing は桁違いに難しくなる。chance@1 は 1/7＝0.14 から 1/万＝0.0001 に落ちる。7面ですら R@1 が 0.40 で頭打ちだったのは意味的に近い候補が混むからで、万件では近傍はさらに密になり、R@1 は下がると見込まれる。同一話題が時間をまたいで蓄積した密クラスタの弁別が、本実験の同一キャラ facet と同じ壁になる。recency と重複の圧縮が新たな軸として重要になる。
- 設計の主役が「いくつ引いて何件注入するか（k）」に移る。収まる／収まらないの二択ではなく、top-k の recall（正解が k 件に入るか）と precision（余分な distractor）のトレードオフが中心になる。結果2（top-3 で recall 0.70、ただし余分2件は distractor）の構図が、そのまま設計ノブになる。
- 結果5（適合度の低下は distractor 由来で、文脈長そのものは無害）は、この設計に直接かかわる。precision さえ保てれば、recall を稼ぐために多めに引いて注入しても、長さでは下がらない。敵は長さではなく誤検索である。ただしこの長さ無害は約1900トークン（7面）までの観測で、万件規模で数十〜数百件を注入する超長コンテキストには外挿できない。そこでは lost-in-the-middle のような真の長さ効果が現れうる。

転移するもの／しないものを分けておく。具体的な数値（R@1 の 0.40 頭打ち、天井 +0.27）は7面レジームの産物で、規模が変われば別の値になる。転移するのは構造の方である。鍵の弁別力が律速であること、誤 facet はコストだが長さは（ある範囲まで）無害であること、検索ショートリストを LLM が再選別する二段が埋め込み top-1 を上回ること。これらは万件でも成り立つと予測する。確かめるには、多数の facet を横断する大規模ストアを作り、プール規模を増やしながら R@1 と適合度の劣化曲線、および k のスイープを測る必要がある。

---

データとコードは本リポジトリ（mrprompt-vector）。採点指標 MS-FA／適合度は mrprompt-repro から逐語で借用。
