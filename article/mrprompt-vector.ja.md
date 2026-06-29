# MRPrompt の cue を外付けにする

## 出発点

先の検証（mrprompt-repro）で、MRPrompt の「cue-addressable な facet 想起」は in-context では支持されないと示した。手がかりキー（cue_phrases）を消しても壊しても出力は動かず、注意は本体に集まりキーには向かなかった。理由は単純で、facet 本体がプロンプトに全部入っているため、モデルは本体の内容で必要な facet を選べてしまい、短いキーは迂回されるからである。アドレスが意味を持つのは、指す先が他の経路で得られないときに限る。

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

STM を query に、各 facet の鍵で最近傍を引き、引けた1枚が cued facet かを top-1 精度で測った（n=100、chance@1=0.139＝約7面）。鍵を壊す wrongkey（cued facet の鍵を隣の facet のもので上書き、mrprompt-repro と同じ操作）と比較する。

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

## まとめ

1. in-context（mrprompt-repro）：cue は不活性。本体が文脈に全部あるので迂回される。
2. 外付け（本実験）：cue は因果的になる（extcue − wrongkey = +0.23、routing でも real > wrong）。さらに、正解 facet だけを入れる oracle は全 facet 投入を +0.45 上回る。選別そのものには価値がある。コンテキストに収まるからといって全部入れるのが最善とは限らない。
3. だが現状の retrieval はその価値を実現できない。同一キャラの facet は意味的に近く、cue でも本体でも top-1 35%・top-3 70%が限界。当たれば oracle 近傍、外せば base 近傍に割れ、平均は all-facets と並ぶ（null）。律速は選別の価値ではなく routing の精度である。

落とし所は二段になる。facet がコンテキストに収まる規模では、retrieval を入れても all-facets 投入と並ぶだけ（しかも routing は当てにくい）なので、素直に全 facet ＋ CoT でよい。retrieval が効いてくるのは、(a) routing 精度を上げられる場合（近接した facet を弁別できる鍵・埋め込み）か、(b) facet がコンテキストに収まらない規模で、そこでは all-facets が選択肢から外れ、多少粗くても retrieval が必須になる。本実験の +0.45 の天井は、その場合に取りに行く価値が確かに存在することを示している。

限界：allctx（全7面）は body_top1／cuesitu_top3 より入力文脈が長く、品質差に長さの非対称が混じりうる。ただし oracle（1面）は allctx（7面）より短い文脈で +0.45 上回っており、長さは短い条件に不利な方向に働く。天井の存在は文脈長では説明されない。採点は GPT-4.1-mini の adherence で判定器バイアスは残る。単一モデル（Qwen3-8B）・100インスタンス・中国語キャラの予備的結果である。

---

データとコードは本リポジトリ（mrprompt-vector）。採点指標 MS-FA／適合度は mrprompt-repro から逐語で借用。
