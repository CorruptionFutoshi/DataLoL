# LoL Flex Rank 分析システム

League of Legends のフレックスランクの試合データを **Riot Games API** から自動収集し、
メンバーの成績・チャンピオン選択・チーム戦略を分析するシステムです。

> **はじめての方・ヨネ → [ヨネでもわかるセットアップガイド](docs/quickstart.md)** を読んでください。
> 以下は詳しい使い方・全コマンドの一覧です。

---

## 目次

- [はじめに必要なもの](#はじめに必要なもの)
- [セットアップ手順](#セットアップ手順)
- [日常の使い方](#日常の使い方)
- [分析 CLI コマンド一覧](#分析-cli-コマンド一覧)
- [スタンドアロン分析スクリプト](#スタンドアロン分析スクリプト)
- [Jupyter Notebook で分析する](#jupyter-notebook-で分析する)
- [Cursor Agent に質問する](#cursor-agent-に質問する)
- [Cursor 以外の環境で使う](#cursor-以外の環境で使う)
- [フォルダ構成](#フォルダ構成)
- [よくある質問](#よくある質問)

---

## はじめに必要なもの

| 必要なもの | 入手先 |
|---|---|
| Python 3.10 以上 | https://www.python.org/downloads/ |
| Riot API キー | https://developer.riotgames.com/ |
| 分析対象メンバーの Riot ID | LoL クライアントの右上に表示される `名前#タグ` |

> **注意**: Riot Developer Portal の Development API Key は **24 時間で失効**します。
> 毎回プレイ前に再生成するか、[Personal API Key を申請](https://developer.riotgames.com/)してください。

---

## セットアップ手順

### Step 1: パッケージのインストール

```bash
cd "D:\データLoL"
pip install -r requirements.txt
```

### Step 2: API キーの設定

`config/settings.yaml` を開き、`api_key` に取得したキーを貼り付けます。

```yaml
riot_api:
  api_key: "RGAPI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"   # ← ここに貼る
  region: "jp1"
  routing: "asia"
```

### Step 3: メンバーの登録

同じファイルの `members` セクションに、フレックスの仲間を追加します。

```yaml
members:
  - game_name: "サモナー名"    # Riot ID の @ より前の部分
    tag_line: "JP1"            # Riot ID の # より後の部分
```

LoL クライアントでフレンドのプロフィールを開くと `サモナー名#JP1` のような Riot ID が確認できます。

### Step 4: データ取得期間の設定（任意）

同じファイルの `collection` セクションにある `start_date` で、いつ以降の試合を取得するか変更できます。

```yaml
collection:
  start_date: "2025-01-01"   # ← この日付以降の試合を取得（デフォルト: 2025-01-01）
```

たとえば「直近だけでいい」なら `"2025-06-01"` のように変えてください。
古い日付にするほど取得に時間がかかりますが、分析できるデータ量は増えます。

---

## 日常の使い方

試合後にデータを更新して分析するには、以下の **2 コマンド** を実行するだけです。

```bash
python src/collect.py
python src/process.py
```

| コマンド | やること | 所要時間 |
|---|---|---|
| `python src/collect.py` | Riot API から新しい試合データを取得 | 数分（API レート制限あり） |
| `python src/process.py` | 生 JSON を分析用 CSV に変換 | 数秒 |

- `settings.yaml` の `start_date` 以降の全試合を取得します（デフォルト: 2025-01-01）
- 2 回目以降は **新しい試合だけ** を差分取得します（前回取得分はスキップ）
- 全件取り直したい場合: `python src/collect.py --full`
- 取得した生データは `data/raw/` に JSON で永久保存されます

---

## 分析 CLI コマンド一覧

Notebook を開かなくても、コマンドラインで即座に分析結果を確認できます。

```bash
python src/analyze.py <コマンド>
```

### 基本コマンド

| コマンド | 何がわかる | 実行例 |
|---|---|---|
| `overview` | チーム全体の勝率・メンバー別成績 | `python src/analyze.py overview` |
| `champion` | チャンピオン別の勝率・KDA | `python src/analyze.py champion` |
| `synergy` | デュオの組み合わせ別勝率 | `python src/analyze.py synergy` |
| `trends` | 週別・時間帯別の勝率推移 | `python src/analyze.py trends` |

### 深掘りコマンド

| コマンド | 何がわかる | 実行例 |
|---|---|---|
| `early` | 15分時点のゴールド差が勝率にどう影響するか | `python src/analyze.py early` |
| `objectives` | ドラゴン・バロンの獲得と勝率の関係 | `python src/analyze.py objectives` |
| `tempo` | 試合時間帯別勝率・スノーボール率・逆転率 | `python src/analyze.py tempo` |
| `teamfight` | 集団戦の最初の死亡者・ポジション分析 | `python src/analyze.py teamfight` |

### 個別調査コマンド

| コマンド | 何がわかる | 実行例 |
|---|---|---|
| `member <名前>` | 特定メンバーの詳細成績 | `python src/analyze.py member メンバー名` |
| `champion --member <名前>` | 特定メンバーのチャンピオン成績 | `python src/analyze.py champion --member メンバー名` |
| `match <試合ID>` | 特定の試合のスコアボード | `python src/analyze.py match JP1_572102341` |
| `early --minute <分>` | 任意の分時点でのゴールド差分析 | `python src/analyze.py early --minute 10` |

---

## スタンドアロン分析スクリプト

より深い分析を行うための個別スクリプトです。プロジェクトルートから実行してください。

```bash
python scripts/<スクリプト名>.py
```

### メイン分析スクリプト

| スクリプト | 何がわかる |
|---|---|
| `ban_analysis.py` | 敵チャンピオンの BAN 優先度ランキング（勝率 60% + ピック率 40% のスコア）、ロール別脅威度 |
| `analyze_teamfight.py` | 中盤 (14〜25分) の集団戦への寄り意識：KP%、集団戦関与率、KP と勝率の相関、マップポジション分析 |
| `teamfight_analysis.py` | 集団戦イニシエート分析：先手を取った側の勝率、時間帯別傾向、メンバー別イニシエート回数 |
| `last_teamfight_catch.py` | 負け試合の最後の集団戦で最初に捕まった人ランキング |
| `midgame_catch_deaths.py` | 中盤（14分以降）の集団戦外キャッチデス：孤立死の頻度・マップゾーン・タイミング |
| `lane_early_analysis.py` | レーン別序盤影響度：どのレーンが序盤に勝っているとチーム勝率に最も直結するか |
| `grub_disaster.py` | ヴォイドグラブ周辺の大事故分析：味方 2 デス以上の試合を特定し、死亡者・グラブ獲得・勝率を分析 |
| `optimal_lane_advanced.py` | 最適レーン配置分析：Ridge 回帰 + Bootstrap 95%CI + CV AUC + 外挿検知。`--bootstrap N` で回数調整。`docs/optimal_lane_algorithm.md` にアルゴリズム解説 |
| `lane_composite_judgment.py` | レーン配置 複合判断：5つのシグナル（回帰効果・チーム勝率・序盤力・適性・実測パターン）の合意度で評価 |
| `lane_reliability_audit.py` | レーン配置モデルの信頼性監査：順列検定・要因比較・時系列安定性・検出力分析 |
| `lane_vulnerability.py` | レーン脆弱性分析：出血率（GD推移）・崩壊頻度・回復力・安定度スコア。JG介入要否の判断材料 |
| `player_type_analysis.py` | プレイヤータイプ分類：KDA/ダメージ/ビジョン/ファームからキャリー・タンク・ユーティリティ等を判定 |
| `member_role_winrate.py` | メンバー×ロール勝率マトリクス + OLS回帰による玉突き効果の分離 |
| `mental_tilt_ranking.py` | メンタル耐性ランキング：10分時点ビハインド(-300G+)時のパフォーマンス低下をロール補正して測定 |
| `vision_analysis.py` | ビジョン総合分析：ワード設置/除去・視界スコア↔勝率相関・エメラルド帯ベンチマーク比較 |
| `comp_analysis.py` | 構成タイプ別勝率分析：ポーク/エンゲージ/カウンターエンゲージの分類・マッチアップ・チャンプ分布 |
| `carry_count_winrate.py` | ダメージキャリー人数 vs 勝率：最適なキャリー人数、メンバー別のキャリー/非キャリー適性 |
| `mid_champion_deep.py` | ミッドレーンチャンピオン別詳細：対面比較(Welch t + Cohen's d)・GD推移・Holm補正 |
| `mid_lane_rigorous.py` | ミッドレーン厳密統計分析：全結論に検定・効果量・信頼区間付き |
| `splitpush_v4.py` | スプリットプッシュ分析 v4：エピソード検出(depth≥0.60, 味方不在, 25分+)→タワー/キル/デス/安全撤退の結果追跡 |
| `benchmark_comparison.py` | エメラルド帯ベンチマーク比較：チーム全体・ロール別・パーセンタイル・強み弱み自動判定。`--member X` で個人詳細 |
| `champion_benchmark.py` | チャンピオン+ロール別ベンチマーク：同チャンプ同ロール比較でチャンピオンバイアスを排除。`--member X` で個人詳細 |

### ユーティリティスクリプト（`_` プレフィクス）

特定のテーマを掘り下げるユーティリティです。一部は引数でメンバーを指定できます。

| スクリプト | 何がわかる | 引数 |
|---|---|---|
| `_solo_kill_analysis.py` | ソロキル(1v1)ランキング・チャンピオン別ソロキル/デス | `[メンバー名]` |
| `_win_loss_structure.py` | 勝ち試合 vs 負け試合の構造的差異 | なし |
| `_top_jungle_proximity.py` | トップレーンの優位は自力 or JG依存？ キルイベント・GD・CS差で検証 | `[メンバー名]` |
| `_carry_weakside_analysis.py` | キャリー側 vs ウィークサイドのリソース配分と勝率 | なし |
| `_dramatic_games.py` | 逆転度・接戦度のドラマスコアで印象的な試合を検索 | なし |
| `_tilt_deep_analysis.py` | ティルト詳細分析 + 序盤ビハインド率で真のティルト王を判定 | なし |
| `_gold_share_winrate.py` | ゴールド配分率と勝率の関係 | なし |
| `_pick_priority.py` | チャンピオンのピック優先度ランキング（勝率順） | なし |
| `_first_tower_winrate.py` | 最初に破壊/喪失したタワー別の勝率 | なし |
| `_first_tower_benchmark.py` | ファーストタワー勝率のエメラルド帯ベンチマーク比較 | なし |
| `_bench_lane_early.py` | ベンチマーク(エメラルド帯)のレーン別序盤GD vs 勝率分析 | なし |
| `_jg_type_analysis.py` | JGチャンピオンのタイプ別(ファイター/タンク/アサシン等)勝率分析 | `[メンバー名]` |
| `_enemy_support_analysis.py` | 敵サポートチャンピオン別の勝率・相性 | `[メンバー名]` |
| `_matchup_analysis.py` | 対面チャンピオン別のマッチアップ勝率 | `[メンバー名]` |

---

## Jupyter Notebook で分析する

グラフや図表を使ったビジュアルな分析は Notebook で行います。

```bash
jupyter notebook notebooks/
```

| Notebook | 内容 |
|---|---|
| `01_overview` | メンバー別勝率テーブル、勝率推移グラフ、ロール分布 |
| `02_champion` | チャンピオン別勝率、得意チャンピオン TOP5、KDA ヒートマップ |
| `03_synergy` | デュオ勝率マトリクス、ロール組み合わせの勝率 |
| `04_trends` | 週別勝率推移、曜日 × 時間帯ヒートマップ、連勝/連敗ストリーク |
| `05_early_game` | 15 分ゴールド差 vs 勝率（ロジスティック回帰）、ファーストブラッド影響度 |
| `06_objectives` | ドラゴン獲得数 vs 勝率、バロン獲得タイミング、ヘラルド→タワー連鎖 |
| `07_game_tempo` | ゴールドリード推移グラフ、逆転勝利パターン、スノーボール率 |

### Notebook の使い方（Cursor / VS Code）

1. Jupyter 拡張機能をインストール（拡張機能タブで「Jupyter」を検索）
2. `.ipynb` ファイルをダブルクリックで開く
3. 右上の **「カーネルを選択」** で Python を選ぶ
4. **Shift + Enter** でセルを上から順に実行 → セルの直下に結果が表示される

---

## Cursor Agent に質問する

このプロジェクトには Agent Skill が組み込まれているため、
Cursor の Agent モードで自然に質問するだけで分析が実行されます。

### 質問の例

| 聞き方 | Agent がやること |
|---|---|
| 「最近の成績はどう？」 | `overview` + `trends` を実行して解説 |
| 「○○の得意チャンピオンは？」 | `champion --member メンバー名` を実行 |
| 「誰と組むと勝てる？」 | `synergy` を実行してベストデュオを提示 |
| 「序盤のゴールド差って大事？」 | `early` を実行してデータで回答 |
| 「ドラゴン優先すべき？」 | `objectives` を実行して判断材料を提示 |
| 「短期戦と長期戦どっちが得意？」 | `tempo` を実行して解説 |
| 「何を BAN すべき？」 | `scripts/ban_analysis.py` を実行して解説 |
| 「中盤の寄りの意識は？」 | `scripts/analyze_teamfight.py` を実行して解説 |
| 「グラブで事故ってない？」 | `scripts/grub_disaster.py` を実行して解説 |
| 「誰が一番メンタル弱い？」 | `scripts/mental_tilt_ranking.py` を実行して解説 |
| 「ビジョン・ワードの分析は？」 | `scripts/vision_analysis.py` を実行して解説 |
| 「構成の勝率は？」 | `scripts/comp_analysis.py` を実行して解説 |
| 「最適なレーン配置は？」 | `optimal_lane_advanced.py` + `lane_composite_judgment.py` の両方を実行 |
| 「同ランク帯と比べてどう？」 | `scripts/benchmark_comparison.py` を実行して解説 |
| 「スプリットプッシュの結果は？」 | `scripts/splitpush_v4.py` を実行して解説 |
| 「集団戦で先に捕まるのは？」 | `scripts/last_teamfight_catch.py` を実行して解説 |
| 「ソロキルが多いのは誰？」 | `scripts/_solo_kill_analysis.py` を実行して解説 |
| 「一番接戦だった試合は？」 | `scripts/_dramatic_games.py` を実行して解説 |

---

## Cursor 以外の環境で使う

このプロジェクトは Cursor IDE 向けに Agent Skill（`.cursor/skills/`）を同梱していますが、
コア機能は **標準的な Python スクリプト** なので、どの環境でも利用できます。

### 共通の前提

どの環境でも以下は同じです:

1. Python 3.10+ と `pip install -r requirements.txt` で依存パッケージをインストール
2. `config/settings.yaml.example` を `config/settings.yaml` にコピーし、API キーとメンバーを設定
3. `python src/collect.py` → `python src/process.py` でデータ収集・CSV 生成
4. `python scripts/<スクリプト名>.py` で分析実行

### ChatGPT Codex

Codex はリポジトリをクローンしてサンドボックス内で作業します。

- **Agent Skill は読み込まれません**: `.cursor/skills/` は Cursor 固有の仕組みです。代わりに `README.md` と `AGENTS.md`（あれば）がコンテキストとして利用されます
- **設定ファイル**: Codex のサンドボックスでは `config/settings.yaml` が `.gitignore` 済みで存在しないため、セットアップタスクとして「`settings.yaml.example` を `settings.yaml` にコピーして API キーを設定して」と指示してください
- **データ収集**: Riot API キーが必要です。Codex にキーを渡してデータ収集から実行するか、事前にローカルで収集した `data/processed/*.csv` をリポジトリに含める（`.gitignore` から除外する）方法があります
- **分析の実行**: CSV が存在すれば `python scripts/ban_analysis.py` 等はそのまま動きます。「○○を分析して」と聞けば、スクリプト一覧（この README の表）を参照して適切なスクリプトを実行してくれます

### GitHub Copilot (VS Code)

- Agent Skill は読まれません。代わりにリポジトリ内の `README.md` とコードが参照されます
- VS Code のターミナルからスクリプトを手動実行する使い方が中心です
- Copilot Chat に「この README を読んで、○○を分析して」と依頼すれば、適切なコマンドを提案してくれます

### Windsurf / Cline / その他の AI IDE

- 各ツール固有のルールファイル（`rules/` や `.windsurfrules` 等）はありません
- `README.md` のスクリプト一覧と Question → Command マッピングが唯一のガイドです
- 必要に応じて `.cursor/skills/lol-flex-analysis/SKILL.md` の内容をそのツール向けのルールファイルに転記してください

### 手動（AI なし）

AI を使わず手動で分析する場合:

```bash
# 1. データ更新
python src/collect.py
python src/process.py

# 2. 全体成績を確認
python src/analyze.py overview

# 3. 深掘り分析
python scripts/ban_analysis.py
python scripts/benchmark_comparison.py --member メンバー名
python scripts/mental_tilt_ranking.py
# ... 他のスクリプトも同様
```

各スクリプトは標準出力にテキストで結果を表示します。特別な UI や Web サーバーは不要です。

---

## フォルダ構成

```
D:\データLoL\
│
├── config/
│   ├── settings.yaml              API キー・メンバーリスト・リージョン設定
│   └── settings.yaml.example      設定ファイルのテンプレート
│
├── src/                            コアパイプライン
│   ├── riot_api.py                Riot API ラッパー（レート制限・リトライ対応）
│   ├── collect.py                 データ収集（差分更新対応）
│   ├── process.py                 JSON → CSV 変換（5 種類の CSV を生成）
│   └── analyze.py                 分析 CLI ツール（10 コマンド）
│
├── scripts/                        スタンドアロン分析スクリプト（36 ファイル）
│   ├── ban_analysis.py            敵チャンピオン BAN 優先度
│   ├── analyze_teamfight.py       中盤の集団戦への寄り分析
│   ├── teamfight_analysis.py      集団戦イニシエート分析
│   ├── last_teamfight_catch.py    負け試合のラスト集団戦捕まり分析
│   ├── midgame_catch_deaths.py    中盤キャッチデス分析
│   ├── lane_early_analysis.py     レーン別序盤影響度分析
│   ├── grub_disaster.py           ヴォイドグラブ大事故分析
│   ├── optimal_lane_advanced.py   最適レーン配置（Ridge回帰+Bootstrap）
│   ├── lane_composite_judgment.py レーン配置複合判断（5シグナル合意）
│   ├── lane_reliability_audit.py  レーン配置モデル信頼性監査
│   ├── lane_vulnerability.py      レーン脆弱性分析
│   ├── player_type_analysis.py    プレイヤータイプ分類
│   ├── member_role_winrate.py     メンバー×ロール勝率+玉突き分析
│   ├── mental_tilt_ranking.py     メンタル耐性ランキング
│   ├── vision_analysis.py         ビジョン総合分析+ベンチマーク比較
│   ├── comp_analysis.py           構成タイプ別勝率
│   ├── carry_count_winrate.py     キャリー人数 vs 勝率
│   ├── mid_champion_deep.py       ミッドチャンピオン別詳細
│   ├── mid_lane_rigorous.py       ミッドレーン厳密統計
│   ├── splitpush_v4.py            スプリットプッシュ分析
│   ├── benchmark_comparison.py    エメラルド帯ベンチマーク比較
│   ├── champion_benchmark.py      チャンピオン別ベンチマーク比較
│   ├── _solo_kill_analysis.py     [utility] ソロキルランキング
│   ├── _win_loss_structure.py     [utility] 勝ち/負け構造比較
│   ├── _top_jungle_proximity.py   [utility] トップ-JG近接度分析
│   ├── _carry_weakside_analysis.py [utility] キャリー/ウィークサイド分析
│   ├── _dramatic_games.py         [utility] ドラマスコア試合検索
│   ├── _tilt_deep_analysis.py     [utility] ティルト詳細分析
│   ├── _gold_share_winrate.py     [utility] ゴールド配分率 vs 勝率
│   ├── _pick_priority.py          [utility] ピック優先度ランキング
│   ├── _first_tower_winrate.py    [utility] ファーストタワー勝率
│   ├── _first_tower_benchmark.py  [utility] ファーストタワーBM比較
│   ├── _bench_lane_early.py       [utility] BM序盤レーン比較
│   ├── _jg_type_analysis.py       [utility] JGタイプ別分析
│   ├── _enemy_support_analysis.py [utility] 敵サポート相性
│   └── _matchup_analysis.py       [utility] 対面マッチアップ分析
│
├── data/
│   ├── raw/
│   │   ├── matches/               試合詳細の生 JSON（1 試合 = 1 ファイル）
│   │   ├── timelines/             タイムラインの生 JSON（1 分毎のフレームデータ）
│   │   └── collection_state.json  収集進捗の状態ファイル
│   └── processed/
│       ├── matches.csv            試合の基本情報（日時・勝敗・試合時間）
│       ├── player_stats.csv       プレイヤー別の試合成績（KDA, CS, ダメージ等）
│       ├── timeline_frames.csv    1 分毎のゴールド・CS・XP スナップショット
│       ├── timeline_events.csv    試合内イベント（キル、ワード、アイテム購入等）
│       └── objectives.csv         オブジェクト獲得ログ（ドラゴン、バロン、タワー等）
│
├── notebooks/                     Jupyter 分析ノートブック（01〜07）
│
├── .cursor/skills/                Cursor Agent 用スキル定義（他環境では不使用）
│
├── requirements.txt               Python パッケージ一覧
└── README.md                      このファイル
```

---

## よくある質問

### Q: API キーが切れたと言われた

Development API Key は 24 時間で失効します。
[Riot Developer Portal](https://developer.riotgames.com/) でキーを再生成し、
`config/settings.yaml` に貼り直してください。

### Q: データを最初から取り直したい

```bash
python src/collect.py --full
python src/process.py
```

### Q: メンバーを追加したい

`config/settings.yaml` の `members` に追記して、再度 `collect.py` → `process.py` を実行してください。

### Q: Notebook のグラフが文字化けする

Jupyter 拡張機能が入っていない場合は、Cursor の拡張機能タブから「Jupyter」をインストールしてください。
フォント関連の文字化けは自動で日本語フォントを検出する設定にしてあるため、
最初のセル（import のセル）から順に再実行すれば解消します。

### Q: 時刻がおかしい

`process.py` は日本時間 (JST, UTC+9) でタイムスタンプを変換します。
CSV を再生成すれば反映されます: `python src/process.py`
