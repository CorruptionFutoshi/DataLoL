# LoL Flex Rank 分析システム

League of Legends のフレックスランクの試合データを **Riot Games API** から自動収集し、
メンバーの成績・チャンピオン選択・チーム戦略を分析するシステムです。

---

## 目次

- [はじめに必要なもの](#はじめに必要なもの)
- [セットアップ手順](#セットアップ手順)
- [日常の使い方](#日常の使い方)
- [分析 CLI コマンド一覧](#分析-cli-コマンド一覧)
- [スタンドアロン分析スクリプト](#スタンドアロン分析スクリプト)
- [Jupyter Notebook で分析する](#jupyter-notebook-で分析する)
- [Cursor Agent に質問する](#cursor-agent-に質問する)
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

- `settings.yaml` の `start_date` 以降の全試合を取得します（デフォルト: 2024-01-01）
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

| スクリプト | 何がわかる |
|---|---|
| `ban_analysis.py` | 敵チャンピオンの BAN 優先度ランキング（勝率 60% + ピック率 40% のスコア）、ロール別脅威度 |
| `analyze_teamfight.py` | 中盤 (14〜25分) の集団戦への寄り意識：KP%、集団戦関与率、KP と勝率の相関、マップポジション分析 |
| `teamfight_analysis.py` | 集団戦イニシエート分析：先手を取った側の勝率、時間帯別傾向、メンバー別イニシエート回数 |
| `lane_early_analysis.py` | レーン別序盤影響度：どのレーンが序盤に勝っているとチーム勝率に最も直結するか |
| `grub_disaster.py` | ヴォイドグラブ周辺の大事故分析：味方 2 デス以上の試合を特定し、死亡者・グラブ獲得・勝率を分析 |
| `benchmark_comparison.py` | 同ランク帯ベンチマーク比較：メンバー vs 同試合の非メンバー（対戦相手＋野良）でKDA・CS・ダメージ等をロール別に比較。パーセンタイル・偏差値・強み課題の自動判定付き。`--member X` で個人詳細 |

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
├── scripts/                        スタンドアロン分析スクリプト
│   ├── ban_analysis.py            敵チャンピオン BAN 優先度
│   ├── analyze_teamfight.py       中盤の集団戦への寄り分析
│   ├── teamfight_analysis.py      集団戦イニシエート分析
│   ├── lane_early_analysis.py     レーン別序盤影響度分析
│   └── grub_disaster.py           ヴォイドグラブ大事故分析
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
├── .cursor/skills/                Cursor Agent 用スキル定義
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
