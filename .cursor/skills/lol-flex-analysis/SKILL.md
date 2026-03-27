---
name: lol-flex-analysis
description: >-
  Analyze League of Legends Flex Rank match data for team members.
  Use when the user asks about win rates, champion performance, synergy,
  early game gold differences, objectives, game tempo, member stats,
  match details, or any LoL Flex Rank analysis question.
---

# LoL Flex Rank Analysis

## Project Structure

```
D:\データLoL\
├── config/settings.yaml        # API key + member list
├── src/                        # Core pipeline (5 files)
│   ├── riot_api.py             # Riot API wrapper (Match/League/Summoner)
│   ├── collect.py              # Data collection (run first)
│   ├── collect_benchmark.py    # Emerald benchmark data collection
│   ├── process.py              # JSON → CSV conversion (run second)
│   └── analyze.py              # CLI analysis tool (10 commands)
├── docs/
│   └── optimal_lane_algorithm.md  # Algorithm explainer (non-technical)
├── scripts/                    # Standalone analysis scripts (27 files)
│   ├── ban_analysis.py         # Enemy champion BAN priority ranking
│   ├── analyze_teamfight.py    # Mid-game teamfight rally / KP analysis
│   ├── lane_early_analysis.py  # Lane-by-lane early game vs win rate
│   ├── grub_disaster.py        # Void Grub fight disaster analysis
│   ├── optimal_lane_advanced.py # Optimal lane assignment (Ridge + Bootstrap CI, CV AUC, extrapolation detection)
│   ├── lane_composite_judgment.py # Multi-signal composite lane judgment (5 signals, consensus rating)
│   ├── lane_reliability_audit.py  # Lane assignment model reliability audit (permutation test, temporal stability)
│   ├── lane_vulnerability.py   # Lane vulnerability: bleed rate, collapse frequency, recovery
│   ├── player_type_analysis.py # Player type profiling (carry/tank/vision etc.)
│   ├── member_role_winrate.py  # Per-member role win rate & displacement analysis
│   ├── teamfight_analysis.py   # Teamfight initiation scoring (multi-signal)
│   ├── last_teamfight_catch.py # Who got caught in the game-losing teamfight
│   ├── midgame_catch_deaths.py # Mid-game isolated catch deaths (non-teamfight)
│   ├── mental_tilt_ranking.py  # Tilt/mental resilience ranking (role-adjusted)
│   ├── vision_analysis.py      # Vision/ward comprehensive analysis + benchmark comparison
│   ├── comp_analysis.py        # Team comp archetype (Poke/Engage/Counter-Engage) win rate
│   ├── carry_count_winrate.py  # Damage carry count vs win rate analysis
│   ├── mid_champion_deep.py    # Mid lane champion-specific deep dive (stats + Holm correction)
│   ├── mid_lane_rigorous.py    # Mid lane rigorous statistical analysis (Welch t, Cohen's d)
│   ├── splitpush_v4.py         # Splitpush episode analysis v4 (tower/kill/death/safe outcomes)
│   ├── benchmark_comparison.py # Compare members vs Emerald-rank benchmark
│   ├── champion_benchmark.py   # Champion+role-specific comparison vs Emerald/Diamond
│   ├── _solo_kill_analysis.py  # [utility] Solo kill (1v1) ranking
│   ├── _win_loss_structure.py  # [utility] Win vs loss structural comparison
│   ├── _top_jungle_proximity.py # [utility] Top-JG proximity analysis
│   ├── _carry_weakside_analysis.py # [utility] Carry vs weak-side resource analysis
│   └── _dramatic_games.py      # [utility] Drama score game finder
├── data/
│   ├── raw/matches/            # Raw match JSON files
│   ├── raw/timelines/          # Raw timeline JSON files
│   ├── raw/benchmark/matches/  # Raw Emerald-rank match JSON files
│   ├── raw/benchmark/timelines/ # Raw Emerald-rank timeline JSON files
│   └── processed/              # Analysis-ready CSVs
│       ├── matches.csv
│       ├── player_stats.csv
│       ├── timeline_frames.csv
│       ├── timeline_events.csv
│       ├── objectives.csv
│       ├── benchmark_stats.csv           # Emerald-rank player stats
│       └── benchmark_timeline_frames.csv # Emerald-rank timeline frames
└── notebooks/                  # Jupyter analysis notebooks (01-07)
```

## Analysis CLI

Run `python src/analyze.py <command>` from the project root. Available commands:

| Command | Description | Example |
|---------|-------------|---------|
| `overview` | Team-wide win rate summary | `python src/analyze.py overview` |
| `champion` | Champion stats per member | `python src/analyze.py champion --member PlayerName` |
| `synergy` | Duo/trio win rates | `python src/analyze.py synergy` |
| `trends` | Weekly and hourly trends | `python src/analyze.py trends` |
| `early` | Gold diff at N minutes vs win rate | `python src/analyze.py early --minute 15` |
| `objectives` | Dragon/Baron impact | `python src/analyze.py objectives` |
| `tempo` | Game duration & snowball rates | `python src/analyze.py tempo` |
| `member` | Deep dive on one player | `python src/analyze.py member PlayerName` |
| `match` | Single match breakdown | `python src/analyze.py match JP1_12345` |
| `teamfight` | Teamfight first-death & positioning | `python src/analyze.py teamfight` |

## Standalone Analysis Scripts

Run `python scripts/<script>.py` from the project root. These provide deeper dives:

| Script | What it does |
|--------|--------------|
| `ban_analysis.py` | Ranks enemy champions by BAN priority (60% win rate + 40% pick rate), shows high-WR threats, frequent picks, and per-role top threats |
| `analyze_teamfight.py` | Mid-game (14-25 min) teamfight rally analysis: KP% per member, role-based KP, teamfight involvement, KP-vs-win-rate correlation, map positioning |
| `lane_early_analysis.py` | Lane-by-lane early game impact: which lane's gold lead correlates most with winning, per-member ahead rates, swing analysis |
| `grub_disaster.py` | Void Grub fight comprehensive analysis: wave clustering (gap>2min), grub pit proximity filter (3000 units), PART1 全体像 + PART2 第一波vs第二波 + PART3 ロール別味方vs敵(=同ランク帯平均)比較 + PART4 メンバー別 + PART5 大事故(2デス+)詳細 |
| `optimal_lane_advanced.py` | Optimal 5-man lane assignment using logistic Ridge regression. Bootstrap 500-iteration 95% CI, CV AUC/accuracy, extrapolation detection, MIN_GAMES=15/MIN_GAMES_SCORE=20. Displacement (玉突き) analysis included. Use `--bootstrap N` to adjust iterations. See `docs/optimal_lane_algorithm.md` for plain-language algorithm explanation |
| `lane_composite_judgment.py` | Multi-signal composite lane evaluation. 5 signals: Ridge regression effect, team WR impact, early laning z-score, player-role fit score, observed pattern WR. Per-member multi-dimensional tables, per-role rankings, consensus-based optimal lineup. Use alongside optimal_lane_advanced.py |
| `lane_reliability_audit.py` | Lane assignment model reliability audit: permutation test (is the model better than random?), explanatory factor comparison (lane vs champion vs GD vs objectives), temporal stability (first half vs second half), sample size power analysis, Bootstrap recommendation stability, observed pattern CI |
| `lane_vulnerability.py` | Lane vulnerability analysis: bleed rate (GD progression at 5/8/10/15 min), collapse frequency (-500G/-1000G), recovery ability (behind@8min→15min trajectory), lane stability composite score. Identifies which lanes can be safely left alone vs need jungler help |
| `player_type_analysis.py` | Player type profiling: classifies each member (carry, tank, vision, utility, etc.) based on KDA, damage, vision, farm stats |
| `member_role_winrate.py` | Per-member role win rate matrix with displacement analysis via OLS regression. Identifies over/under-rated role assignments by comparing simple WR vs independent effects |
| `teamfight_analysis.py` | Teamfight initiation analysis using multi-signal scoring (first kill team, fight location, assist coordination, kill chains, engage champion involvement) |
| `last_teamfight_catch.py` | Analyzes the game-deciding last teamfight in losses: who got caught first (= initiated the losing fight), frequency per member |
| `midgame_catch_deaths.py` | Mid-game (14min+) isolated catch death analysis: identifies deaths outside of teamfights (≤2 nearby kills within 15s), ranks members by catch death frequency, shows map zone and timing patterns |
| `mental_tilt_ranking.py` | Mental resilience ranking (role-adjusted): measures performance collapse when behind at 10min (-300G+). Compares KDA/deaths/KP/damage/vision/CS degradation vs same-role average players. Identifies who tilts hardest under pressure |
| `vision_analysis.py` | Comprehensive vision analysis: team vs enemy ward placement/removal rates, per-member vision contribution, role-level comparison, ward survival analysis (place→destroy time delta), vision score ↔ win rate correlation, Emerald benchmark comparison |
| `comp_analysis.py` | Team composition archetype analysis: classifies games by comp type (Poke/Engage/Counter-Engage) using champion scoring, shows win rate by archetype, archetype matchup matrix, per-member champion pool archetype distribution |
| `carry_count_winrate.py` | Damage carry count vs win rate: classifies champions as carry/non-carry, analyzes how many carries in team correlates with winning. Shows optimal carry count, win rate by carry count, member-level carry/non-carry preference and performance |
| `mid_champion_deep.py` | Mid lane champion-specific deep dive: per-champion stats for mid players (7+ games), comparison vs non-member mid laners (Welch t-test + Cohen's d), personal champion pool comparison, early GD progression per champion, bleed/collapse patterns. All tests Holm-corrected for multiple comparisons |
| `mid_lane_rigorous.py` | Mid lane rigorous statistical analysis: all conclusions backed by Welch t-test, 1-sample t-test, Wilson score CI, Cohen's d, Spearman correlation. Avoids small-sample win rate claims. Focuses on continuous metrics (GD, DPG, KDA) with effect sizes |
| `splitpush_v4.py` | Splitpush episode analysis v4: frame-based episode detection (depth ≥ 0.60, no allies within 3000 units, 25min+), filters out free pushes (3+ enemies dead). Tracks outcomes: tower taken, solo kill, death, safe retreat. Per-member ranking with success rates |
| `benchmark_comparison.py` | Compare members vs Emerald-rank benchmark (separately collected data, NOT in-game opponents). Team overview, role-by-role, per-member percentile dashboard, early game GD benchmark, auto strength/weakness detection. Requires benchmark data (`collect_benchmark.py` + `process.py`). Use `--member X` for deep dive |
| `champion_benchmark.py` | Champion+role-specific comparison vs Emerald/Diamond-rank players. Eliminates champion bias by comparing same-champion-in-same-role performance (e.g. "your Jinx BOT vs Em/Dia Jinx BOT"). Role-matched when possible, falls back to all-role with warning. Shows ⚠少量 for <30 benchmark games. Requires benchmark data. Use `--member X` for deep dive |

## Lane Assignment Scripts — Limitations & Correct Usage

### What these scripts CANNOT do (current data: ~530 matches)

`optimal_lane_advanced.py`, `lane_composite_judgment.py`, `lane_reliability_audit.py` は **「誰をどのロールに置くべきか」というポジティブな最適配置を統計的に有意に導くことはできない**。

理由:
- CV AUC ≈ 0.47 (ランダム以下) — モデルは勝敗を区別できていない
- 正則化パラメータ C=0.01 が選択される → モデル自身が「効果はほぼゼロ」と判断
- ほぼ全てのメンバー×ロールの95%CIがゼロをまたぐ
- 推奨配置の予測勝率CIも広い (例: 52.0% [45.6%, 58.0%])
- 推奨される「最適配置」は多くが未観測パターン（外挿）

### What these scripts CAN do — ネガティブフィルター

95%CIがゼロをまたがない（= 統計的に有意）ケースは **「避けるべき配置」** として信頼できる。
例: PlayerX@ミッド = -2.5pp [-4.2, -1.0] → CIが完全にマイナス → 避けるべき

### How to answer lane assignment questions

ユーザーが「最適なレーン配置は？」と聞いた場合:

1. `python scripts/optimal_lane_advanced.py` と `python scripts/lane_composite_judgment.py` の**両方**を実行（必要に応じて `python scripts/lane_reliability_audit.py` も）
2. 結果を解釈するとき、以下を必ず伝える:
   - **ポジティブな推奨は参考程度** — 統計的に有意ではない
   - **ネガティブフィルター（95%CIがゼロをまたがない悪い配置）は信頼できる**
   - **実測パターン (PART 3) が最も信頼できるデータ** — 実際にプレイされた配置の勝率
   - 複合判断スクリプトの合意度が高い（◎/○）配置は複数の視点で一致しており、やや信頼性が高い
3. 「この配置が最適です」ではなく「この配置は避けた方がいい」「実測ではこの配置が最も勝っている」という伝え方をする

### Data volume needed for positive recommendations

ポジティブな推奨が可能になるための目安:
- 1000試合以上: 主要な配置パターンの効果が有意に検出され始める可能性
- 各メンバー×ロールで50試合以上: Bootstrap CIが十分に狭くなる
- 現在の2倍程度のデータがあれば、一部の効果は有意になる可能性がある

## Answering Analysis Questions

When the user asks an analysis question:

1. **Determine which command best answers it** from the tables above
2. **Run the command** via Shell tool in `D:\データLoL`
3. **Interpret the output** — highlight key insights, not just raw numbers
4. **平均プレイヤーとの比較を重視する** — 分析結果を解釈する際、可能な限り同ランク帯の平均的なプレイヤーとの比較（ベンチマーク）を含める。メンバーの数値が高い/低いだけでなく、「平均と比べてどの程度優れているか・劣っているか」を示すことで、強み・弱みをより客観的に伝える。`benchmark_comparison.py` の結果を併用したり、パーセンタイルや偏差を言及するなど、相対的な評価を常に意識する
5. **Suggest follow-ups** — e.g. "member X has a low early-game gold diff; try running `early --minute 10` to see if the pattern starts even earlier"

### Question → Command Mapping

- "How is everyone doing?" / "成績は？" → `overview`
- "What champions should X play?" / "得意チャンピオンは？" → `champion --member X`
- "Who should duo together?" / "誰と組むと勝てる？" → `synergy`
- "Are we improving?" / "最近の調子は？" → `trends`
- "How important is early gold lead?" / "序盤のゴールド差の影響は？" → `early`
- "Should we prioritize dragon?" / "ドラゴンの重要度は？" → `objectives`
- "Do we win more in short or long games?" / "短期戦と長期戦どっちが得意？" → `tempo`
- "How is X performing?" / "Xの成績は？" → `member X`
- "What happened in match Y?" / "この試合どうだった？" → `match Y`
- "Who dies first in teamfights?" / "集団戦で先に死ぬのは？" → `teamfight`
- "Teamfight positioning?" / "集団戦の立ち位置は？" → `teamfight --member X`
- "What should we ban?" / "何をBANすべき？" → `python scripts/ban_analysis.py`
- "Mid-game rally awareness?" / "中盤の寄りの意識は？" → `python scripts/analyze_teamfight.py`
- "Which lane matters most early?" / "どのレーンの序盤が一番大事？" → `python scripts/lane_early_analysis.py`
- "Grub fights going badly?" / "グラブファイトで事故ってる？" → `python scripts/grub_disaster.py`
- "Optimal lane assignment?" / "誰をどのレーンに置くのが最適？" → `python scripts/optimal_lane_advanced.py` (with CI + CV metrics) + `python scripts/lane_composite_judgment.py` (multi-signal composite)
- "What type of player is X?" / "プレイヤータイプは？" → `python scripts/player_type_analysis.py`
- "Role win rates considering displacement?" / "玉突き込みのロール勝率は？" → `python scripts/member_role_winrate.py`
- "Who initiates teamfights?" / "集団戦のイニシエートは？" → `python scripts/teamfight_analysis.py`
- "Who gets caught in the last fight?" / "最後の集団戦で捕まるのは？" → `python scripts/last_teamfight_catch.py`
- "Splitpush analysis?" / "スプリットプッシュの結果は？" → `python scripts/splitpush_v4.py`
- "How do we compare to average?" / "平均的なプレイヤーと比べてどう？" → `python scripts/benchmark_comparison.py` (エメラルド帯ベンチマークと比較)
- "How does X compare to average?" / "Xは平均と比べてどう？" → `python scripts/benchmark_comparison.py --member X` (エメラルド帯ベンチマークと比較)
- "What percentile are we?" / "パーセンタイルは？" / "同ランク帯との比較" / "ベンチマーク" → `python scripts/benchmark_comparison.py` (エメラルド帯ベンチマークと比較)
- "Champion-adjusted comparison?" / "チャンピオン補正済みの比較" / "エメラルドと比べてどう？" → `python scripts/champion_benchmark.py`
- "How is my Jinx vs Emerald?" / "俺のジンクスはエメラルド平均と比べてどう？" → `python scripts/champion_benchmark.py --member X`
- "Who gets caught mid-game?" / "中盤のキャッチデスは？" → `python scripts/midgame_catch_deaths.py`
- "Who tilts the most?" / "誰が一番メンタル弱い？" / "ティルトしやすいのは？" → `python scripts/mental_tilt_ranking.py`
- "Vision analysis?" / "ビジョン・ワードの分析は？" / "視界がない原因は？" → `python scripts/vision_analysis.py`
- "Team comp win rates?" / "構成の勝率は？" / "ポーク/エンゲージどっちが強い？" → `python scripts/comp_analysis.py`
- "How many carries should we have?" / "キャリー何人が最適？" → `python scripts/carry_count_winrate.py`
- "Mid lane champion details?" / "ミッドのチャンピオン別詳細は？" → `python scripts/mid_champion_deep.py`
- "Rigorous mid lane stats?" / "ミッドの厳密な統計分析" → `python scripts/mid_lane_rigorous.py`
- "Is the lane model reliable?" / "レーン配置モデルの信頼性は？" → `python scripts/lane_reliability_audit.py`
- "Which lane bleeds most?" / "どのレーンが一番崩れやすい？" / "レーンの脆弱性は？" → `python scripts/lane_vulnerability.py`

## For Deeper Custom Analysis

If the CLI doesn't cover the question, read CSVs directly with pandas:

```python
import pandas as pd
df = pd.read_csv('D:\\データLoL\\data\\processed\\player_stats.csv')
```

### Key CSV Schemas

**player_stats.csv**: matchId, summonerName, tagLine, teamId, role, championName, win, kills, deaths, assists, kda, cs, goldEarned, totalDamageDealtToChampions, visionScore, firstBloodKill, firstTowerKill

**timeline_frames.csv**: matchId, timestampMin, summonerName, teamId, role, championName, win, totalGold, cs, goldDiffVsOpponent, csDiffVsOpponent, level, xp

**objectives.csv**: matchId, timestampMin, objectiveType (DRAGON/BARON/RIFT_HERALD/TOWER), teamId, monsterSubType, isFirst

**benchmark_stats.csv**: Same schema as player_stats.csv but from Emerald-rank Flex games. Used by `champion_benchmark.py` for champion-specific comparison.

**benchmark_timeline_frames.csv**: Same schema as timeline_frames.csv but from Emerald-rank games. Enables early-game gold diff comparison per champion.

## Data Pipeline

Collection is time-based: all Flex games since `start_date` in `settings.yaml` (default 2024-01-01) are fetched, with automatic pagination. Subsequent runs only fetch new games since the last collection.

If the user needs fresh data:

```
python src/collect.py          # Fetch new matches from Riot API
python src/collect.py --full   # Re-fetch all matches since start_date
python src/process.py          # Rebuild CSVs from raw JSON
```

### Emerald/Diamond Benchmark Data

For champion-specific comparison vs Emerald+Diamond players. Current data: 1422 matches from 80 players (EMERALD+DIAMOND Flex), 172 champions covered.

```
python src/collect_benchmark.py                          # Default: EMERALD+DIAMOND, 80 players
python src/collect_benchmark.py --players 120            # Sample more players for more coverage
python src/collect_benchmark.py --tier EMERALD DIAMOND PLATINUM  # Add more tiers
python src/process.py                                    # Generates benchmark_stats.csv automatically
```

The benchmark collection fetches ranked Flex players from League-V4 API, downloads their recent matches + timelines, and stores raw JSON in `data/raw/benchmark/`. Running `process.py` after collection produces `benchmark_stats.csv` and `benchmark_timeline_frames.csv`. Re-running `collect_benchmark.py` incrementally adds more data (skips already-collected players/matches).

## Language

The user may speak Japanese or English. Respond in whichever language they use. Analysis output labels are in Japanese.
