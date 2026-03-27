"""JG champion type analysis for a specific player."""
import pandas as pd
import yaml
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / 'config/settings.yaml', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)
_all_members = [m['game_name'] for m in _cfg['members']]
MEMBER = sys.argv[1] if len(sys.argv) > 1 else _all_members[0]

CHAMPION_TYPES = {
    "ファイター/ブルーザー": [
        "Warwick", "Vi", "XinZhao", "Xin Zhao", "LeeSin", "Lee Sin",
        "JarvanIV", "Jarvan IV", "Hecarim",
        "Olaf", "Trundle", "Volibear", "MonkeyKing", "Wukong",
        "RekSai", "Rek'Sai", "Udyr",
        "Skarner", "Nasus", "Jax", "Belveth", "Bel'Veth", "Briar", "Viego",
        "Nocturne", "Shyvana", "Gwen", "Mordekaiser", "Ambessa",
        "Urgot", "Yorick",
    ],
    "タンク": [
        "Amumu", "Sejuani", "Zac", "Rammus", "Nunu", "Nunu & Willump",
        "Maokai", "Poppy", "Gragas", "Cho'Gath", "DrMundo", "Dr. Mundo",
        "Shen",
    ],
    "アサシン": [
        "Khazix", "Kha'Zix", "Rengar", "Kayn", "Talon", "Evelynn", "Elise",
        "Ekko", "Shaco", "Qiyana", "Diana", "Naafiri",
        "MasterYi", "Master Yi", "Yone", "Fizz",
    ],
    "メイジ/AP キャリー": [
        "Karthus", "Brand", "Lillia", "FiddleSticks", "Fiddlesticks",
        "Nidalee", "Taliyah", "Zyra", "Morgana", "Ahri", "AurelionSol",
        "Sylas", "Aurora", "Ivern", "Teemo", "Zaahen",
    ],
    "ADC/レンジキャリー": [
        "Graves", "Kindred", "Jinx", "Twitch", "Vayne",
        "Smolder",
    ],
}

def classify(champ):
    """Classify a champion, allowing multiple types."""
    types = []
    for t, champs in CHAMPION_TYPES.items():
        if champ in champs:
            types.append(t)
    return types if types else ["その他"]


df = pd.read_csv(ROOT / "data" / "processed" / "player_stats.csv")
jg = df[(df["summonerName"] == MEMBER) & (df["role"] == "JUNGLE")].copy()

print("=" * 70)
print(f"  {MEMBER} — JG チャンピオン型別分析")
print(f"  総JG試合数: {len(jg)}  |  JG勝率: {jg['win'].mean():.1%}")
print("=" * 70)

# -- Per-champion stats --
print("\n" + "=" * 70)
print("  PART 1: チャンピオン別 成績一覧 (2試合以上)")
print("=" * 70)

champ_stats = jg.groupby("championName").agg(
    games=("win", "count"),
    wins=("win", "sum"),
    wr=("win", "mean"),
    avg_kills=("kills", "mean"),
    avg_deaths=("deaths", "mean"),
    avg_assists=("assists", "mean"),
    avg_kda=("kda", "mean"),
    avg_dmg=("totalDamageDealtToChampions", "mean"),
    avg_cs=("cs", "mean"),
    avg_gold=("goldEarned", "mean"),
    avg_vision=("visionScore", "mean"),
).sort_values("games", ascending=False)

for _, row in champ_stats.iterrows():
    if row["games"] < 2:
        continue
    champ = row.name
    types = classify(champ)
    type_str = " / ".join(types)
    wr_bar = "█" * int(row["wr"] * 10) + "░" * (10 - int(row["wr"] * 10))
    print(
        f"  {champ:<16} [{type_str}]"
        f"\n    {int(row['games']):>3}試合  {wr_bar} {row['wr']:.0%} "
        f"({int(row['wins'])}W)  "
        f"KDA {row['avg_kills']:.1f}/{row['avg_deaths']:.1f}/{row['avg_assists']:.1f} "
        f"({row['avg_kda']:.2f})  "
        f"DMG {row['avg_dmg']:.0f}  CS {row['avg_cs']:.0f}  Vision {row['avg_vision']:.0f}"
    )

# -- Type classification --
print("\n" + "=" * 70)
print("  PART 2: 型別 集計")
print("=" * 70)

rows_by_type = {}
for _, row in jg.iterrows():
    types = classify(row["championName"])
    for t in types:
        if t not in rows_by_type:
            rows_by_type[t] = []
        rows_by_type[t].append(row)

type_summary = []
for t, rws in rows_by_type.items():
    tdf = pd.DataFrame(rws)
    champs_used = tdf["championName"].nunique()
    type_summary.append({
        "type": t,
        "games": len(tdf),
        "wins": int(tdf["win"].sum()),
        "wr": tdf["win"].mean(),
        "avg_kills": tdf["kills"].mean(),
        "avg_deaths": tdf["deaths"].mean(),
        "avg_assists": tdf["assists"].mean(),
        "avg_kda": tdf["kda"].mean(),
        "avg_dmg": tdf["totalDamageDealtToChampions"].mean(),
        "avg_cs": tdf["cs"].mean(),
        "avg_gold": tdf["goldEarned"].mean(),
        "avg_vision": tdf["visionScore"].mean(),
        "champs": champs_used,
        "champ_list": ", ".join(sorted(tdf["championName"].unique())),
    })

type_summary.sort(key=lambda x: -x["games"])

for s in type_summary:
    wr_bar = "█" * int(s["wr"] * 10) + "░" * (10 - int(s["wr"] * 10))
    print(
        f"\n  【{s['type']}】 ({s['champs']}チャンピオン使用)"
        f"\n    {s['games']:>3}試合  {wr_bar} {s['wr']:.1%} ({s['wins']}W)"
        f"\n    KDA {s['avg_kills']:.1f}/{s['avg_deaths']:.1f}/{s['avg_assists']:.1f} "
        f"({s['avg_kda']:.2f})  "
        f"DMG {s['avg_dmg']:.0f}  CS {s['avg_cs']:.0f}  Vision {s['avg_vision']:.0f}"
        f"\n    使用チャンピオン: {s['champ_list']}"
    )

# -- Type ranking --
print("\n" + "=" * 70)
print("  PART 3: 型別 ランキング (得意順)")
print("=" * 70)

min_games = 10
qualified = [s for s in type_summary if s["games"] >= min_games]
qualified.sort(key=lambda x: -x["wr"])

overall_wr = jg["win"].mean()
overall_kda = jg["kda"].mean()
overall_dmg = jg["totalDamageDealtToChampions"].mean()

print(f"\n  ※ {min_games}試合以上の型のみ  |  全体JG勝率: {overall_wr:.1%}")
print(f"  全体平均 KDA: {overall_kda:.2f}  DMG: {overall_dmg:.0f}\n")

for rank, s in enumerate(qualified, 1):
    wr_diff = s["wr"] - overall_wr
    kda_diff = s["avg_kda"] - overall_kda
    dmg_diff = s["avg_dmg"] - overall_dmg
    wr_arrow = "↑" if wr_diff > 0 else "↓" if wr_diff < 0 else "→"
    kda_arrow = "↑" if kda_diff > 0 else "↓" if kda_diff < 0 else "→"
    dmg_arrow = "↑" if dmg_diff > 0 else "↓" if dmg_diff < 0 else "→"

    if wr_diff > 0.05:
        verdict = "★ 得意"
    elif wr_diff > -0.03:
        verdict = "○ 普通"
    else:
        verdict = "▼ 苦手"

    print(
        f"  {rank}. {s['type']:<20} {verdict}"
        f"\n     勝率 {s['wr']:.1%} ({wr_arrow}{abs(wr_diff):.1%})  "
        f"KDA {s['avg_kda']:.2f} ({kda_arrow}{abs(kda_diff):.2f})  "
        f"DMG {s['avg_dmg']:.0f} ({dmg_arrow}{abs(dmg_diff):.0f})"
        f"\n     {s['games']}試合  使用: {s['champ_list']}"
    )

# -- Benchmark comparison (if available) --
print("\n" + "=" * 70)
print("  PART 4: ベンチマーク比較 (エメラルド帯 JG)")
print("=" * 70)

try:
    bench = pd.read_csv(r"D:\データLoL\data\processed\benchmark_stats.csv")
    bench_jg = bench[bench["role"] == "JUNGLE"]
    if len(bench_jg) > 0:
        bench_avg_kda = bench_jg["kda"].mean()
        bench_avg_dmg = bench_jg["totalDamageDealtToChampions"].mean()
        bench_avg_cs = bench_jg["cs"].mean()
        bench_avg_vision = bench_jg["visionScore"].mean()
        bench_avg_kills = bench_jg["kills"].mean()
        bench_avg_deaths = bench_jg["deaths"].mean()
        bench_avg_assists = bench_jg["assists"].mean()

        print(f"\n  エメラルド帯 JG 平均 ({len(bench_jg)}試合)")
        print(f"    KDA {bench_avg_kills:.1f}/{bench_avg_deaths:.1f}/{bench_avg_assists:.1f} ({bench_avg_kda:.2f})")
        print(f"    DMG {bench_avg_dmg:.0f}  CS {bench_avg_cs:.0f}  Vision {bench_avg_vision:.0f}")
        print()

        for s in qualified:
            kda_vs = s["avg_kda"] - bench_avg_kda
            dmg_vs = s["avg_dmg"] - bench_avg_dmg
            cs_vs = s["avg_cs"] - bench_avg_cs
            vision_vs = s["avg_vision"] - bench_avg_vision

            kda_emoji = "↑" if kda_vs > 0 else "↓"
            dmg_emoji = "↑" if dmg_vs > 0 else "↓"

            print(
                f"  {s['type']:<20} vs エメラルド帯:"
                f"  KDA {kda_emoji}{abs(kda_vs):+.2f}  "
                f"DMG {dmg_emoji}{abs(dmg_vs):+.0f}  "
                f"CS {cs_vs:+.0f}  Vision {vision_vs:+.0f}"
            )
    else:
        print("  ベンチマークデータにJGデータなし")
except FileNotFoundError:
    print("  ベンチマークデータが見つかりません")

# -- Best champions per type --
print("\n" + "=" * 70)
print("  PART 5: 型別 ベストチャンピオン (3試合以上)")
print("=" * 70)

for s in type_summary:
    tdf = pd.DataFrame(rows_by_type[s["type"]])
    champ_in_type = tdf.groupby("championName").agg(
        games=("win", "count"),
        wr=("win", "mean"),
        avg_kda=("kda", "mean"),
        avg_dmg=("totalDamageDealtToChampions", "mean"),
    )
    champ_in_type = champ_in_type[champ_in_type["games"] >= 3].sort_values("wr", ascending=False)
    if len(champ_in_type) == 0:
        continue
    print(f"\n  【{s['type']}】")
    for _, cr in champ_in_type.iterrows():
        wr_bar = "█" * int(cr["wr"] * 10) + "░" * (10 - int(cr["wr"] * 10))
        print(
            f"    {cr.name:<16} {int(cr['games']):>3}試合  {wr_bar} {cr['wr']:.0%}  "
            f"KDA {cr['avg_kda']:.2f}  DMG {cr['avg_dmg']:.0f}"
        )

print("\n" + "=" * 70)
print("  分析完了")
print("=" * 70)
