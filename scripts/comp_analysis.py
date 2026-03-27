"""Analyze win rates by team composition archetype (Poke / Engage / Counter-Engage)."""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "settings.yaml"

# ── Champion → Archetype classification ──
# Each champion gets (engage_score, poke_score, counter_engage_score).
# 2 = strong fit, 1 = moderate fit, 0 = no fit.
CHAMPION_SCORES = {
    # ===== ENGAGE (hard initiation) =====
    # Strong engage (2)
    'Malphite': (2,0,0), 'Leona': (2,0,0), 'Nautilus': (2,0,0),
    'Amumu': (2,0,0), 'Diana': (2,0,0), 'JarvanIV': (2,0,0),
    'Vi': (2,0,0), 'Sejuani': (2,0,0), 'Alistar': (2,0,0),
    'Rell': (2,0,0), 'Zac': (2,0,0), 'Rakan': (2,0,0),
    'Ornn': (2,0,0), 'Sion': (2,0,0), 'Galio': (2,0,0),
    'Kled': (2,0,0), 'Hecarim': (2,0,0), 'Camille': (2,0,0),
    'Annie': (2,0,0), 'Lissandra': (2,0,0), 'Malzahar': (2,0,0),
    'Skarner': (2,0,0), 'XinZhao': (2,0,0),
    # Moderate engage (1)
    'Aatrox': (1,0,0), 'Sett': (1,0,0), 'Renekton': (1,0,0),
    'Thresh': (1,0,1), 'Blitzcrank': (1,0,0), 'Pyke': (1,0,0),
    'Gnar': (1,0,0), 'Kennen': (1,0,0), 'Shen': (1,0,0),
    'Nocturne': (1,0,0), 'Briar': (1,0,0), 'MonkeyKing': (1,0,0),
    'Neeko': (1,1,0), 'Qiyana': (1,0,0), 'Sylas': (1,0,0),
    'TwistedFate': (1,0,0), 'Samira': (1,0,0), 'Nilah': (1,0,0),
    'RekSai': (1,0,0), 'Gragas': (1,0,0), 'Ekko': (1,0,0),
    'Swain': (1,0,0), 'Mordekaiser': (1,0,0), 'Darius': (1,0,0),
    'Pantheon': (1,0,0), 'Volibear': (1,0,0), 'Irelia': (1,0,0),
    'Vex': (1,0,0), 'Illaoi': (1,0,0), 'Warwick': (1,0,0),
    'Olaf': (1,0,0), 'Trundle': (1,0,0), 'Udyr': (1,0,0),
    'Rammus': (1,0,0), 'Nunu': (1,0,0), 'Fizz': (1,0,0),
    'Akali': (1,0,0), 'Katarina': (1,0,0), 'Talon': (1,0,0),
    'Zed': (1,0,0), 'Naafiri': (1,0,0), 'Yone': (1,0,0),
    'Yasuo': (1,0,0), 'Riven': (1,0,0), 'Jax': (1,0,0),
    'Fiora': (1,0,0), 'Gwen': (1,0,0), 'Viego': (1,0,0),
    'Belveth': (1,0,0), 'LeeSin': (1,0,0), 'Elise': (1,0,0),
    'KSante': (1,0,0), 'Ambessa': (1,0,0), 'Rengar': (1,0,0),
    'Khazix': (1,0,0), 'Shaco': (1,0,0), 'MasterYi': (1,0,0),
    'Tryndamere': (1,0,0), 'Kayn': (1,0,0),

    # ===== POKE (long-range, siege, whittle) =====
    # Strong poke (2)
    'Xerath': (0,2,0), 'Ziggs': (0,2,0), 'Velkoz': (0,2,0),
    'Jayce': (0,2,0), 'Nidalee': (0,2,0), 'Hwei': (0,2,0),
    'Corki': (0,2,0), 'Zoe': (0,2,0), 'Caitlyn': (0,2,0),
    'Varus': (0,2,0), 'KogMaw': (0,2,0), 'Smolder': (0,2,0),
    # Moderate poke (1)
    'Lux': (0,1,1), 'Viktor': (0,1,0), 'Ahri': (0,1,0),
    'Syndra': (0,1,0), 'Jhin': (0,1,0), 'MissFortune': (0,1,0),
    'Ashe': (0,1,1), 'Sivir': (0,1,0), 'Jinx': (0,1,0),
    'Brand': (0,1,0), 'Karma': (0,1,1), 'Zyra': (0,1,0),
    'Senna': (0,1,0), 'Heimerdinger': (0,1,0), 'Gangplank': (0,1,0),
    'Rumble': (0,1,0), 'Azir': (0,1,0), 'Orianna': (0,1,0),
    'Teemo': (0,1,0), 'Aphelios': (0,1,0), 'Kaisa': (0,1,0),
    'Tristana': (0,1,0), 'Xayah': (0,1,0), 'Draven': (0,1,0),
    'Ezreal': (0,2,0), 'Lucian': (0,1,0), 'Cassiopeia': (0,1,1),
    'AurelionSol': (0,1,0), 'Leblanc': (0,1,0), 'Ryze': (0,1,0),
    'Graves': (0,1,0), 'Kindred': (0,1,0), 'Twitch': (0,1,0),
    'Vayne': (0,1,0), 'Kalista': (0,1,0), 'Akshan': (0,1,0),
    'Zeri': (0,1,0), 'Karthus': (0,1,0), 'Kassadin': (0,1,0),
    'Vladimir': (0,1,0), 'Kayle': (0,1,0), 'Nasus': (0,1,0),
    'DrMundo': (0,0,0), 'Yorick': (0,0,0), 'Singed': (0,0,0),
    'Urgot': (0,1,0), 'Quinn': (0,1,0),
    'Yunara': (0,1,0), 'Mel': (0,1,0), 'Zaahen': (0,1,0),

    # ===== COUNTER-ENGAGE / DISENGAGE / PROTECT =====
    # Strong counter-engage (2)
    'Janna': (0,0,2), 'Lulu': (0,0,2), 'Braum': (0,0,2),
    'Milio': (0,0,2), 'Yuumi': (0,0,2), 'Soraka': (0,0,2),
    'Ivern': (0,0,2), 'Taric': (0,0,2), 'Anivia': (0,0,2),
    # Moderate counter-engage (1)
    'Nami': (0,0,1), 'Renata': (0,0,1), 'Sona': (0,0,1),
    'Morgana': (0,0,1), 'Zilean': (0,0,1), 'Poppy': (1,0,1),
    'Seraphine': (0,1,1), 'Lillia': (0,0,1), 'Bard': (0,0,1),
    'TahmKench': (0,0,1), 'Maokai': (1,0,1),
    'Taliyah': (0,1,1), 'Aurora': (0,0,1),

    # ===== NEUTRAL / MIXED =====
    'FiddleSticks': (1,0,0), 'Shyvana': (0,0,0),
}

DEFAULT_SCORE = (0, 0, 0)


def classify_team(champions: list[str]) -> tuple[str, dict]:
    """Classify a team's composition based on champion picks.
    Returns (archetype_name, score_dict).
    """
    total = {'engage': 0, 'poke': 0, 'counter_engage': 0}
    for champ in champions:
        e, p, c = CHAMPION_SCORES.get(champ, DEFAULT_SCORE)
        total['engage'] += e
        total['poke'] += p
        total['counter_engage'] += c

    max_score = max(total.values())
    if max_score == 0:
        return 'mixed', total

    dominant = max(total, key=total.get)
    second = sorted(total.values(), reverse=True)[1]

    if max_score - second <= 1 and max_score <= 3:
        return 'mixed', total

    label_map = {
        'engage': 'エンゲージ構成',
        'poke': 'ポーク構成',
        'counter_engage': 'カウンターエンゲージ構成',
    }
    return label_map[dominant], total


def main():
    df = pd.read_csv(DATA / 'player_stats.csv')

    with open(CONFIG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    members = [m['game_name'] for m in cfg.get('members', [])]

    our_rows = df[df['summonerName'].isin(members)]
    our_matches = our_rows[['matchId', 'teamId']].drop_duplicates()
    match_team = our_matches.groupby('matchId')['teamId'].first().reset_index()
    match_team.columns = ['matchId', 'ourTeamId']

    df2 = df.merge(match_team, on='matchId', how='inner')
    ally = df2[df2['teamId'] == df2['ourTeamId']].copy()
    enemy = df2[df2['teamId'] != df2['ourTeamId']].copy()

    total_matches = match_team['matchId'].nunique()
    print(f'=== チーム構成アーキタイプ分析: {total_matches} 試合 ===')
    print()

    # ── Classify our team and enemy team per match ──
    ally_comps = ally.groupby('matchId')['championName'].apply(list).reset_index()
    ally_comps.columns = ['matchId', 'ally_champs']
    ally_comps['our_comp'], ally_comps['our_scores'] = zip(
        *ally_comps['ally_champs'].apply(classify_team)
    )

    enemy_comps = enemy.groupby('matchId')['championName'].apply(list).reset_index()
    enemy_comps.columns = ['matchId', 'enemy_champs']
    enemy_comps['enemy_comp'], enemy_comps['enemy_scores'] = zip(
        *enemy_comps['enemy_champs'].apply(classify_team)
    )

    win_by_match = ally.groupby('matchId')['win'].first().reset_index()
    win_by_match.columns = ['matchId', 'win']

    comp_df = ally_comps.merge(enemy_comps, on='matchId').merge(win_by_match, on='matchId')

    # ── PART 1: 自チームの構成別勝率 ──
    print('=' * 60)
    print('PART 1: 自チームの構成別勝率')
    print('=' * 60)
    print()

    our_comp_stats = comp_df.groupby('our_comp').agg(
        games=('win', 'count'),
        wins=('win', 'sum')
    ).reset_index()
    our_comp_stats['win_rate'] = our_comp_stats['wins'] / our_comp_stats['games'] * 100
    our_comp_stats = our_comp_stats.sort_values('win_rate', ascending=False)

    overall_wr = comp_df['win'].mean() * 100

    for _, row in our_comp_stats.iterrows():
        diff = row['win_rate'] - overall_wr
        bar = '█' * int(row['win_rate'] / 2)
        sign = '+' if diff > 0 else ''
        print(f"  {row['our_comp']:<20s}  {row['games']:>3.0f}試合  "
              f"{row['wins']:.0f}勝  勝率 {row['win_rate']:5.1f}%  "
              f"(全体比 {sign}{diff:+.1f}pp)  {bar}")
    print()
    print(f"  全体勝率: {overall_wr:.1f}%")
    print()

    # ── PART 2: 敵チームの構成別 被勝率 ──
    print('=' * 60)
    print('PART 2: 敵チームの構成に対する勝率')
    print('=' * 60)
    print()

    enemy_comp_stats = comp_df.groupby('enemy_comp').agg(
        games=('win', 'count'),
        wins=('win', 'sum')
    ).reset_index()
    enemy_comp_stats['win_rate'] = enemy_comp_stats['wins'] / enemy_comp_stats['games'] * 100
    enemy_comp_stats = enemy_comp_stats.sort_values('win_rate', ascending=False)

    for _, row in enemy_comp_stats.iterrows():
        diff = row['win_rate'] - overall_wr
        bar = '█' * int(row['win_rate'] / 2)
        sign = '+' if diff > 0 else ''
        print(f"  vs {row['enemy_comp']:<20s}  {row['games']:>3.0f}試合  "
              f"{row['wins']:.0f}勝  勝率 {row['win_rate']:5.1f}%  "
              f"(全体比 {sign}{diff:+.1f}pp)  {bar}")
    print()

    # ── PART 3: マッチアップ (自構成 vs 敵構成) ──
    print('=' * 60)
    print('PART 3: 構成マッチアップ (自チーム構成 vs 敵チーム構成)')
    print('=' * 60)
    print()

    matchup = comp_df.groupby(['our_comp', 'enemy_comp']).agg(
        games=('win', 'count'),
        wins=('win', 'sum')
    ).reset_index()
    matchup['win_rate'] = matchup['wins'] / matchup['games'] * 100
    matchup = matchup.sort_values('games', ascending=False)

    min_games = 3
    significant = matchup[matchup['games'] >= min_games].copy()

    print(f"  (最低{min_games}試合以上のマッチアップのみ表示)")
    print()
    print(f"  {'自チーム構成':<22s} {'vs':<4s} {'敵チーム構成':<22s} {'試合':>4s} {'勝利':>4s} {'勝率':>6s}")
    print(f"  {'─'*22} {'──':<4s} {'─'*22} {'──':>4s} {'──':>4s} {'───':>6s}")

    for _, row in significant.iterrows():
        wr = row['win_rate']
        if wr >= 60:
            indicator = '◎'
        elif wr >= 50:
            indicator = '○'
        elif wr >= 40:
            indicator = '△'
        else:
            indicator = '×'
        print(f"  {row['our_comp']:<22s} vs   {row['enemy_comp']:<22s} "
              f"{row['games']:>4.0f}  {row['wins']:>4.0f}  {wr:>5.1f}% {indicator}")
    print()

    # ── PART 4: ピボットテーブル (勝率マトリクス) ──
    print('=' * 60)
    print('PART 4: 構成マッチアップ 勝率マトリクス')
    print('=' * 60)
    print()

    archetype_order = ['エンゲージ構成', 'ポーク構成', 'カウンターエンゲージ構成', 'mixed']

    pivot_wr = matchup.pivot_table(
        index='our_comp', columns='enemy_comp',
        values='win_rate', aggfunc='first'
    )
    pivot_n = matchup.pivot_table(
        index='our_comp', columns='enemy_comp',
        values='games', aggfunc='first'
    )

    existing_order = [a for a in archetype_order if a in pivot_wr.index]
    existing_cols = [a for a in archetype_order if a in pivot_wr.columns]

    pivot_wr = pivot_wr.reindex(index=existing_order, columns=existing_cols)
    pivot_n = pivot_n.reindex(index=existing_order, columns=existing_cols)

    col_labels = [c[:8] for c in existing_cols]
    header = f"  {'自＼敵':<22s} " + " ".join(f"{c:>12s}" for c in col_labels)
    print(header)
    print(f"  {'─'*22} " + " ".join(f"{'─'*12}" for _ in col_labels))

    for idx in existing_order:
        if idx not in pivot_wr.index:
            continue
        row_label = idx
        cells = []
        for col in existing_cols:
            wr = pivot_wr.loc[idx, col] if not pd.isna(pivot_wr.loc[idx, col]) else None
            n = pivot_n.loc[idx, col] if not pd.isna(pivot_n.loc[idx, col]) else None
            if wr is not None and n is not None:
                cells.append(f"{wr:5.1f}%({int(n):>2d})")
            else:
                cells.append(f"{'--':>10s}")
        print(f"  {row_label:<22s} " + " ".join(f"{c:>12s}" for c in cells))
    print()

    # ── PART 5: 構成の分布 ──
    print('=' * 60)
    print('PART 5: 構成タイプの分布')
    print('=' * 60)
    print()

    our_dist = comp_df['our_comp'].value_counts()
    enemy_dist = comp_df['enemy_comp'].value_counts()

    print("  自チーム構成分布:")
    for comp, count in our_dist.items():
        pct = count / total_matches * 100
        bar = '█' * int(pct / 2)
        print(f"    {comp:<22s}  {count:>3d}試合 ({pct:5.1f}%)  {bar}")
    print()
    print("  敵チーム構成分布:")
    for comp, count in enemy_dist.items():
        pct = count / total_matches * 100
        bar = '█' * int(pct / 2)
        print(f"    {comp:<22s}  {count:>3d}試合 ({pct:5.1f}%)  {bar}")
    print()

    # ── PART 6: 未分類チャンピオン警告 ──
    all_champs_used = set(df2['championName'].unique())
    classified = set(CHAMPION_SCORES.keys())
    unclassified = all_champs_used - classified
    if unclassified:
        print(f"  ※ 未分類チャンピオン ({len(unclassified)}体): {', '.join(sorted(unclassified))}")
        unclass_count = df2[df2['championName'].isin(unclassified)].shape[0]
        total_rows = df2.shape[0]
        print(f"    → 全ピック中の割合: {unclass_count}/{total_rows} ({unclass_count/total_rows*100:.1f}%)")
        print()

    # ── PART 7: 構成スコア詳細 (平均スコア) ──
    print('=' * 60)
    print('PART 6: 構成スコア詳細 (勝ち試合 vs 負け試合の平均スコア)')
    print('=' * 60)
    print()

    def extract_scores(row):
        s = row['our_scores']
        return pd.Series({
            'engage_score': s['engage'],
            'poke_score': s['poke'],
            'counter_engage_score': s['counter_engage'],
        })

    score_df = comp_df.apply(extract_scores, axis=1)
    score_df['win'] = comp_df['win'].values

    for label, grp in score_df.groupby('win'):
        result = '勝ち' if label else '負け'
        print(f"  {result}試合の平均構成スコア:")
        print(f"    エンゲージスコア:        {grp['engage_score'].mean():.2f}")
        print(f"    ポークスコア:            {grp['poke_score'].mean():.2f}")
        print(f"    カウンターエンゲージスコア: {grp['counter_engage_score'].mean():.2f}")
        print()


if __name__ == '__main__':
    main()
