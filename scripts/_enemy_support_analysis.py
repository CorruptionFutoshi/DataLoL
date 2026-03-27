import pandas as pd
import yaml
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / 'config/settings.yaml', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)
_all_members = [m['game_name'] for m in _cfg['members']]

def main():
    member = sys.argv[1] if len(sys.argv) > 1 else _all_members[0]

    df = pd.read_csv('data/processed/player_stats.csv')
    target = df[(df['summonerName'] == member) & (df['role'] == 'BOTTOM')].copy()
    print(f'{member} BOT 試合数: {len(target)}')
    print()

    results = []
    for _, row in target.iterrows():
        match_id = row['matchId']
        team_id = row['teamId']
        win = row['win']
        my_champ = row['championName']

        enemy_sup = df[(df['matchId'] == match_id) & (df['role'] == 'UTILITY') & (df['teamId'] != team_id)]
        if len(enemy_sup) == 1:
            results.append({
                'matchId': match_id,
                'my_champ': my_champ,
                'enemy_sup': enemy_sup.iloc[0]['championName'],
                'win': win,
                'my_kills': row['kills'],
                'my_deaths': row['deaths'],
                'my_assists': row['assists'],
                'my_gold': row['goldEarned'],
                'my_damage': row['totalDamageDealtToChampions'],
            })

    res_df = pd.DataFrame(results)
    print(f'敵サポート特定成功: {len(res_df)} 試合')
    print()

    tl = pd.read_csv('data/processed/timeline_frames.csv')
    tl_s = tl[tl['summonerName'] == member]

    sup_stats = res_df.groupby('enemy_sup').agg(
        games=('win', 'count'),
        wins=('win', 'sum'),
        avg_k=('my_kills', 'mean'),
        avg_d=('my_deaths', 'mean'),
        avg_a=('my_assists', 'mean'),
        avg_gold=('my_gold', 'mean'),
        avg_dmg=('my_damage', 'mean'),
    ).reset_index()
    sup_stats['losses'] = sup_stats['games'] - sup_stats['wins']
    sup_stats['wr'] = (sup_stats['wins'] / sup_stats['games'] * 100).round(1)

    early_gd = {}
    for _, row in res_df.iterrows():
        fr = tl_s[(tl_s['matchId'] == row['matchId']) & (tl_s['timestampMin'] == 10)]
        if len(fr) == 1:
            gd = fr.iloc[0].get('goldDiffVsOpponent', None)
            if gd is not None and not pd.isna(gd):
                sc = row['enemy_sup']
                early_gd.setdefault(sc, []).append(gd)
    avg_gd10 = {k: round(sum(v) / len(v)) for k, v in early_gd.items() if len(v) >= 2}
    sup_stats['gd10'] = sup_stats['enemy_sup'].map(avg_gd10)

    filt = sup_stats[sup_stats['games'] >= 2].sort_values('wr')

    # --- 苦手サポート ---
    print('=' * 70)
    print(f'{member} が苦手な敵サポート (2試合以上、勝率低い順)')
    print('=' * 70)
    worst = filt[filt['wr'] <= 40.0]
    if len(worst) == 0:
        worst = filt.head(15)

    header = f"  {'敵サポート':<16s} {'戦績':^12s} {'勝率':>6s} {'平均KDA':^14s} {'GD@10':>7s}"
    print(header)
    print('-' * 70)
    for _, r in worst.iterrows():
        gd_str = f"{r['gd10']:+.0f}" if pd.notna(r.get('gd10')) else "   -"
        kda = f"{r['avg_k']:.1f}/{r['avg_d']:.1f}/{r['avg_a']:.1f}"
        rec = f"{r['wins']:.0f}W {r['losses']:.0f}L ({r['games']:.0f}G)"
        print(f"  {r['enemy_sup']:<16s} {rec:<12s} {r['wr']:5.1f}%  {kda:<14s} {gd_str:>7s}")
    print()

    # --- 3試合以上で特に深刻 ---
    print('=' * 70)
    print('特に注意 (3試合以上 & 勝率33%以下)')
    print('=' * 70)
    severe = filt[(filt['games'] >= 3) & (filt['wr'] <= 33.4)]
    if len(severe) == 0:
        print('  該当なし')
    else:
        for _, r in severe.iterrows():
            gd_str = f"{r['gd10']:+.0f}" if pd.notna(r.get('gd10')) else "   -"
            kda = f"{r['avg_k']:.1f}/{r['avg_d']:.1f}/{r['avg_a']:.1f}"
            rec = f"{r['wins']:.0f}W {r['losses']:.0f}L ({r['games']:.0f}G)"
            print(f"  {r['enemy_sup']:<16s} {rec:<12s} {r['wr']:5.1f}%  {kda:<14s} {gd_str:>7s}")
    print()

    # --- 自チャンプ vs 敵サポートの組み合わせ ---
    print('=' * 70)
    print('苦手マッチアップ詳細 (自チャンプ vs 敵サポート、2試合以上)')
    print('=' * 70)
    matchup = res_df.groupby(['my_champ', 'enemy_sup']).agg(
        games=('win', 'count'),
        wins=('win', 'sum'),
    ).reset_index()
    matchup['losses'] = matchup['games'] - matchup['wins']
    matchup['wr'] = (matchup['wins'] / matchup['games'] * 100).round(1)
    matchup_worst = matchup[(matchup['games'] >= 2) & (matchup['wr'] <= 33.4)].sort_values('wr')
    if len(matchup_worst) == 0:
        matchup_worst = matchup[matchup['games'] >= 2].sort_values('wr').head(10)

    for _, r in matchup_worst.iterrows():
        print(f"  {r['my_champ']:<14s} vs {r['enemy_sup']:<14s}  {r['wins']:.0f}W {r['losses']:.0f}L ({r['games']:.0f}G)  勝率 {r['wr']:5.1f}%")
    print()

    # --- 得意サポート ---
    print('=' * 70)
    print(f'{member} が得意な敵サポート (3試合以上、勝率高い順 Top10)')
    print('=' * 70)
    good = filt[filt['games'] >= 3].sort_values('wr', ascending=False).head(10)
    for _, r in good.iterrows():
        gd_str = f"{r['gd10']:+.0f}" if pd.notna(r.get('gd10')) else "   -"
        kda = f"{r['avg_k']:.1f}/{r['avg_d']:.1f}/{r['avg_a']:.1f}"
        rec = f"{r['wins']:.0f}W {r['losses']:.0f}L ({r['games']:.0f}G)"
        print(f"  {r['enemy_sup']:<16s} {rec:<12s} {r['wr']:5.1f}%  {kda:<14s} {gd_str:>7s}")
    print()

    # --- サポートタイプ別集計 ---
    engage_sups = ['Leona', 'Nautilus', 'Thresh', 'Blitzcrank', 'Alistar', 'Rakan', 'Rell', 'Pyke', 'Maokai', 'Amumu', 'Braum']
    enchanter_sups = ['Lulu', 'Janna', 'Nami', 'Soraka', 'Yuumi', 'Sona', 'Karma', 'Seraphine', 'Renata', 'Milio', 'Lux', 'Morgana']
    mage_sups = ['Brand', 'Zyra', 'Xerath', 'Velkoz', 'Swain', 'Senna', 'Hwei', 'Annie', 'Shaco']

    def classify(champ):
        if champ in engage_sups:
            return 'エンゲージ'
        elif champ in enchanter_sups:
            return 'エンチャンター'
        elif champ in mage_sups:
            return 'メイジ/ダメージ'
        else:
            return 'その他'

    res_df['sup_type'] = res_df['enemy_sup'].apply(classify)
    type_stats = res_df.groupby('sup_type').agg(
        games=('win', 'count'),
        wins=('win', 'sum'),
    ).reset_index()
    type_stats['losses'] = type_stats['games'] - type_stats['wins']
    type_stats['wr'] = (type_stats['wins'] / type_stats['games'] * 100).round(1)
    type_stats = type_stats.sort_values('wr')

    print('=' * 70)
    print('敵サポートタイプ別 勝率')
    print('=' * 70)
    for _, r in type_stats.iterrows():
        rec = f"{r['wins']:.0f}W {r['losses']:.0f}L ({r['games']:.0f}G)"
        print(f"  {r['sup_type']:<16s} {rec:<16s} 勝率 {r['wr']:5.1f}%")
    print()

if __name__ == '__main__':
    main()
