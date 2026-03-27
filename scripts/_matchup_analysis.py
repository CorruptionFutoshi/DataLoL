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
    
    target = df[df['summonerName'] == member].copy()
    print(f'{member} 総試合数: {len(target)}')
    print(f'ロール分布:')
    print(target['role'].value_counts().to_string())
    print()

    results = []
    for _, row in target.iterrows():
        match_id = row['matchId']
        role = row['role']
        team_id = row['teamId']
        win = row['win']
        my_champ = row['championName']
        
        opponent = df[(df['matchId'] == match_id) & (df['role'] == role) & (df['teamId'] != team_id)]
        if len(opponent) == 1:
            opp_champ = opponent.iloc[0]['championName']
            opp_kills = opponent.iloc[0]['kills']
            opp_deaths = opponent.iloc[0]['deaths']
            opp_assists = opponent.iloc[0]['assists']
            results.append({
                'matchId': match_id,
                'role': role,
                'my_champ': my_champ,
                'opponent_champ': opp_champ,
                'win': win,
                'my_kills': row['kills'],
                'my_deaths': row['deaths'],
                'my_assists': row['assists'],
                'my_gold': row['goldEarned'],
                'my_damage': row['totalDamageDealtToChampions'],
            })

    res_df = pd.DataFrame(results)
    print(f'対面特定成功: {len(res_df)} 試合')
    print()

    # --- Timeline data for early gold diff ---
    try:
        tl = pd.read_csv('data/processed/timeline_frames.csv')
        tl_target = tl[(tl['summonerName'] == member)]
        has_timeline = True
    except Exception:
        has_timeline = False

    opp_stats = res_df.groupby('opponent_champ').agg(
        games=('win', 'count'),
        wins=('win', 'sum'),
        avg_kills=('my_kills', 'mean'),
        avg_deaths=('my_deaths', 'mean'),
        avg_assists=('my_assists', 'mean'),
    ).reset_index()
    opp_stats['losses'] = opp_stats['games'] - opp_stats['wins']
    opp_stats['winrate'] = (opp_stats['wins'] / opp_stats['games'] * 100).round(1)
    opp_stats['avg_kills'] = opp_stats['avg_kills'].round(1)
    opp_stats['avg_deaths'] = opp_stats['avg_deaths'].round(1)
    opp_stats['avg_assists'] = opp_stats['avg_assists'].round(1)

    # Early gold diff vs each opponent champion (10min)
    if has_timeline:
        early_gd = {}
        for _, row in res_df.iterrows():
            mid = row['matchId']
            frame = tl_target[(tl_target['matchId'] == mid) & (tl_target['timestampMin'] == 10)]
            if len(frame) == 1:
                gd = frame.iloc[0].get('goldDiffVsOpponent', None)
                if gd is not None and not pd.isna(gd):
                    opp_c = row['opponent_champ']
                    if opp_c not in early_gd:
                        early_gd[opp_c] = []
                    early_gd[opp_c].append(gd)
        
        avg_gd10 = {k: round(sum(v)/len(v)) for k, v in early_gd.items() if len(v) >= 2}
        opp_stats['avg_gd10'] = opp_stats['opponent_champ'].map(avg_gd10)
    else:
        opp_stats['avg_gd10'] = None

    # 2試合以上、勝率低い順
    filtered = opp_stats[opp_stats['games'] >= 2].sort_values('winrate')

    print('=' * 70)
    print(f'{member} 苦手な対面チャンピオン (2試合以上、勝率低い順)')
    print('=' * 70)
    worst = filtered[filtered['winrate'] <= 40.0]
    if len(worst) == 0:
        worst = filtered.head(15)
    
    print(f'{"対面チャンプ":^16s} {"戦績":^10s} {"勝率":>6s} {"平均KDA":^14s} {"GD@10":>7s}')
    print('-' * 70)
    for _, r in worst.iterrows():
        gd_str = f"{r['avg_gd10']:+.0f}" if pd.notna(r.get('avg_gd10')) else "  -"
        kda_str = f"{r['avg_kills']}/{r['avg_deaths']}/{r['avg_assists']}"
        print(f"  {r['opponent_champ']:<16s} {r['wins']:.0f}W {r['losses']:.0f}L ({r['games']:.0f}G) {r['winrate']:5.1f}%  {kda_str:^14s} {gd_str:>7s}")
    
    print()

    # 3試合以上で特に深刻なもの
    print('=' * 70)
    print('特に注意すべき対面 (3試合以上 & 勝率33%以下)')
    print('=' * 70)
    severe = filtered[(filtered['games'] >= 3) & (filtered['winrate'] <= 33.4)]
    if len(severe) == 0:
        print('  該当なし')
    else:
        for _, r in severe.iterrows():
            gd_str = f"{r['avg_gd10']:+.0f}" if pd.notna(r.get('avg_gd10')) else "  -"
            kda_str = f"{r['avg_kills']}/{r['avg_deaths']}/{r['avg_assists']}"
            print(f"  {r['opponent_champ']:<16s} {r['wins']:.0f}W {r['losses']:.0f}L ({r['games']:.0f}G) {r['winrate']:5.1f}%  {kda_str:^14s} {gd_str:>7s}")
    print()

    # ロール別
    print('=' * 70)
    print('ロール別 苦手対面 (2試合以上 & 勝率40%以下)')
    print('=' * 70)
    for role in target['role'].value_counts().index:
        role_df = res_df[res_df['role'] == role]
        if len(role_df) == 0:
            continue
        role_opp = role_df.groupby('opponent_champ').agg(
            games=('win', 'count'),
            wins=('win', 'sum'),
            avg_kills=('my_kills', 'mean'),
            avg_deaths=('my_deaths', 'mean'),
            avg_assists=('my_assists', 'mean'),
        ).reset_index()
        role_opp['losses'] = role_opp['games'] - role_opp['wins']
        role_opp['winrate'] = (role_opp['wins'] / role_opp['games'] * 100).round(1)
        role_opp['avg_kills'] = role_opp['avg_kills'].round(1)
        role_opp['avg_deaths'] = role_opp['avg_deaths'].round(1)
        role_opp['avg_assists'] = role_opp['avg_assists'].round(1)

        if has_timeline:
            role_matchids = set(role_df['matchId'])
            role_gd = {}
            for _, row in role_df.iterrows():
                mid = row['matchId']
                frame = tl_target[(tl_target['matchId'] == mid) & (tl_target['timestampMin'] == 10)]
                if len(frame) == 1:
                    gd = frame.iloc[0].get('goldDiffVsOpponent', None)
                    if gd is not None and not pd.isna(gd):
                        opp_c = row['opponent_champ']
                        if opp_c not in role_gd:
                            role_gd[opp_c] = []
                        role_gd[opp_c].append(gd)
            role_avg_gd = {k: round(sum(v)/len(v)) for k, v in role_gd.items() if len(v) >= 2}
            role_opp['avg_gd10'] = role_opp['opponent_champ'].map(role_avg_gd)
        else:
            role_opp['avg_gd10'] = None

        role_worst = role_opp[(role_opp['games'] >= 2) & (role_opp['winrate'] <= 40.0)].sort_values('winrate')
        if len(role_worst) > 0:
            print(f'\n  [{role}] ({len(role_df)}試合)')
            for _, r in role_worst.iterrows():
                gd_str = f"{r['avg_gd10']:+.0f}" if pd.notna(r.get('avg_gd10')) else "  -"
                kda_str = f"{r['avg_kills']}/{r['avg_deaths']}/{r['avg_assists']}"
                print(f"    {r['opponent_champ']:<16s} {r['wins']:.0f}W {r['losses']:.0f}L ({r['games']:.0f}G) {r['winrate']:5.1f}%  {kda_str:^14s} {gd_str:>7s}")

    print()

    # 自分のチャンピオンと対面の組み合わせで最も勝率が低いもの
    print('=' * 70)
    print('苦手マッチアップ詳細 (自チャンプ vs 対面、2試合以上)')
    print('=' * 70)
    matchup = res_df.groupby(['my_champ', 'opponent_champ']).agg(
        games=('win', 'count'),
        wins=('win', 'sum')
    ).reset_index()
    matchup['losses'] = matchup['games'] - matchup['wins']
    matchup['winrate'] = (matchup['wins'] / matchup['games'] * 100).round(1)
    matchup_worst = matchup[(matchup['games'] >= 2) & (matchup['winrate'] <= 40.0)].sort_values('winrate')
    if len(matchup_worst) == 0:
        matchup_worst = matchup[matchup['games'] >= 2].sort_values('winrate').head(10)
    
    for _, r in matchup_worst.iterrows():
        print(f"  {r['my_champ']:<14s} vs {r['opponent_champ']:<14s}  {r['wins']:.0f}W {r['losses']:.0f}L ({r['games']:.0f}G)  勝率 {r['winrate']:5.1f}%")

    print()

if __name__ == '__main__':
    main()
