import pandas as pd
import numpy as np
import yaml
import sys, io
from pathlib import Path
from itertools import combinations

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / 'config/settings.yaml', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)
MEMBERS = [m['game_name'] for m in _cfg['members']]

ps = pd.read_csv('data/processed/player_stats.csv')
tf = pd.read_csv('data/processed/timeline_frames.csv')
ROLE_JP = {'TOP': 'トップ', 'JUNGLE': 'ジャングル', 'MIDDLE': 'ミッド',
           'BOTTOM': 'ボトム', 'UTILITY': 'サポート'}
ROLE_ORDER = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

ps_m = ps[ps['summonerName'].isin(MEMBERS)].copy()

# =====================================================
#  1. メンバー × ロール別 勝率
# =====================================================
print('=' * 75)
print('  メンバー × ロール別 勝率・対面GD')
print('=' * 75)
print()

tf15 = tf[tf['timestampMin'] == 15]

# Role-based baseline win rate (all players in dataset)
role_wr_baseline = ps.groupby('role')['win'].mean() * 100

for member in MEMBERS:
    member_ps = ps_m[ps_m['summonerName'] == member]
    member_tf = tf15[tf15['summonerName'] == member]
    total_games = len(member_ps)
    total_wr = member_ps['win'].mean() * 100

    print(f'  ■ {member}  （全体: {total_games}試合 勝率{total_wr:.1f}%）')
    print()

    header = (f'    {"ロール":>8}  {"試合":>4}  {"勝率":>6}  {"ロール平均WR":>10}  '
              f'{"差":>6}  {"15分GD":>7}  {"KDA":>5}  {"キル":>4}  {"デス":>4}  {"ダメージ":>8}')
    print(header)
    print('    ' + '-' * 85)

    for role in ROLE_ORDER:
        role_ps = member_ps[member_ps['role'] == role]
        if len(role_ps) < 3:
            continue
        games = len(role_ps)
        wr = role_ps['win'].mean() * 100
        baseline = role_wr_baseline.get(role, 50)
        diff = wr - baseline

        role_tf = member_tf[member_tf['role'] == role]
        gd15 = role_tf['goldDiffVsOpponent'].mean() if len(role_tf) > 0 else np.nan

        kda = role_ps['kda'].mean()
        kills = role_ps['kills'].mean()
        deaths = role_ps['deaths'].mean()
        dmg = role_ps['totalDamageDealtToChampions'].mean()

        marker = ''
        if diff > 5:
            marker = ' ★'
        elif diff < -5:
            marker = ' ▼'

        gd_str = f'{gd15:>+7.0f}' if not np.isnan(gd15) else '    N/A'

        print(f'    {ROLE_JP[role]:>8}  {games:>4}  {wr:>5.1f}%  {baseline:>9.1f}%  '
              f'{diff:>+5.1f}%  {gd_str}  {kda:>5.2f}  {kills:>4.1f}  {deaths:>4.1f}  {dmg:>8,.0f}{marker}')

    print()

# =====================================================
#  2. チーム勝率への影響: メンバー×ロール
# =====================================================
print()
print('=' * 75)
print('  チーム勝率への影響: このメンバーがこのロールにいるときの"チーム"勝率')
print('=' * 75)
print()
print('  ※ 個人勝率ではなく、その配置のときのチーム全体の勝率')
print()

# For each match, get match result + who played what role
match_results = ps_m.groupby('matchId').first()[['win']].rename(columns={'win': 'team_win'})
member_role_map = ps_m[['matchId', 'summonerName', 'role']].copy()

# Overall team WR
overall_wr = match_results['team_win'].mean() * 100
total_matches = len(match_results)
print(f'  チーム全体: {total_matches}試合 勝率{overall_wr:.1f}%')
print()

results = []

for member in MEMBERS:
    for role in ROLE_ORDER:
        member_role_matches = member_role_map[
            (member_role_map['summonerName'] == member) &
            (member_role_map['role'] == role)
        ]['matchId'].unique()

        if len(member_role_matches) < 3:
            continue

        wr = match_results.loc[match_results.index.isin(member_role_matches), 'team_win'].mean() * 100

        # WR when member is NOT in this role (but present in game)
        member_other_matches = member_role_map[
            (member_role_map['summonerName'] == member) &
            (member_role_map['role'] != role)
        ]['matchId'].unique()

        wr_other = np.nan
        if len(member_other_matches) >= 3:
            wr_other = match_results.loc[
                match_results.index.isin(member_other_matches), 'team_win'
            ].mean() * 100

        # WR when member is absent
        all_member_matches = member_role_map[
            member_role_map['summonerName'] == member
        ]['matchId'].unique()
        absent_matches = match_results.index[~match_results.index.isin(all_member_matches)]
        wr_absent = np.nan
        if len(absent_matches) >= 3:
            wr_absent = match_results.loc[absent_matches, 'team_win'].mean() * 100

        results.append({
            'member': member,
            'role': role,
            'games': len(member_role_matches),
            'wr': wr,
            'wr_other_role': wr_other,
            'wr_absent': wr_absent,
            'diff_vs_overall': wr - overall_wr,
        })

res_df = pd.DataFrame(results)

header = (f'  {"メンバー":>12}  {"ロール":>8}  {"試合":>4}  '
          f'{"このロール時WR":>12}  {"他ロール時WR":>11}  {"不在時WR":>8}  {"チーム平均比":>9}')
print(header)
print('  ' + '-' * 90)

for _, r in res_df.sort_values('diff_vs_overall', ascending=False).iterrows():
    other_str = f'{r["wr_other_role"]:>5.1f}%' if not np.isnan(r['wr_other_role']) else '  N/A'
    absent_str = f'{r["wr_absent"]:>5.1f}%' if not np.isnan(r['wr_absent']) else '  N/A'
    marker = ''
    if r['diff_vs_overall'] > 5:
        marker = ' ★'
    elif r['diff_vs_overall'] < -5:
        marker = ' ▼'

    print(f'  {r["member"]:>12}  {ROLE_JP[r["role"]]:>8}  {r["games"]:>4}  '
          f'{r["wr"]:>11.1f}%  {other_str:>11}  {absent_str:>8}  '
          f'{r["diff_vs_overall"]:>+8.1f}pp{marker}')

# =====================================================
#  3. ベスト・ワースト ロール配置
# =====================================================
print()
print()
print('=' * 75)
print('  ベスト/ワースト ロール配置 TOP5')
print('=' * 75)
print()

print('  【ベスト配置】チーム勝率が最も上がる配置')
for i, (_, r) in enumerate(res_df.sort_values('diff_vs_overall', ascending=False).head(5).iterrows()):
    print(f'    {i+1}. {r["member"]} → {ROLE_JP[r["role"]]}: '
          f'勝率{r["wr"]:.1f}% (チーム平均比 {r["diff_vs_overall"]:+.1f}pp, {int(r["games"])}試合)')

print()
print('  【ワースト配置】チーム勝率が最も下がる配置')
for i, (_, r) in enumerate(res_df.sort_values('diff_vs_overall', ascending=True).head(5).iterrows()):
    print(f'    {i+1}. {r["member"]} → {ROLE_JP[r["role"]]}: '
          f'勝率{r["wr"]:.1f}% (チーム平均比 {r["diff_vs_overall"]:+.1f}pp, {int(r["games"])}試合)')

# =====================================================
#  4. ロール固定 vs フレックス の影響
# =====================================================
print()
print()
print('=' * 75)
print('  メインロール vs オフロール 勝率比較')
print('=' * 75)
print()

for member in MEMBERS:
    member_ps = ps_m[ps_m['summonerName'] == member]
    role_counts = member_ps['role'].value_counts()
    if len(role_counts) < 2:
        main_role = role_counts.index[0]
        main_games = member_ps[member_ps['role'] == main_role]
        main_wr = main_games['win'].mean() * 100
        print(f'  {member:>12}: メイン {ROLE_JP[main_role]} のみ ({len(main_games)}試合 {main_wr:.1f}%)')
        continue

    main_role = role_counts.index[0]
    main_games = member_ps[member_ps['role'] == main_role]
    off_games = member_ps[member_ps['role'] != main_role]

    main_wr = main_games['win'].mean() * 100
    off_wr = off_games['win'].mean() * 100 if len(off_games) > 0 else 0
    diff = main_wr - off_wr

    print(f'  {member:>12}: メイン {ROLE_JP[main_role]} {main_wr:>5.1f}% ({len(main_games)}試合)'
          f'  オフロール {off_wr:>5.1f}% ({len(off_games)}試合)'
          f'  差 {diff:>+5.1f}pp')

# =====================================================
#  5. ロール別: 同ロールの「全プレイヤーベース」と比較した勝率
# =====================================================
print()
print()
print('=' * 75)
print('  ロール別 個人勝率 vs 全プレイヤー平均勝率')
print('=' * 75)
print()
print('  ※ 各ロールの全プレイヤー(敵味方含む885人)の平均勝率は約50%')
print('  ※ ここでは「同ロール・5試合以上」のプレイヤー全体の中での偏差値も表示')
print()

for role in ROLE_ORDER:
    print(f'  【{ROLE_JP[role]}】')

    # All players in this role with 5+ games
    role_all = ps[ps['role'] == role]
    player_wr = role_all.groupby('summonerName').agg(
        games=('win', 'count'), wr=('win', 'mean')
    ).reset_index()
    player_wr['wr'] = player_wr['wr'] * 100
    player_wr_5 = player_wr[player_wr['games'] >= 5]
    wr_mean = player_wr_5['wr'].mean()
    wr_std = player_wr_5['wr'].std()

    for member in MEMBERS:
        mp = player_wr[player_wr['summonerName'] == member]
        if len(mp) == 0 or mp.iloc[0]['games'] < 3:
            continue
        m_wr = mp.iloc[0]['wr']
        m_games = int(mp.iloc[0]['games'])
        if wr_std > 0 and m_games >= 5:
            z = (m_wr - wr_mean) / wr_std
            dev = 50 + z * 10
            print(f'    {member:>12}: {m_wr:>5.1f}% ({m_games:>3}試合)  偏差値 {dev:>4.1f}')
        else:
            print(f'    {member:>12}: {m_wr:>5.1f}% ({m_games:>3}試合)')
    print()
