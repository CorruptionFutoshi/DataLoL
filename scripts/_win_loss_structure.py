"""Structural comparison between wins and losses."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
CONFIG = ROOT / "config" / "settings.yaml"


def main():
    with open(CONFIG, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    member_names_list = [m['game_name'] for m in cfg.get('members', [])]
    members = {f'{m["game_name"]}#{m["tag_line"]}' for m in cfg.get('members', [])}

    ps = pd.read_csv(DATA / 'player_stats.csv')
    ps['riotId'] = ps['summonerName'].astype(str) + '#' + ps['tagLine'].astype(str)
    ps['is_member'] = ps['riotId'].isin(members)

    tf = pd.read_csv(DATA / 'timeline_frames.csv')
    tf['is_member'] = tf['summonerName'].isin(member_names_list)

    our = ps[ps['is_member']].copy()
    match_stats = our.groupby('matchId').agg(
        win=('win', 'first'),
        team_kills=('kills', 'sum'),
        team_deaths=('deaths', 'sum'),
        team_assists=('assists', 'sum'),
        team_dmg=('totalDamageDealtToChampions', 'sum'),
        team_gold=('goldEarned', 'sum'),
        team_vision=('visionScore', 'sum'),
    ).reset_index()

    matches = pd.read_csv(DATA / 'matches.csv')
    match_stats = match_stats.merge(matches[['matchId', 'gameDurationMin']], on='matchId', how='left')
    match_stats['duration_min'] = match_stats['gameDurationMin']

    gd15 = tf[tf['timestampMin'] == 15].copy()
    our_gd15 = gd15[gd15['is_member']].groupby('matchId')['goldDiffVsOpponent'].sum().reset_index()
    our_gd15.columns = ['matchId', 'team_gd15']
    match_stats = match_stats.merge(our_gd15, on='matchId', how='left')

    wins = match_stats[match_stats['win'] == True]
    losses = match_stats[match_stats['win'] == False]

    print("=" * 70)
    print("  勝ち試合 vs 負け試合: 構造比較")
    print("=" * 70)

    metrics = [
        ('チームキル合計', 'team_kills'),
        ('チームデス合計', 'team_deaths'),
        ('チームアシスト合計', 'team_assists'),
        ('チームDMG合計', 'team_dmg'),
        ('チームGold合計', 'team_gold'),
        ('チームVision合計', 'team_vision'),
        ('試合時間(分)', 'duration_min'),
        ('チームGD@15合計', 'team_gd15'),
    ]

    print(f"\n  {'指標':<20} {'勝ち平均':>12} {'負け平均':>12} {'差':>12}")
    print("  " + "-" * 58)
    for label, col in metrics:
        w = wins[col].mean()
        l = losses[col].mean()
        d = w - l
        sign = '+' if d > 0 else ''
        print(f"  {label:<20} {w:>12.1f} {l:>12.1f} {sign}{d:>11.1f}")

    print(f"\n■ 負け試合のチームデス数分布")
    bins = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 100)]
    for lo, hi in bins:
        cnt = len(losses[(losses['team_deaths'] >= lo) & (losses['team_deaths'] < hi)])
        pct = cnt / len(losses) * 100
        print(f"  {lo}-{hi}デス: {cnt}試合 ({pct:.1f}%)")

    gd_losses = losses.dropna(subset=['team_gd15'])
    gd_wins = wins.dropna(subset=['team_gd15'])

    print(f"\n■ 負け試合のGD@15分布")
    behind = len(gd_losses[gd_losses['team_gd15'] < 0])
    ahead = len(gd_losses[gd_losses['team_gd15'] > 0])
    print(f"  15分時点でビハインド: {behind}試合 ({behind / len(gd_losses) * 100:.1f}%)")
    print(f"  15分時点でリード: {ahead}試合 ({ahead / len(gd_losses) * 100:.1f}%)")
    print(f"  平均GD@15（負け試合）: {gd_losses['team_gd15'].mean():.0f}G")
    print(f"  平均GD@15（勝ち試合）: {gd_wins['team_gd15'].mean():.0f}G")

    losses_c = losses.copy()
    wins_c = wins.copy()
    losses_c['kill_per_death'] = losses_c['team_kills'] / losses_c['team_deaths'].clip(lower=1)
    wins_c['kill_per_death'] = wins_c['team_kills'] / wins_c['team_deaths'].clip(lower=1)

    print(f"\n■ キル効率（チームキル/チームデス）")
    print(f"  勝ち試合: {wins_c['kill_per_death'].mean():.2f}")
    print(f"  負け試合: {losses_c['kill_per_death'].mean():.2f}")

    losses_c['dmg_per_death'] = losses_c['team_dmg'] / losses_c['team_deaths'].clip(lower=1)
    wins_c['dmg_per_death'] = wins_c['team_dmg'] / wins_c['team_deaths'].clip(lower=1)
    print(f"\n■ 1デスあたりのダメージ出力")
    print(f"  勝ち試合: {wins_c['dmg_per_death'].mean():.0f}")
    print(f"  負け試合: {losses_c['dmg_per_death'].mean():.0f}")
    print(f"  差: {wins_c['dmg_per_death'].mean() - losses_c['dmg_per_death'].mean():.0f}")

    thresholds = [500, 1000, 2000, 3000]
    print(f"\n■ ビハインド幅別の逆転率")
    for th in thresholds:
        bw = len(gd_wins[gd_wins['team_gd15'] < -th])
        bl = len(gd_losses[gd_losses['team_gd15'] < -th])
        total = bw + bl
        if total > 0:
            print(f"  {th}G以上ビハインド: {total}試合 → 勝ち{bw} ({bw / total * 100:.1f}%)")

    print(f"\n■ リード幅別のスノーボール成功率")
    for th in thresholds:
        lw = len(gd_wins[gd_wins['team_gd15'] > th])
        ll = len(gd_losses[gd_losses['team_gd15'] > th])
        total = lw + ll
        if total > 0:
            blown = ll
            print(f"  {th}G以上リード: {total}試合 → 勝ち{lw} ({lw / total * 100:.1f}%) / 逆転された{blown}試合")

    # KP bracket analysis (from teamfight data)
    print(f"\n■ 中盤KP帯別の勝率（分析済みデータ再掲）")
    print(f"  KP 0-30%:  勝率41.6%")
    print(f"  KP 30-50%: 勝率48.5%")
    print(f"  KP 50-70%: 勝率53.4% ← ピーク")
    print(f"  KP 70-100%:勝率45.2% ← 過剰参加で低下")

    # Vision in wins vs losses
    print(f"\n■ ビジョンスコア比較")
    print(f"  勝ち試合: 平均{wins['team_vision'].mean():.1f}")
    print(f"  負け試合: 平均{losses['team_vision'].mean():.1f}")
    print(f"  差: {wins['team_vision'].mean() - losses['team_vision'].mean():.1f}")

    # Average deaths per minute in wins vs losses (death pacing)
    wins_c['deaths_per_min'] = wins_c['team_deaths'] / wins_c['duration_min']
    losses_c['deaths_per_min'] = losses_c['team_deaths'] / losses_c['duration_min']
    print(f"\n■ 1分あたりチームデス数（死亡ペース）")
    print(f"  勝ち試合: {wins_c['deaths_per_min'].mean():.2f} デス/分")
    print(f"  負け試合: {losses_c['deaths_per_min'].mean():.2f} デス/分")
    print(f"  負けのほうが {losses_c['deaths_per_min'].mean() / wins_c['deaths_per_min'].mean():.1f}倍速く死んでいる")

    # Deaths in last 10 min vs first 15 min comparison
    print(f"\n■ 勝ち試合 vs 負け試合: デスタイミング分析")
    events = pd.read_csv(DATA / 'timeline_events.csv')
    champ_kills = events[events['eventType'] == 'CHAMPION_KILL'].copy()

    mt = ps[ps['is_member']][['matchId', 'teamId']].drop_duplicates()

    early_deaths_w = []
    late_deaths_w = []
    early_deaths_l = []
    late_deaths_l = []

    for _, row in mt.iterrows():
        mid = row['matchId']
        tid = row['teamId']
        win = match_stats[match_stats['matchId'] == mid]['win'].values
        if len(win) == 0:
            continue
        win = win[0]
        dur = match_stats[match_stats['matchId'] == mid]['duration_min'].values[0]

        mk = champ_kills[champ_kills['matchId'] == mid]
        our_d = mk[mk['victimTeamId'] == tid]

        early = len(our_d[our_d['timestampMin'] <= 15])
        late = len(our_d[our_d['timestampMin'] > dur - 10]) if dur > 10 else 0

        if win:
            early_deaths_w.append(early)
            late_deaths_w.append(late)
        else:
            early_deaths_l.append(early)
            late_deaths_l.append(late)

    print(f"  勝ち試合 - 序盤デス(0-15分): 平均{np.mean(early_deaths_w):.1f}  終盤デス(ラスト10分): 平均{np.mean(late_deaths_w):.1f}")
    print(f"  負け試合 - 序盤デス(0-15分): 平均{np.mean(early_deaths_l):.1f}  終盤デス(ラスト10分): 平均{np.mean(late_deaths_l):.1f}")
    print(f"  → 負け試合の終盤デスは勝ちの{np.mean(late_deaths_l) / max(np.mean(late_deaths_w), 0.1):.1f}倍")


if __name__ == '__main__':
    main()
