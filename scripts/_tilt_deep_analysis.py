"""ティルト詳細分析 + 序盤ビハインド率 → 真のティルト王ランキング"""
import pandas as pd
import numpy as np
import yaml
import sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / 'config/settings.yaml', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)
MEMBERS = [m['game_name'] for m in _cfg['members']]

ps = pd.read_csv('data/processed/player_stats.csv')
tf = pd.read_csv('data/processed/timeline_frames.csv')
ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
ROLE_JP = {'TOP': 'トップ', 'JUNGLE': 'ジャングル', 'MIDDLE': 'ミッド',
           'BOTTOM': 'ボトム', 'UTILITY': 'サポート'}
METRICS = ['kda', 'deaths', 'kp', 'dmg_share', 'visionScore', 'cs']

EARLY_MIN = 10
BEHIND_THRESHOLD = -300

# ────────────────────── prep ──────────────────────
ps_all = ps[ps['role'].notna()].copy()
team_kills = ps.groupby(['matchId', 'teamId'])['kills'].transform('sum')
ps['kp'] = (ps['kills'] + ps['assists']) / team_kills.replace(0, 1)
ps_all['kp'] = ps.loc[ps_all.index, 'kp']
team_dmg = ps.groupby(['matchId', 'teamId'])['totalDamageDealtToChampions'].transform('sum')
ps['dmg_share'] = ps['totalDamageDealtToChampions'] / team_dmg.replace(0, 1)
ps_all['dmg_share'] = ps.loc[ps_all.index, 'dmg_share']

early = tf[tf['timestampMin'] == EARLY_MIN][['matchId', 'summonerName', 'goldDiffVsOpponent']].copy()
early = early.rename(columns={'goldDiffVsOpponent': 'earlyGD'})
ps_all = ps_all.merge(early, on=['matchId', 'summonerName'], how='left')
ps_all = ps_all.dropna(subset=['earlyGD'])
ps_all['behind'] = ps_all['earlyGD'] < BEHIND_THRESHOLD
ps_all['ahead'] = ps_all['earlyGD'] > abs(BEHIND_THRESHOLD)
ps_all['even'] = (~ps_all['behind']) & (~ps_all['ahead'])
ps_all['is_member'] = ps_all['summonerName'].isin(MEMBERS)

# ═══════════════════════════════════════════════════════════════
#  PART 1: 序盤ビハインド率ランキング
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 90)
print('  PART 1: 序盤ビハインド率ランキング')
print('  〜 10分時点で GD < -300G のビハインドを背負う頻度 〜')
print('=' * 90)
print()

non_member = ps_all[~ps_all['is_member']]
nm_behind_rate = non_member['behind'].mean() * 100
print(f'  📊 全プレイヤー平均ビハインド率: {nm_behind_rate:.1f}%')
print()

behind_data = []
for name in MEMBERS:
    p = ps_all[ps_all['summonerName'] == name]
    total = len(p)
    n_behind = int(p['behind'].sum())
    n_ahead = int(p['ahead'].sum())
    n_even = int(p['even'].sum())
    behind_rate = n_behind / total * 100
    ahead_rate = n_ahead / total * 100
    even_rate = n_even / total * 100
    avg_gd = p['earlyGD'].mean()
    median_gd = p['earlyGD'].median()
    behind_avg_gd = p[p['behind']]['earlyGD'].mean() if n_behind > 0 else 0
    behind_wr = p[p['behind']]['win'].mean() * 100 if n_behind > 0 else 0
    ahead_wr = p[p['ahead']]['win'].mean() * 100 if n_ahead > 0 else 0

    n_severe = int((p['earlyGD'] < -1000).sum())
    severe_rate = n_severe / total * 100

    behind_data.append({
        'name': name, 'total': total,
        'n_behind': n_behind, 'behind_rate': behind_rate,
        'n_ahead': n_ahead, 'ahead_rate': ahead_rate,
        'n_even': n_even, 'even_rate': even_rate,
        'avg_gd': avg_gd, 'median_gd': median_gd,
        'behind_avg_gd': behind_avg_gd,
        'behind_wr': behind_wr, 'ahead_wr': ahead_wr,
        'n_severe': n_severe, 'severe_rate': severe_rate,
    })

bdf = pd.DataFrame(behind_data).sort_values('behind_rate', ascending=False)

hdr = (f'  {"メンバー":<14} {"試合数":>6} {"ビハインド率":>10} {"イーブン率":>10}'
       f' {"アヘッド率":>10} {"平均GD@10":>10} {"負け時平均GD":>12}')
print(hdr)
print(f'  {"─"*14} {"─"*6} {"─"*10} {"─"*10} {"─"*10} {"─"*10} {"─"*12}')
for _, r in bdf.iterrows():
    bar = '█' * int(r['behind_rate'] / 3)
    print(f'  {r["name"]:<14} {int(r["total"]):>6}'
          f' {r["behind_rate"]:>8.1f}%  {r["even_rate"]:>8.1f}%'
          f'  {r["ahead_rate"]:>8.1f}%  {r["avg_gd"]:>+8.0f}G'
          f'  {r["behind_avg_gd"]:>+10.0f}G')
    print(f'  {"":>22}{bar}')

print()
print('  ※ イーブン = GD が -300G ～ +300G の範囲')

# ── ロール別ビハインド率 ──
print()
print('  【ロール別ビハインド率】')
header = f'  {"メンバー":<14}'
for role in ROLES:
    header += f' {ROLE_JP[role]:>12}'
print(header)
print(f'  {"─"*14}', end='')
for _ in ROLES:
    print(f' {"─"*12}', end='')
print()

for _, r in bdf.iterrows():
    p = ps_all[ps_all['summonerName'] == r['name']]
    line = f'  {r["name"]:<14}'
    for role in ROLES:
        pr = p[p['role'] == role]
        if len(pr) >= 5:
            rate = pr['behind'].mean() * 100
            line += f' {rate:>5.0f}%({len(pr):>3})'
        else:
            line += f'       {"−":>5}'
    print(line)


# ═══════════════════════════════════════════════════════════════
#  PART 2: ビハインド時の勝率（逆転力）
# ═══════════════════════════════════════════════════════════════
print()
print()
print('=' * 90)
print('  PART 2: ビハインド時の勝率（逆転力）')
print('  〜 序盤負けからどれだけ逆転できるか 〜')
print('=' * 90)
print()

bdf_wr = bdf.sort_values('behind_wr', ascending=True)
print(f'  {"メンバー":<14} {"ビハインドWR":>12} {"アヘッドWR":>10}'
      f' {"WR差":>8} {"ビハインド試合":>12}')
print(f'  {"─"*14} {"─"*12} {"─"*10} {"─"*8} {"─"*12}')
for _, r in bdf_wr.iterrows():
    diff = r['ahead_wr'] - r['behind_wr']
    print(f'  {r["name"]:<14} {r["behind_wr"]:>10.1f}%  {r["ahead_wr"]:>8.1f}%'
          f'  {diff:>+6.1f}pp  {int(r["n_behind"]):>6}/{int(r["total"])}')


# ═══════════════════════════════════════════════════════════════
#  PART 3: ビハインド深度別パフォーマンス（ティルト段階分析）
# ═══════════════════════════════════════════════════════════════
print()
print()
print('=' * 90)
print('  PART 3: ビハインド深度別パフォーマンス')
print('  〜 GD差が深くなるにつれてどう崩れるか 〜')
print('=' * 90)
print()

tiers = [
    ('アヘッド (+300G~)',     300,  99999),
    ('イーブン (-300~+300)', -300,  300),
    ('軽微 (-300~-600G)',    -600, -300),
    ('中程度 (-600~-1000G)', -1000, -600),
    ('深刻 (-1000~-1500G)', -1500, -1000),
    ('壊滅 (-1500G~)',      -99999, -1500),
]

for name in MEMBERS:
    p = ps_all[ps_all['summonerName'] == name]
    if len(p) < 20:
        continue

    print(f'  ■ {name}')
    print(f'    {"深度":<24} {"試合":>4} {"KDA":>6} {"デス":>6}'
          f' {"KP":>6} {"WR":>6} {"ダメ割":>6}')
    print(f'    {"─"*24} {"─"*4} {"─"*6} {"─"*6}'
          f' {"─"*6} {"─"*6} {"─"*6}')

    for tier_name, lo, hi in tiers:
        if lo == -99999:
            tier = p[p['earlyGD'] < hi]
        elif hi == 99999:
            tier = p[p['earlyGD'] >= lo]
        else:
            tier = p[(p['earlyGD'] >= lo) & (p['earlyGD'] < hi)]
        n = len(tier)
        if n < 2:
            print(f'    {tier_name:<24} {n:>4}  （データ不足）')
            continue
        t_kda = tier['kda'].mean()
        t_deaths = tier['deaths'].mean()
        t_kp = tier['kp'].mean()
        t_wr = tier['win'].mean() * 100
        t_dmg = tier['dmg_share'].mean()
        print(f'    {tier_name:<24} {n:>4} {t_kda:>6.2f} {t_deaths:>6.1f}'
              f' {t_kp:>5.0%}  {t_wr:>4.0f}%  {t_dmg:>4.0%}')
    print()


# ═══════════════════════════════════════════════════════════════
#  PART 4: 連敗ティルト分析
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 90)
print('  PART 4: 連敗ティルト分析')
print('  〜 直前の試合に負けた後、次の試合のパフォーマンスは落ちるか 〜')
print('=' * 90)
print()

streak_records = []
for name in MEMBERS:
    p = ps_all[ps_all['summonerName'] == name].sort_values('matchId').reset_index(drop=True)
    if len(p) < 20:
        continue

    p['prev_win'] = p['win'].shift(1)
    p['prev2_win'] = p['win'].shift(2)
    p = p.dropna(subset=['prev_win'])

    after_win = p[p['prev_win'] == True]
    after_loss = p[p['prev_win'] == False]
    after_streak = p[(p['prev_win'] == False) & (p.get('prev2_win', pd.Series()) == False)]

    aw_kda = after_win['kda'].mean()
    aw_deaths = after_win['deaths'].mean()
    aw_wr = after_win['win'].mean() * 100

    al_kda = after_loss['kda'].mean()
    al_deaths = after_loss['deaths'].mean()
    al_wr = after_loss['win'].mean() * 100

    kda_diff = ((aw_kda - al_kda) / aw_kda * 100) if aw_kda > 0 else 0
    death_diff = ((al_deaths - aw_deaths) / aw_deaths * 100) if aw_deaths > 0 else 0

    print(f'  ■ {name}')
    print(f'    勝利後 → KDA {aw_kda:.2f}  デス {aw_deaths:.1f}'
          f'  WR {aw_wr:.0f}%  ({len(after_win)}試合)')
    print(f'    敗北後 → KDA {al_kda:.2f}  デス {al_deaths:.1f}'
          f'  WR {al_wr:.0f}%  ({len(after_loss)}試合)'
          f'  │ KDA{kda_diff:+.0f}%  デス{death_diff:+.0f}%')

    if len(after_streak) >= 5:
        as_kda = after_streak['kda'].mean()
        as_deaths = after_streak['deaths'].mean()
        as_wr = after_streak['win'].mean() * 100
        sk_diff = ((aw_kda - as_kda) / aw_kda * 100) if aw_kda > 0 else 0
        print(f'    2連敗後→ KDA {as_kda:.2f}  デス {as_deaths:.1f}'
              f'  WR {as_wr:.0f}%  ({len(after_streak)}試合)'
              f'  │ KDA{sk_diff:+.0f}%')

    streak_records.append({
        'name': name,
        'kda_drop_after_loss': kda_diff,
        'death_inc_after_loss': death_diff,
        'wr_after_loss': al_wr,
    })
    print()

print('  【連敗ティルトサマリー】')
sdf = pd.DataFrame(streak_records).sort_values('kda_drop_after_loss', ascending=False)
print(f'  {"メンバー":<14} {"KDA低下":>8} {"デス増加":>8} {"敗北後WR":>8}')
print(f'  {"─"*14} {"─"*8} {"─"*8} {"─"*8}')
for _, r in sdf.iterrows():
    print(f'  {r["name"]:<14} {r["kda_drop_after_loss"]:>+6.0f}%'
          f' {r["death_inc_after_loss"]:>+6.0f}%'
          f' {r["wr_after_loss"]:>6.0f}%')


# ═══════════════════════════════════════════════════════════════
#  PART 5: 真のティルト王ランキング
# ═══════════════════════════════════════════════════════════════
print()
print()
print('=' * 90)
print('  PART 5: 真のティルト王ランキング')
print('  〜 ティルト度 × ビハインド頻度 = チームへの実害度 〜')
print('=' * 90)
print()
print('  計算式: 真のスコア = ティルトスコア × (ビハインド率 / チーム平均ビハインド率)')
print('  → ティルトしやすく、かつ頻繁にビハインドを背負う人 = 真のティルト王')
print()

def calc_degradation_fn(df_ahead, df_behind):
    if len(df_ahead) < 3 or len(df_behind) < 3:
        return None
    rec = {}
    for m in METRICS:
        val_a = df_ahead[m].mean()
        val_b = df_behind[m].mean()
        if m == 'deaths':
            rec[f'{m}_change'] = ((val_b - val_a) / val_a * 100) if val_a > 0 else 0
        elif m in ('kp', 'dmg_share'):
            rec[f'{m}_change'] = (val_a - val_b) * 100
        else:
            rec[f'{m}_change'] = ((val_a - val_b) / val_a * 100) if val_a > 0 else 0
    rec['wr_drop'] = df_ahead['win'].mean() * 100 - df_behind['win'].mean() * 100
    return rec

role_baselines = {}
for role in ROLES:
    rd = ps_all[ps_all['role'] == role]
    bl = calc_degradation_fn(rd[rd['ahead']], rd[rd['behind']])
    if bl:
        role_baselines[role] = bl

member_records = []
for name in MEMBERS:
    player = ps_all[ps_all['summonerName'] == name]
    p_behind = player[player['behind']]
    p_ahead = player[player['ahead']]
    if len(p_behind) < 3 or len(p_ahead) < 3:
        continue

    role_diffs = []
    for role in ROLES:
        pr = player[player['role'] == role]
        if len(pr) < 3 or role not in role_baselines:
            continue
        pr_b = pr[pr['behind']]
        pr_a = pr[pr['ahead']]
        if len(pr_b) < 2 or len(pr_a) < 2:
            continue
        bl = role_baselines[role]
        pd_res = calc_degradation_fn(pr_a, pr_b)
        if pd_res is None:
            continue
        excess = {}
        for m in METRICS:
            excess[m] = pd_res[f'{m}_change'] - bl[f'{m}_change']
        excess['wr_drop'] = pd_res['wr_drop'] - bl['wr_drop']
        role_diffs.append({'role': role, 'n': len(pr),
                           **{f'excess_{k}': v for k, v in excess.items()}})

    if not role_diffs:
        continue
    rd_df = pd.DataFrame(role_diffs)
    total_n = rd_df['n'].sum()
    weights = rd_df['n'] / total_n

    rec = {'name': name, 'total': len(player),
           'n_behind': int(p_behind.shape[0]),
           'behind_rate': len(p_behind) / len(player) * 100}
    for m in METRICS:
        rec[f'excess_{m}'] = (rd_df[f'excess_{m}'] * weights).sum()
    rec['excess_wr_drop'] = (rd_df['excess_wr_drop'] * weights).sum()
    member_records.append(rec)

mdf = pd.DataFrame(member_records)

def norm(series, higher_is_worse=True):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    n = (series - mn) / (mx - mn) * 100
    return n if higher_is_worse else (100 - n)

mdf['tilt_score'] = (
    norm(mdf['excess_kda']) * 0.25 +
    norm(mdf['excess_deaths']) * 0.25 +
    norm(mdf['excess_kp']) * 0.15 +
    norm(mdf['excess_dmg_share']) * 0.10 +
    norm(mdf['excess_visionScore']) * 0.05 +
    norm(mdf['excess_cs']) * 0.05 +
    norm(mdf['excess_wr_drop']) * 0.15
)

avg_behind = mdf['behind_rate'].mean()
mdf['behind_mult'] = mdf['behind_rate'] / avg_behind
mdf['true_tilt'] = mdf['tilt_score'] * mdf['behind_mult']

# Add severe behind stats
for name in MEMBERS:
    p = ps_all[ps_all['summonerName'] == name]
    n_sv = int((p['earlyGD'] < -1000).sum())
    sv_rate = n_sv / len(p) * 100 if len(p) > 0 else 0
    idx = mdf[mdf['name'] == name].index
    if len(idx) > 0:
        mdf.loc[idx[0], 'severe_rate'] = sv_rate
        mdf.loc[idx[0], 'n_severe'] = n_sv

mdf = mdf.sort_values('true_tilt', ascending=False)

medals = ['👑 1位', '🔥 2位', '💀 3位', '😰 4位', '😐 5位', '🙂 6位', '🧘 7位']

print(f'  {"順位":<8} {"メンバー":<14} {"ティルト度":>8} {"ビハインド率":>10}'
      f' {"倍率":>6} {"真スコア":>8} {"深刻率":>8}')
print(f'  {"─"*8} {"─"*14} {"─"*8} {"─"*10}'
      f' {"─"*6} {"─"*8} {"─"*8}')
for rank, (_, r) in enumerate(mdf.iterrows()):
    medal = medals[rank] if rank < len(medals) else f'   {rank+1}位'
    sv = r.get('severe_rate', 0)
    n_sv = int(r.get('n_severe', 0))
    print(f'  {medal}  {r["name"]:<14}'
          f' {r["tilt_score"]:>6.1f}  {r["behind_rate"]:>8.1f}%'
          f'  {r["behind_mult"]:>5.2f}x {r["true_tilt"]:>7.1f}'
          f'  {sv:>5.1f}%')

print()
print(f'  ※ 倍率 = ビハインド率 ÷ チーム平均ビハインド率({avg_behind:.1f}%)')
print(f'  ※ 深刻率 = 10分時点で GD < -1000G の試合の割合')
print()

king = mdf.iloc[0]
saint = mdf.iloc[-1]

print('  ╔' + '═' * 72 + '╗')
print(f'  ║  👑 真のティルト王: {king["name"]}')
print(f'  ║')
print(f'  ║  ティルト度: {king["tilt_score"]:.1f}/100（序盤負けた時の崩れやすさ）')
print(f'  ║  ビハインド率: {king["behind_rate"]:.1f}%'
      f'（チーム平均{avg_behind:.1f}%の{king["behind_mult"]:.2f}倍）')
print(f'  ║  → 真のティルトスコア: {king["true_tilt"]:.1f}')
print(f'  ║')
print(f'  ║  🧘 最も無害: {saint["name"]}')
print(f'  ║  ティルト度: {saint["tilt_score"]:.1f}/100')
print(f'  ║  ビハインド率: {saint["behind_rate"]:.1f}%'
      f'（{saint["behind_mult"]:.2f}倍）')
print(f'  ║  → 真のティルトスコア: {saint["true_tilt"]:.1f}')
print('  ╚' + '═' * 72 + '╝')
print()
