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

STAT_COLS = [
    'kills', 'deaths', 'assists', 'kda',
    'cs', 'goldEarned', 'totalDamageDealtToChampions', 'totalDamageTaken',
    'visionScore', 'wardsPlaced', 'wardsKilled',
    'turretKills', 'dragonKills', 'baronKills',
    'firstBloodKill', 'doubleKills', 'tripleKills',
]

STAT_LABELS = {
    'kills': 'キル', 'deaths': 'デス', 'assists': 'アシスト', 'kda': 'KDA',
    'cs': 'CS', 'goldEarned': 'ゴールド', 'totalDamageDealtToChampions': 'ダメージ',
    'totalDamageTaken': '被ダメージ', 'visionScore': 'ビジョン',
    'wardsPlaced': 'ワード設置', 'wardsKilled': 'ワード除去',
    'turretKills': 'タワー', 'dragonKills': 'ドラゴン', 'baronKills': 'バロン',
    'firstBloodKill': 'FB率', 'doubleKills': 'ダブルキル', 'tripleKills': 'トリプルキル',
}

ROLE_JP = {'TOP': 'トップ', 'JUNGLE': 'ジャングル', 'MIDDLE': 'ミッド',
           'BOTTOM': 'ボトム', 'UTILITY': 'サポート'}

# =============================================================
#  Step 1: ロール別の全プレイヤーベース平均・標準偏差
# =============================================================
role_baseline = ps.groupby('role')[STAT_COLS].agg(['mean', 'std'])

# Early game gold diff baseline per role
for minute in [10, 15]:
    frame = tf[tf['timestampMin'] == minute]
    gd_stats = frame.groupby('role')['goldDiffVsOpponent'].agg(['mean', 'std'])
    role_baseline[(f'gd_{minute}m', 'mean')] = gd_stats['mean']
    role_baseline[(f'gd_{minute}m', 'std')] = gd_stats['std']

# Damage share: player damage / team total damage per game
team_dmg = ps.groupby(['matchId', 'teamId'])['totalDamageDealtToChampions'].transform('sum')
ps['dmg_share'] = ps['totalDamageDealtToChampions'] / team_dmg
dmg_share_baseline = ps.groupby('role')['dmg_share'].agg(['mean', 'std'])
role_baseline[('dmg_share', 'mean')] = dmg_share_baseline['mean']
role_baseline[('dmg_share', 'std')] = dmg_share_baseline['std']

# Gold share
team_gold = ps.groupby(['matchId', 'teamId'])['goldEarned'].transform('sum')
ps['gold_share'] = ps['goldEarned'] / team_gold
gold_share_baseline = ps.groupby('role')['gold_share'].agg(['mean', 'std'])
role_baseline[('gold_share', 'mean')] = gold_share_baseline['mean']
role_baseline[('gold_share', 'std')] = gold_share_baseline['std']

# KP (kill participation)
team_kills = ps.groupby(['matchId', 'teamId'])['kills'].transform('sum')
ps['kp'] = (ps['kills'] + ps['assists']) / team_kills.replace(0, 1)
kp_baseline = ps.groupby('role')['kp'].agg(['mean', 'std'])
role_baseline[('kp', 'mean')] = kp_baseline['mean']
role_baseline[('kp', 'std')] = kp_baseline['std']


# =============================================================
#  Step 2: メンバーごとにロール別にz-scoreを計算
# =============================================================
ps_m = ps[ps['summonerName'].isin(MEMBERS) & ps['role'].notna()].copy()

# Add early GD columns to ps_m
for minute in [10, 15]:
    frame = tf[tf['timestampMin'] == minute][['matchId', 'summonerName', 'goldDiffVsOpponent']]
    frame = frame.rename(columns={'goldDiffVsOpponent': f'gd_{minute}m'})
    ps_m = ps_m.merge(frame, on=['matchId', 'summonerName'], how='left')

ANALYSIS_COLS = STAT_COLS + ['gd_10m', 'gd_15m', 'dmg_share', 'gold_share', 'kp']

ANALYSIS_LABELS = {**STAT_LABELS,
    'gd_10m': '10分GD', 'gd_15m': '15分GD',
    'dmg_share': 'ダメ割合', 'gold_share': 'ゴールド割合', 'kp': 'KP'}

def calc_z(val, mean, std):
    if pd.isna(val) or pd.isna(mean) or pd.isna(std) or std == 0:
        return 0.0
    return (val - mean) / std

# Per-game z-scores: compare each game to the role baseline
z_records = []
for _, row in ps_m.iterrows():
    role = row['role']
    rec = {'matchId': row['matchId'], 'summonerName': row['summonerName'],
           'role': role, 'win': row['win']}
    for col in ANALYSIS_COLS:
        m = role_baseline.loc[role, (col, 'mean')]
        s = role_baseline.loc[role, (col, 'std')]
        rec[f'z_{col}'] = calc_z(row[col], m, s)
        rec[f'raw_{col}'] = row[col]
    z_records.append(rec)

z_df = pd.DataFrame(z_records)

# Aggregate z-scores per member (mean across all their games)
z_cols = [f'z_{c}' for c in ANALYSIS_COLS]
raw_cols = [f'raw_{c}' for c in ANALYSIS_COLS]

member_z = z_df.groupby('summonerName').agg(
    games=('win', 'count'),
    wins=('win', 'sum'),
    **{c: (c, 'mean') for c in z_cols},
    **{c: (c, 'mean') for c in raw_cols},
).reset_index()
member_z['winrate'] = member_z['wins'] / member_z['games'] * 100

# Role distribution
role_dist = ps_m.groupby(['summonerName', 'role']).size().unstack(fill_value=0)
role_pct = role_dist.div(role_dist.sum(axis=1), axis=0) * 100

# Top champions
top_champs = {}
for name in MEMBERS:
    player = ps_m[ps_m['summonerName'] == name]
    cs = player.groupby('championName').agg(g=('win', 'count'), w=('win', 'sum')).reset_index()
    cs['wr'] = cs['w'] / cs['g'] * 100
    cs = cs.sort_values('g', ascending=False).head(3)
    top_champs[name] = [(r['championName'], int(r['g']), r['wr']) for _, r in cs.iterrows()]


# =============================================================
#  Step 3: ロール別ベースライン表示
# =============================================================
print('=' * 75)
print('  ロール別 プレイヤーベース平均（全885人・212試合）')
print('=' * 75)
print()

display_cols = ['kills', 'deaths', 'assists', 'kda', 'cs', 'goldEarned',
                'totalDamageDealtToChampions', 'visionScore', 'kp', 'dmg_share']
header = f"{'ロール':>8}"
for c in display_cols:
    label = ANALYSIS_LABELS.get(c, c)
    header += f"  {label:>8}"
print(header)
print('-' * len(header))

for role in ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']:
    line = f"{ROLE_JP[role]:>8}"
    for c in display_cols:
        val = role_baseline.loc[role, (c, 'mean')]
        if c in ['kp', 'dmg_share']:
            line += f"  {val:>7.0%}"
        elif c in ['totalDamageDealtToChampions', 'goldEarned']:
            line += f"  {val:>8,.0f}"
        else:
            line += f"  {val:>8.1f}"
    print(line)

print()

# =============================================================
#  Step 4: 各メンバーの詳細分析
# =============================================================
print('=' * 75)
print('  プレイヤータイプ分析 — ロール平均比較ベース')
print('=' * 75)

# Classify based on z-scores
def classify_v2(row):
    tags = []

    z = lambda c: row[f'z_{c}']

    # --- Aggression / Kill pressure ---
    if z('kills') > 0.5 and z('kp') > 0.3:
        tags.append(('🗡', 'アグレッシブ', f"キルz={z('kills'):+.2f}, KPz={z('kp'):+.2f}"))
    elif z('kills') > 0.5:
        tags.append(('🗡', 'キル志向', f"キルz={z('kills'):+.2f}"))

    # --- Carry / Damage dealer ---
    if z('totalDamageDealtToChampions') > 0.4 and z('dmg_share') > 0.3:
        tags.append(('💥', 'ダメージキャリー', f"ダメz={z('totalDamageDealtToChampions'):+.2f}, ダメ割合z={z('dmg_share'):+.2f}"))

    # --- Tanky / Frontline ---
    if z('totalDamageTaken') > 0.5:
        tags.append(('🛡', 'フロントライン（同ロール比）', f"被ダメz={z('totalDamageTaken'):+.2f}"))

    # --- Support / Playmaking ---
    if z('assists') > 0.5 and z('kp') > 0.3:
        tags.append(('🤝', 'チームファイター', f"アシストz={z('assists'):+.2f}, KPz={z('kp'):+.2f}"))

    # --- Vision ---
    if z('visionScore') > 0.5:
        tags.append(('👁', '視界管理（同ロール比）', f"ビジョンz={z('visionScore'):+.2f}"))

    # --- Farm / Economy ---
    if z('cs') > 0.5:
        tags.append(('🌾', 'ファーム上手', f"CSz={z('cs'):+.2f}"))
    elif z('cs') < -0.5:
        tags.append(('🌾', 'ファーム課題', f"CSz={z('cs'):+.2f}"))

    # --- Early game ---
    gd10 = z('gd_10m')
    gd15 = z('gd_15m')
    if gd10 > 0.3 and gd15 > 0.3:
        tags.append(('⏰', '序盤強者（同ロール比）', f"10分GDz={gd10:+.2f}, 15分GDz={gd15:+.2f}"))
    elif gd10 < -0.3 and gd15 < -0.3:
        tags.append(('⏰', 'スロースターター（同ロール比）', f"10分GDz={gd10:+.2f}, 15分GDz={gd15:+.2f}"))

    # --- Death tendency ---
    if z('deaths') > 0.5:
        tags.append(('💀', 'デス多め（同ロール比）', f"デスz={z('deaths'):+.2f}"))
    elif z('deaths') < -0.3:
        tags.append(('✨', '生存力高い（同ロール比）', f"デスz={z('deaths'):+.2f}"))

    # --- Objective focus ---
    obj_z = (z('turretKills') + z('dragonKills') + z('baronKills')) / 3
    if obj_z > 0.3:
        tags.append(('🏰', 'オブジェクト重視', f"OBJ平均z={obj_z:+.2f}"))

    # --- Playmaker (multi-kills) ---
    mk_z = (z('doubleKills') + z('tripleKills')) / 2
    if mk_z > 0.5:
        tags.append(('⚡', 'プレイメーカー', f"マルチキルz={mk_z:+.2f}"))

    if not tags:
        tags.append(('⚖', 'ロール平均的', '目立った偏りなし'))

    return tags


for _, row in member_z.sort_values('winrate', ascending=False).iterrows():
    name = row['summonerName']

    # Role info
    roles_played = role_pct.loc[name]
    role_str_parts = []
    for role in roles_played.sort_values(ascending=False).index:
        pct = roles_played[role]
        if pct >= 5:
            role_str_parts.append(f"{ROLE_JP.get(role, role)} {pct:.0f}%")
    role_str = ' / '.join(role_str_parts)

    tags = classify_v2(row)

    print()
    print('━' * 75)
    print(f"  ■ {name}   {int(row['games'])}試合  勝率{row['winrate']:.1f}%")
    print(f"    ロール: {role_str}")
    print()

    # Show z-score deviation chart
    print('    【ロール平均との比較（z-score）】')
    print('    ※ 0=ロール平均  正=平均以上  負=平均以下')
    print()

    key_stats = [
        ('kills', 'キル'), ('deaths', 'デス'), ('assists', 'アシスト'),
        ('kda', 'KDA'), ('cs', 'CS'), ('kp', 'KP'),
        ('totalDamageDealtToChampions', 'ダメージ'),
        ('totalDamageTaken', '被ダメ'),
        ('dmg_share', 'ダメ割合'), ('gold_share', 'Gold割合'),
        ('visionScore', 'ビジョン'),
        ('gd_10m', '10分GD'), ('gd_15m', '15分GD'),
    ]

    for col, label in key_stats:
        z_val = row[f'z_{col}']
        raw_val = row[f'raw_{col}']

        # Visual bar
        bar_width = 20
        center = bar_width
        filled = int(min(abs(z_val), 2.5) / 2.5 * bar_width)
        if z_val >= 0:
            bar = '·' * center + '█' * filled + '·' * (bar_width - filled)
            marker = '▸'
        else:
            bar = '·' * (center - filled) + '█' * filled + '·' * bar_width
            marker = '◂'

        # Format raw value
        if col in ['kp', 'dmg_share', 'gold_share']:
            raw_str = f"{raw_val:.0%}"
        elif col in ['totalDamageDealtToChampions', 'totalDamageTaken', 'goldEarned']:
            raw_str = f"{raw_val:,.0f}"
        elif col in ['gd_10m', 'gd_15m']:
            raw_str = f"{raw_val:+.0f}"
        else:
            raw_str = f"{raw_val:.1f}"

        sign = '+' if z_val >= 0 else ''
        print(f"      {label:>8} {bar} {sign}{z_val:.2f}  ({raw_str})")

    print()
    print('    【タイプ判定】')
    for icon, tag, detail in tags:
        print(f"      {icon} {tag}  — {detail}")

    # Top champions
    champs = top_champs.get(name, [])
    if champs:
        champ_str = '  '.join([f"{c}({g}戦 {wr:.0f}%)" for c, g, wr in champs])
        print(f"    【主要チャンプ】 {champ_str}")


# =============================================================
#  Step 5: チーム内ランキング（ロール補正済み z-score ベース）
# =============================================================
print()
print()
print('=' * 75)
print('  チーム内ランキング（ロール平均との差 z-score 順）')
print('=' * 75)
print()
print('  ※ 各メンバーを「同ロールの全プレイヤー」と比較した上での順位')
print()

rank_items = [
    ('🗡 キル', 'z_kills'),
    ('🤝 アシスト', 'z_assists'),
    ('⚔ KDA', 'z_kda'),
    ('💥 ダメージ', 'z_totalDamageDealtToChampions'),
    ('🛡 被ダメ', 'z_totalDamageTaken'),
    ('🌾 CS', 'z_cs'),
    ('👁 ビジョン', 'z_visionScore'),
    ('🎯 KP', 'z_kp'),
    ('📊 ダメ割合', 'z_dmg_share'),
    ('⏰ 10分GD', 'z_gd_10m'),
    ('⏰ 15分GD', 'z_gd_15m'),
    ('💀 デス(低=良)', 'z_deaths'),
]

for label, col in rank_items:
    ascending = col == 'z_deaths'
    sorted_df = member_z.sort_values(col, ascending=ascending).reset_index(drop=True)
    entries = []
    for i, (_, r) in enumerate(sorted_df.head(3).iterrows()):
        medal = ['🥇', '🥈', '🥉'][i]
        val = r[col]
        entries.append(f"{medal}{r['summonerName']}({val:+.2f})")
    print(f"  {label}: {'  '.join(entries)}")


# =============================================================
#  Step 6: サマリー
# =============================================================
print()
print()
print('=' * 75)
print('  プレイヤータイプ サマリー（ロール補正済み）')
print('=' * 75)
print()

for _, row in member_z.sort_values('winrate', ascending=False).iterrows():
    name = row['summonerName']
    tags = classify_v2(row)

    # Build description from z-score profile
    strengths = []
    weaknesses = []

    z = lambda c: row[f'z_{c}']

    if z('kills') > 0.3: strengths.append('キル力')
    if z('assists') > 0.3: strengths.append('アシスト力')
    if z('kda') > 0.3: strengths.append('KDA')
    if z('totalDamageDealtToChampions') > 0.3: strengths.append('ダメージ')
    if z('visionScore') > 0.3: strengths.append('ビジョン')
    if z('cs') > 0.3: strengths.append('CS')
    if z('gd_15m') > 0.3: strengths.append('序盤GD')
    if z('kp') > 0.3: strengths.append('KP')
    if z('deaths') < -0.3: strengths.append('生存力')

    if z('kills') < -0.3: weaknesses.append('キル力')
    if z('deaths') > 0.3: weaknesses.append('デスの多さ')
    if z('cs') < -0.3: weaknesses.append('CS')
    if z('gd_15m') < -0.3: weaknesses.append('序盤GD')
    if z('visionScore') < -0.3: weaknesses.append('ビジョン')
    if z('kp') < -0.3: weaknesses.append('KP')

    tag_str = ' / '.join([f"{icon}{tag}" for icon, tag, _ in tags])

    roles_played = role_pct.loc[name]
    main_r = roles_played.idxmax()

    print(f"  ■ {name} ({ROLE_JP.get(main_r, main_r)})  — {tag_str}")
    if strengths:
        print(f"    ↑ 強み: {', '.join(strengths)}")
    if weaknesses:
        print(f"    ↓ 課題: {', '.join(weaknesses)}")
    print()
