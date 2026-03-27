"""
レーン配置 複合判断スクリプト

5つの独立したシグナルを統合し、各メンバー×ロールを多次元で評価する。
回帰モデル (optimal_lane_advanced.py) の結果は1つのシグナルとして扱い、
他の視点と合わせた合意度で総合判断を行う。

シグナル:
  1. 回帰独立効果 — Ridge回帰による玉突き分離後の限界効果
  2. チーム勝率影響 — そのメンバーがそのロールにいるときのチーム勝率変化
  3. 序盤レーニング力 — 15分GD z-score × レーン勝敗相関
  4. プレイヤー適性 — ロール期待プロファイルへのフィット度
  5. 実測パターン実績 — そのメンバー×ロールを含む配置の加重平均勝率
"""
import sys, io, warnings, argparse
import yaml
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
import numpy as np
from itertools import combinations, permutations
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / 'config/settings.yaml', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)
MEMBERS = [m['game_name'] for m in _cfg['members']]

parser = argparse.ArgumentParser()
parser.add_argument('--last', type=int, default=0,
                    help='直近N試合に絞る (0=全試合)')
args = parser.parse_args()

# ─── データ読み込み ───
ps = pd.read_csv('data/processed/player_stats.csv')
tf = pd.read_csv('data/processed/timeline_frames.csv')
ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
ROLE_JP = {'TOP': 'トップ', 'JUNGLE': 'JG', 'MIDDLE': 'ミッド',
           'BOTTOM': 'BOT', 'UTILITY': 'SUP'}

MIN_GAMES = 15

ps_m = ps[ps['summonerName'].isin(MEMBERS)].copy()
ps_m = ps_m[ps_m['role'].notna()]

if args.last > 0:
    recent = (ps_m.groupby('matchId')['matchId'].first()
              .sort_index().tail(args.last).index)
    ps_m = ps_m[ps_m['matchId'].isin(recent)]
    tf = tf[tf['matchId'].isin(recent)]

match_win = ps_m.groupby('matchId')['win'].first()
overall_wr = match_win.mean() * 100

# ═══════════════════════════════════════════════════════════════
#  シグナル 1: 回帰独立効果（軽量版 — Bootstrap なし）
# ═══════════════════════════════════════════════════════════════

feature_rows = []
for mid in match_win.index:
    row = {}
    mp = ps_m[ps_m['matchId'] == mid]
    for _, p in mp.iterrows():
        row[f'{p["summonerName"]}@{p["role"]}'] = 1
    feature_rows.append(row)

feat_df = pd.DataFrame(feature_rows, index=match_win.index).fillna(0)
valid_cols = [c for c in feat_df.columns if feat_df[c].sum() >= MIN_GAMES]
feat_df = feat_df[valid_cols]

X = feat_df.values.astype(float)
y = match_win.values.astype(float)

logit = LogisticRegressionCV(Cs=np.logspace(-2, 2, 30), cv=5,
                             solver='lbfgs', max_iter=5000, random_state=42)
logit.fit(X, y)

marginal_pp = logit.coef_[0] * 0.25 * 100
reg_map = {}
for i, col in enumerate(valid_cols):
    member, role = col.split('@')
    if member in MEMBERS:
        reg_map[(member, role)] = {
            'pp': marginal_pp[i],
            'n': int(feat_df[col].sum()),
        }

# ═══════════════════════════════════════════════════════════════
#  シグナル 2: チーム勝率影響
# ═══════════════════════════════════════════════════════════════

member_role_matches = {}
for member in MEMBERS:
    mp = ps_m[ps_m['summonerName'] == member]
    for role in ROLES:
        ids = mp[mp['role'] == role]['matchId'].unique()
        if len(ids) >= MIN_GAMES:
            wr = match_win.loc[match_win.index.isin(ids)].mean() * 100
            member_role_matches[(member, role)] = {
                'team_wr': wr,
                'diff_pp': wr - overall_wr,
                'n': len(ids),
            }

# ═══════════════════════════════════════════════════════════════
#  シグナル 3: 序盤レーニング力 (15分 GD z-score × 勝敗相関)
# ═══════════════════════════════════════════════════════════════

tf15 = tf[tf['timestampMin'] == 15].copy()
tf15_members = tf15[tf15['summonerName'].isin(MEMBERS)]

role_gd_base = tf15.groupby('role')['goldDiffVsOpponent'].agg(['mean', 'std'])

role_corr = {}
for role in ROLES:
    rd = tf15[tf15['role'] == role]
    if len(rd) > 5:
        role_corr[role] = rd['goldDiffVsOpponent'].corr(
            rd['win'].astype(float))
    else:
        role_corr[role] = 0.0

lane_signal = {}
for member in MEMBERS:
    for role in ROLES:
        md = tf15_members[(tf15_members['summonerName'] == member)
                          & (tf15_members['role'] == role)]
        if len(md) < MIN_GAMES or role not in role_gd_base.index:
            continue
        gd_mean = md['goldDiffVsOpponent'].mean()
        base_mean = role_gd_base.loc[role, 'mean']
        base_std = role_gd_base.loc[role, 'std']
        z = (gd_mean - base_mean) / base_std if base_std > 0 else 0
        corr = role_corr.get(role, 0)
        lane_signal[(member, role)] = {
            'gd_z': z,
            'lane_corr': corr,
            'impact': z * abs(corr),
            'n': len(md),
        }

# ═══════════════════════════════════════════════════════════════
#  シグナル 4: プレイヤー適性 (ロール期待プロファイル vs 実績 z-score)
# ═══════════════════════════════════════════════════════════════

STAT_COLS = ['kills', 'deaths', 'assists', 'kda', 'cs', 'goldEarned',
             'totalDamageDealtToChampions', 'totalDamageTaken',
             'visionScore', 'wardsPlaced']

role_stat_base = ps.groupby('role')[STAT_COLS].agg(['mean', 'std'])

team_kills = ps.groupby(['matchId', 'teamId'])['kills'].transform('sum')
ps['kp'] = (ps['kills'] + ps['assists']) / team_kills.replace(0, 1)
kp_base = ps.groupby('role')['kp'].agg(['mean', 'std'])

team_dmg = ps.groupby(['matchId', 'teamId'])[
    'totalDamageDealtToChampions'].transform('sum')
ps['dmg_share'] = ps['totalDamageDealtToChampions'] / team_dmg.replace(0, 1)
dmg_share_base = ps.groupby('role')['dmg_share'].agg(['mean', 'std'])

ROLE_PROFILES = {
    'TOP': {'totalDamageTaken': 1.0, 'kda': 0.6, 'cs': 0.7},
    'JUNGLE': {'kp': 1.0, 'visionScore': 0.5, 'assists': 0.6},
    'MIDDLE': {'totalDamageDealtToChampions': 1.0, 'kills': 0.7, 'cs': 0.5},
    'BOTTOM': {'totalDamageDealtToChampions': 1.0, 'cs': 1.0, 'kills': 0.6},
    'UTILITY': {'visionScore': 1.0, 'assists': 0.8, 'wardsPlaced': 0.6},
}


def calc_z(val, mean, std):
    if pd.isna(val) or pd.isna(mean) or pd.isna(std) or std == 0:
        return 0.0
    return (val - mean) / std


def calc_fit(member, role):
    """ロール期待プロファイルに対するフィット度 (加重平均 z-score)"""
    md = ps_m[(ps_m['summonerName'] == member) & (ps_m['role'] == role)]
    if len(md) < MIN_GAMES or role not in role_stat_base.index:
        return None

    profile = ROLE_PROFILES.get(role, {})
    if not profile:
        return None

    weighted_z = 0
    total_w = 0
    for stat, weight in profile.items():
        if stat == 'kp':
            md_kp = (md['kills'] + md['assists']) / \
                team_kills.loc[md.index].replace(0, 1)
            val = md_kp.mean()
            m = kp_base.loc[role, 'mean'] if role in kp_base.index else 0.5
            s = kp_base.loc[role, 'std'] if role in kp_base.index else 0.1
        elif stat in role_stat_base.columns.get_level_values(0):
            val = md[stat].mean()
            m = role_stat_base.loc[role, (stat, 'mean')]
            s = role_stat_base.loc[role, (stat, 'std')]
        else:
            continue

        z = calc_z(val, m, s)
        if stat == 'deaths':
            z = -z
        weighted_z += z * weight
        total_w += weight

    return weighted_z / total_w if total_w > 0 else 0.0


fit_signal = {}
for member in MEMBERS:
    for role in ROLES:
        f = calc_fit(member, role)
        if f is not None:
            md = ps_m[(ps_m['summonerName'] == member)
                      & (ps_m['role'] == role)]
            fit_signal[(member, role)] = {'fit': f, 'n': len(md)}

# ═══════════════════════════════════════════════════════════════
#  シグナル 5: 実測パターン実績
# ═══════════════════════════════════════════════════════════════

match_assignments = {}
for mid in match_win.index:
    mp = ps_m[ps_m['matchId'] == mid]
    assign = {}
    for _, p in mp.iterrows():
        if p['summonerName'] in MEMBERS and pd.notna(p['role']):
            assign[p['role']] = p['summonerName']
    ak = tuple(sorted(assign.items()))
    if ak not in match_assignments:
        match_assignments[ak] = []
    match_assignments[ak].append(match_win[mid])

pattern_stats = []
for ak, wins in match_assignments.items():
    if len(wins) < 3:
        continue
    pattern_stats.append({
        'assignment': dict(ak),
        'n': len(wins), 'wr': np.mean(wins) * 100,
    })

obs_signal = {}
for member in MEMBERS:
    for role in ROLES:
        relevant = [p for p in pattern_stats
                    if p['assignment'].get(role) == member]
        if not relevant:
            continue
        total_n = sum(p['n'] for p in relevant)
        if total_n < MIN_GAMES:
            continue
        weighted_wr = sum(p['wr'] * p['n'] for p in relevant) / total_n
        obs_signal[(member, role)] = {
            'wr': weighted_wr, 'n': total_n,
            'patterns': len(relevant),
        }

# ═══════════════════════════════════════════════════════════════
#  シグナル統合 + 合意度計算
# ═══════════════════════════════════════════════════════════════

THRESH_REG = 2.0
THRESH_TEAM = 2.0
THRESH_LANE = 0.08
THRESH_FIT = 0.15
THRESH_OBS_HI = 52.0
THRESH_OBS_LO = 48.0


def classify_signal(val, thresh_pos, thresh_neg=None):
    if thresh_neg is None:
        thresh_neg = -thresh_pos
    if val > thresh_pos:
        return +1
    elif val < thresh_neg:
        return -1
    return 0


composite = {}
for member in MEMBERS:
    for role in ROLES:
        signals = {}
        has_any = False

        # S1: regression
        r = reg_map.get((member, role))
        if r and r['n'] >= MIN_GAMES:
            signals['reg'] = {
                'val': r['pp'], 'n': r['n'],
                'sign': classify_signal(r['pp'], THRESH_REG)}
            has_any = True

        # S2: team WR
        t = member_role_matches.get((member, role))
        if t:
            signals['team'] = {
                'val': t['diff_pp'], 'n': t['n'],
                'sign': classify_signal(t['diff_pp'], THRESH_TEAM)}
            has_any = True

        # S3: lane
        l = lane_signal.get((member, role))
        if l:
            signals['lane'] = {
                'val': l['impact'], 'n': l['n'],
                'sign': classify_signal(l['impact'], THRESH_LANE)}
            has_any = True

        # S4: fit
        f = fit_signal.get((member, role))
        if f:
            signals['fit'] = {
                'val': f['fit'], 'n': f['n'],
                'sign': classify_signal(f['fit'], THRESH_FIT)}
            has_any = True

        # S5: observed
        o = obs_signal.get((member, role))
        if o:
            signals['obs'] = {
                'val': o['wr'], 'n': o['n'],
                'sign': classify_signal(o['wr'], THRESH_OBS_HI, THRESH_OBS_LO)}
            has_any = True

        if not has_any:
            continue

        pos = sum(1 for s in signals.values() if s['sign'] > 0)
        neg = sum(1 for s in signals.values() if s['sign'] < 0)
        total = len(signals)

        if pos >= 4:
            consensus = '◎ 強推奨'
        elif pos >= 3:
            consensus = '○ 推奨'
        elif pos >= 2:
            consensus = '△ 条件付き'
        elif neg >= 3:
            consensus = '▽ 非推奨'
        else:
            consensus = '─ 中立'

        composite[(member, role)] = {
            'signals': signals, 'pos': pos, 'neg': neg,
            'total': total, 'consensus': consensus,
        }

# ═══════════════════════════════════════════════════════════════
#  出力
# ═══════════════════════════════════════════════════════════════

print('=' * 90)
print('  レーン配置 複合判断分析')
print('=' * 90)
print()
print(f'  分析試合数: {len(match_win)}')
print(f'  チーム平均勝率: {overall_wr:.1f}%')
print(f'  最低試合数: {MIN_GAMES}')
print()
print('  5つのシグナル:')
print('    S1 回帰独立効果  — 玉突き分離後の限界効果 (pp)')
print('    S2 チーム勝率影響 — そのロール時のチーム勝率 vs 平均 (pp)')
print('    S3 序盤レーニング — 15分GD z-score × レーン勝敗相関')
print('    S4 プレイヤー適性 — ロール期待値とのフィット度')
print('    S5 実測パターン   — 含む配置パターンの加重平均WR (%)')
print()
print('  合意度: ◎=4-5シグナル推奨  ○=3  △=2  ─=中立  ▽=3以上が非推奨')

# ─── メンバー別 多次元評価テーブル ───
print()
print('=' * 90)
print('  メンバー別 多次元評価')
print('=' * 90)

SIGN_CHAR = {1: '+', -1: '-', 0: '·'}

for member in MEMBERS:
    roles_data = [(r, composite.get((member, r)))
                  for r in ROLES if (member, r) in composite]
    if not roles_data:
        continue

    print()
    print(f'  ■ {member}')
    print()
    print(f'    {"ロール":>6}  {"S1回帰":>10}  {"S2チームWR":>10}  '
          f'{"S3序盤":>8}  {"S4適性":>8}  {"S5実測WR":>10}  '
          f'{"合意度":>10}')
    print(f'    {"-"*6}  {"-"*10}  {"-"*10}  '
          f'{"-"*8}  {"-"*8}  {"-"*10}  {"-"*10}')

    for role, data in sorted(roles_data,
                             key=lambda x: -x[1]['pos'] if x[1] else 0):
        if data is None:
            continue
        sigs = data['signals']

        def fmt_sig(key, fmt_str, suffix=''):
            s = sigs.get(key)
            if s is None:
                return f'{"---":>8}'
            sign = SIGN_CHAR[s['sign']]
            return f'{sign}{fmt_str.format(s["val"])}{suffix}'

        s1 = fmt_sig('reg', '{:+.1f}', 'pp')
        s2 = fmt_sig('team', '{:+.1f}', 'pp')
        s3 = fmt_sig('lane', '{:+.2f}', '')
        s4 = fmt_sig('fit', '{:+.2f}', '')
        s5 = fmt_sig('obs', '{:.1f}', '%')

        print(f'    {ROLE_JP[role]:>6}  {s1:>10}  {s2:>10}  '
              f'{s3:>8}  {s4:>8}  {s5:>10}  {data["consensus"]:>10}')

# ─── ロール別ベストメンバー ───
print()
print('=' * 90)
print('  ロール別 ベストメンバー（合意度順）')
print('=' * 90)
print()

for role in ROLES:
    candidates = [(m, composite[(m, role)])
                  for m in MEMBERS if (m, role) in composite]
    if not candidates:
        print(f'  【{ROLE_JP[role]}】 データ不足')
        continue

    candidates.sort(key=lambda x: (-x[1]['pos'], x[1]['neg']))
    print(f'  【{ROLE_JP[role]}】')
    for m, data in candidates:
        sigs = data['signals']
        detail_parts = []
        if 'reg' in sigs:
            detail_parts.append(f'回帰{sigs["reg"]["val"]:+.1f}pp')
        if 'team' in sigs:
            detail_parts.append(f'チームWR{sigs["team"]["val"]:+.1f}pp')
        if 'obs' in sigs:
            detail_parts.append(f'実測{sigs["obs"]["val"]:.1f}%')
        detail = '  '.join(detail_parts)
        print(f'    {data["consensus"]:>10}  {m:<14}  {detail}')
    print()

# ═══════════════════════════════════════════════════════════════
#  最終サマリー: 合意度が最も高い5人配置
# ═══════════════════════════════════════════════════════════════
print('=' * 90)
print('  総合推奨配置（合意度ベース）')
print('=' * 90)
print()
print('  全5ロールの組み合わせを合意度スコアで評価し、')
print('  最も多くのシグナルが一致する配置を推奨します。')
print()

active = [m for m in MEMBERS if any((m, r) in composite for r in ROLES)]

scored_combos = []
for combo in combinations(active, 5):
    for perm in permutations(range(5)):
        assignment = [(ROLES[perm[i]], combo[i]) for i in range(5)]
        total_pos = 0
        total_neg = 0
        total_signals = 0
        details = []
        ok = True
        for rn, mn in assignment:
            c = composite.get((mn, rn))
            if c is None:
                ok = False
                break
            total_pos += c['pos']
            total_neg += c['neg']
            total_signals += c['total']
            details.append((rn, mn, c))
        if not ok:
            continue
        score = total_pos - total_neg
        scored_combos.append((score, total_pos, total_neg, details))

scored_combos.sort(key=lambda x: (-x[0], -x[1], x[2]))

shown = set()
rank = 0
for score, tp, tn, details in scored_combos:
    key = tuple(sorted((d[0], d[1]) for d in details))
    if key in shown:
        continue
    shown.add(key)
    rank += 1
    if rank > 3:
        break

    print(f'  ─── 第{rank}位 (合意スコア: {score}, '
          f'+{tp} / -{tn}) ───')
    for rn, mn, c in sorted(details, key=lambda x: ROLES.index(x[0])):
        sigs = c['signals']
        detail_parts = []
        if 'reg' in sigs:
            detail_parts.append(f'回帰{sigs["reg"]["val"]:+.1f}pp')
        if 'team' in sigs:
            detail_parts.append(f'チームWR{sigs["team"]["val"]:+.1f}pp')
        if 'obs' in sigs:
            detail_parts.append(f'実測{sigs["obs"]["val"]:.1f}%')
        detail = '  '.join(detail_parts)
        print(f'    {ROLE_JP[rn]:<6} → {mn:<14} {c["consensus"]:>10}  '
              f'{detail}')

    # Conflict check
    conflicts = []
    for rn, mn, c in details:
        sigs = c['signals']
        signs = [s['sign'] for s in sigs.values()]
        if 1 in signs and -1 in signs:
            pos_names = [k for k, s in sigs.items() if s['sign'] > 0]
            neg_names = [k for k, s in sigs.items() if s['sign'] < 0]
            sig_names = {'reg': '回帰', 'team': 'チームWR',
                         'lane': '序盤', 'fit': '適性', 'obs': '実測'}
            pos_jp = [sig_names[n] for n in pos_names]
            neg_jp = [sig_names[n] for n in neg_names]
            conflicts.append(
                f'{mn}@{ROLE_JP[rn]}: '
                f'{",".join(pos_jp)}は推奨 / {",".join(neg_jp)}は非推奨')

    if conflicts:
        print()
        print('    ⚠ シグナル間の矛盾:')
        for cf in conflicts:
            print(f'      {cf}')
    print()

# ─── 注意事項 ───
print('=' * 90)
print('  注意事項')
print('=' * 90)
print()
print('  - 各シグナルは独立に計算されていますが、完全に独立ではありません')
print('    (例: 回帰効果と実測パターンは同じ試合データを使用)')
print('  - 合意度が高いほど複数の視点から一致した評価であり、信頼性が高いです')
print('  - シグナル間に矛盾がある場合、その配置には注意が必要です')
print('  - 回帰効果の詳細は optimal_lane_advanced.py を参照してください')
print()
print('=' * 90)
print('  分析完了')
print('=' * 90)
