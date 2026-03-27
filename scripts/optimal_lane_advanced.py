"""
最適レーン配置分析 — 精度改善版 (Advanced)

optimal_lane.py をベースに以下の統計的改善を追加:
  1. MIN_GAMES=15 (特徴量), MIN_GAMES_SCORE=20 (推奨) に引き上げ
  2. Bootstrap 500回による95%信頼区間
  3. CV AUC・分類精度の報告
  4. 推奨配置の外挿検知・警告
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
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / 'config/settings.yaml', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)
MEMBERS = [m['game_name'] for m in _cfg['members']]

parser = argparse.ArgumentParser()
parser.add_argument('--last', type=int, default=0,
                    help='直近N試合に絞る (0=全試合)')
parser.add_argument('--bootstrap', type=int, default=500,
                    help='Bootstrap繰り返し回数 (default=500)')
parser.add_argument('--combo', type=str, default='',
                    help='特定の5人組に絞る (カンマ区切り, 例: "Member1,Member2,Member3,Member4,Member5")')
args = parser.parse_args()

N_BOOTSTRAP = args.bootstrap

# ─── データ読み込み ───
ps = pd.read_csv(ROOT / 'data/processed/player_stats.csv')
tf = pd.read_csv(ROOT / 'data/processed/timeline_frames.csv')
ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
ROLE_JP = {'TOP': 'トップ', 'JUNGLE': 'JG', 'MIDDLE': 'ミッド',
           'BOTTOM': 'BOT', 'UTILITY': 'SUP'}

ps_m = ps[ps['summonerName'].isin(MEMBERS)].copy()
ps_m = ps_m[ps_m['role'].notna()]

if args.combo:
    combo_members = set(m.strip() for m in args.combo.split(','))
    combo_match_ids = (ps_m.groupby('matchId')['summonerName']
                       .apply(lambda x: set(x.unique()))
                       .reset_index())
    combo_match_ids = combo_match_ids[
        combo_match_ids['summonerName'].apply(
            lambda s: s == combo_members)
    ]['matchId'].values
    ps_m = ps_m[ps_m['matchId'].isin(combo_match_ids)]
    tf = tf[tf['matchId'].isin(combo_match_ids)]
    MEMBERS = sorted(combo_members)

if args.last > 0:
    recent_matches = (ps_m.groupby('matchId')['matchId'].first()
                      .sort_index().tail(args.last).index)
    ps_m = ps_m[ps_m['matchId'].isin(recent_matches)]
    tf = tf[tf['matchId'].isin(recent_matches)]

match_win = ps_m.groupby('matchId')['win'].first()

MIN_GAMES = 15
MIN_GAMES_SCORE = 20

# ═══════════════════════════════════════════════════════════════
#  STEP 1: 特徴量構築 + ロジスティック回帰
# ═══════════════════════════════════════════════════════════════

feature_rows = []
for mid in match_win.index:
    row = {}
    match_ps = ps_m[ps_m['matchId'] == mid]
    for _, p in match_ps.iterrows():
        key = f'{p["summonerName"]}@{p["role"]}'
        row[key] = 1
    feature_rows.append(row)

feat_df = pd.DataFrame(feature_rows, index=match_win.index).fillna(0)
valid_cols = [c for c in feat_df.columns if feat_df[c].sum() >= MIN_GAMES]
feat_df = feat_df[valid_cols]
valid_cols_list = list(valid_cols)

X = feat_df.values.astype(float)
y = match_win.values.astype(float)

Cs = np.logspace(-2, 2, 30)
logit = LogisticRegressionCV(Cs=Cs, cv=5, solver='lbfgs',
                             max_iter=5000, random_state=42)
logit.fit(X, y)

intercept = logit.intercept_[0]
best_C = logit.C_[0]

raw_coeffs = logit.coef_[0]
marginal_effects = raw_coeffs * 0.25 * 100

# ═══════════════════════════════════════════════════════════════
#  STEP 1b: CV AUC・精度
# ═══════════════════════════════════════════════════════════════

cv_proba = cross_val_predict(
    LogisticRegression(C=best_C, solver='lbfgs', max_iter=5000, random_state=42),
    X, y, cv=5, method='predict_proba',
)
cv_auc = roc_auc_score(y, cv_proba[:, 1])
cv_pred = (cv_proba[:, 1] >= 0.5).astype(float)
cv_acc = accuracy_score(y, cv_pred)

# ═══════════════════════════════════════════════════════════════
#  STEP 1c: Bootstrap 信頼区間
# ═══════════════════════════════════════════════════════════════

print(f'  Bootstrap {N_BOOTSTRAP}回 実行中...', end='', flush=True)

rng = np.random.RandomState(42)
n_matches = len(match_win)
boot_coeffs = np.zeros((N_BOOTSTRAP, len(valid_cols_list)))
boot_intercepts = np.zeros(N_BOOTSTRAP)

for b in range(N_BOOTSTRAP):
    boot_idx = rng.choice(n_matches, size=n_matches, replace=True)
    X_boot = X[boot_idx]
    y_boot = y[boot_idx]
    if len(np.unique(y_boot)) < 2:
        boot_coeffs[b] = np.nan
        boot_intercepts[b] = np.nan
        continue
    try:
        m = LogisticRegression(C=best_C, solver='lbfgs',
                               max_iter=5000, random_state=b)
        m.fit(X_boot, y_boot)
        boot_coeffs[b] = m.coef_[0]
        boot_intercepts[b] = m.intercept_[0]
    except Exception:
        boot_coeffs[b] = np.nan
        boot_intercepts[b] = np.nan

print(' 完了')

boot_marginal = boot_coeffs * 0.25 * 100
boot_ci_low = np.nanpercentile(boot_marginal, 2.5, axis=0)
boot_ci_high = np.nanpercentile(boot_marginal, 97.5, axis=0)

# ─── reg_effects DataFrame ───
reg_effects = pd.DataFrame({
    'feature': valid_cols_list,
    'log_coeff': raw_coeffs,
    'marginal_pp': marginal_effects,
    'ci_low': boot_ci_low,
    'ci_high': boot_ci_high,
    'games': [int(feat_df[c].sum()) for c in valid_cols_list],
})
reg_effects['member'] = reg_effects['feature'].str.split('@').str[0]
reg_effects['role'] = reg_effects['feature'].str.split('@').str[1]
reg_effects = reg_effects[reg_effects['member'].isin(MEMBERS)].copy()

reg_effects['simple_wr'] = reg_effects.apply(
    lambda r: ps_m[(ps_m['summonerName'] == r['member'])
                    & (ps_m['role'] == r['role'])]['win'].mean() * 100,
    axis=1,
)
reg_effects['gap'] = reg_effects['marginal_pp'] - (reg_effects['simple_wr'] - 50)

# ═══════════════════════════════════════════════════════════════
#  STEP 2: 条件付き勝率 — 実際の配置パターン
# ═══════════════════════════════════════════════════════════════

match_assignments = {}
for mid in match_win.index:
    match_ps = ps_m[ps_m['matchId'] == mid]
    assign = {}
    for _, p in match_ps.iterrows():
        if p['summonerName'] in MEMBERS and pd.notna(p['role']):
            assign[p['role']] = p['summonerName']
    assign_key = tuple(sorted(assign.items()))
    if assign_key not in match_assignments:
        match_assignments[assign_key] = []
    match_assignments[assign_key].append(match_win[mid])

pattern_stats = []
for assign_key, wins in match_assignments.items():
    if len(wins) < 3:
        continue
    assign_dict = dict(assign_key)
    n = len(wins)
    wr = np.mean(wins) * 100
    pattern_stats.append({
        'assignment': assign_dict,
        'n': n, 'wr': wr, 'wins': sum(wins), 'losses': n - sum(wins),
    })
pattern_stats.sort(key=lambda x: -x['n'])

# ═══════════════════════════════════════════════════════════════
#  STEP 3: 15分ゴールド差 z-score
# ═══════════════════════════════════════════════════════════════

tf15 = tf[(tf['timestampMin'] == 15) & (tf['summonerName'].isin(MEMBERS))]
role_gd_baseline = (tf[tf['timestampMin'] == 15]
                    .groupby('role')['goldDiffVsOpponent']
                    .agg(['mean', 'std']))


def get_lane_z(member, role):
    m_tf = tf15[(tf15['summonerName'] == member) & (tf15['role'] == role)]
    if len(m_tf) < 3 or role not in role_gd_baseline.index:
        return np.nan
    gd = m_tf['goldDiffVsOpponent'].mean()
    baseline = role_gd_baseline.loc[role, 'mean']
    std = role_gd_baseline.loc[role, 'std']
    return (gd - baseline) / std if std > 0 else 0


reg_effects['lane_z'] = reg_effects.apply(
    lambda r: get_lane_z(r['member'], r['role']), axis=1)


# ═══════════════════════════════════════════════════════════════
#  ヘルパー関数
# ═══════════════════════════════════════════════════════════════

def find_observed_pattern(details):
    target = {role: name for role, name, *_ in details}
    target_key = tuple(sorted(target.items()))
    for p in pattern_stats:
        if tuple(sorted(p['assignment'].items())) == target_key:
            return p
    return None


def get_bootstrap_pred_ci(details):
    x_vec = np.zeros(len(valid_cols_list))
    for role_name, member_name, *_ in details:
        feat = f'{member_name}@{role_name}'
        if feat in valid_cols_list:
            x_vec[valid_cols_list.index(feat)] = 1

    probs = []
    for b in range(N_BOOTSTRAP):
        if np.isnan(boot_intercepts[b]):
            continue
        logit_val = boot_intercepts[b] + np.dot(boot_coeffs[b], x_vec)
        probs.append(1 / (1 + np.exp(-logit_val)) * 100)

    if len(probs) < 10:
        return None, None
    return np.percentile(probs, 2.5), np.percentile(probs, 97.5)


def get_score(member, role):
    row = reg_effects[(reg_effects['member'] == member)
                      & (reg_effects['role'] == role)]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    if r['games'] < MIN_GAMES_SCORE:
        return None
    reliability = min(np.sqrt(r['games']) / np.sqrt(MIN_GAMES_SCORE), 1.0)
    return {
        'marginal': r['marginal_pp'], 'ci_low': r['ci_low'],
        'ci_high': r['ci_high'], 'log_coeff': r['log_coeff'],
        'simple_wr': r['simple_wr'], 'n': int(r['games']),
        'reliability': reliability,
        'score': r['marginal_pp'] * reliability,
        'lane_z': r['lane_z'], 'gap': r['gap'],
    }


# ═══════════════════════════════════════════════════════════════
#  出力
# ═══════════════════════════════════════════════════════════════

print('=' * 82)
print('  最適レーン配置分析 — 精度改善版 (Advanced)')
print('=' * 82)
print()
print(f'  分析試合数: {len(match_win)}')
print(f'  特徴量の最低試合数: {MIN_GAMES}')
print(f'  推奨配置の最低試合数: {MIN_GAMES_SCORE}')
print(f'  有効特徴量数: {len(valid_cols_list)}')
print(f'  ロジスティック回帰 C: {best_C:.2f} (5-fold CV自動選択)')
base_wr = 1 / (1 + np.exp(-intercept)) * 100
print(f'  切片 (ロジット): {intercept:.3f}  →  ベースライン勝率: {base_wr:.1f}%')
print()
print('  ─── モデル信頼度 ───')
auc_comment = ('⚠ モデルの識別力が低い（ランダムに近い）' if cv_auc < 0.55
               else '△ 識別力は限定的' if cv_auc < 0.65
               else '○ 一定の識別力あり')
print(f'  CV AUC: {cv_auc:.3f}  {auc_comment}')
print(f'  CV 分類精度: {cv_acc * 100:.1f}%')
print(f'  Bootstrap: {N_BOOTSTRAP}回')
print()

# ─── PART 1 ───
print('=' * 82)
print('  PART 1: 独立効果マトリクス（限界効果 pp + 95%CI）')
print('=' * 82)
print()
header = f'  {"":>14}'
for r in ROLES:
    header += f'  {ROLE_JP[r]:>18}'
header += '   ベスト'
print(header)
print('  ' + '-' * 112)

for member in MEMBERS:
    me = reg_effects[reg_effects['member'] == member]
    print(f'  {member:>14}', end='')
    best_val, best_role = -999, ''
    for role in ROLES:
        row = me[me['role'] == role]
        if len(row) > 0:
            c = row.iloc[0]['marginal_pp']
            ci_l = row.iloc[0]['ci_low']
            ci_h = row.iloc[0]['ci_high']
            n = int(row.iloc[0]['games'])
            print(f' {c:>+5.1f}[{ci_l:>+5.1f},{ci_h:>+5.1f}]', end='')
            if c > best_val and n >= MIN_GAMES:
                best_val, best_role = c, role
        else:
            print(f'  {"---":>18}', end='')
    print(f'   {ROLE_JP[best_role]}' if best_role else '')

print()
print(f'  凡例: 値=限界効果(pp)  [低,高]=Bootstrap 95%信頼区間')
print(f'  ※ 特徴量最低試合数 {MIN_GAMES}試合未満は除外')
ci_widths = reg_effects['ci_high'] - reg_effects['ci_low']
print(f'  ※ CI幅の中央値: {ci_widths.median():.1f}pp  (小さいほど推定が安定)')

# ─── PART 2 ───
print()
print('=' * 82)
print('  PART 2: 単純勝率 vs 独立効果 — 玉突きの可視化')
print('=' * 82)
print()
print('  差がプラス → 単純勝率が過小評価（他メンバーの悪配置の巻き添え）')
print('  差がマイナス → 単純勝率が過大評価（他メンバーの好配置のおかげ）')
print()

for member in MEMBERS:
    me = reg_effects[(reg_effects['member'] == member)
                     & (reg_effects['games'] >= MIN_GAMES)].copy()
    if len(me) == 0:
        continue
    me = me.sort_values('marginal_pp', ascending=False)
    print(f'  ■ {member}')
    for _, r in me.iterrows():
        arrow = ('↑過小評価' if r['gap'] > 3
                 else '↓過大評価' if r['gap'] < -3
                 else '≈一致')
        ci_s = f'[{r["ci_low"]:+.1f},{r["ci_high"]:+.1f}]'
        lz = f'lane-z={r["lane_z"]:+.2f}' if not np.isnan(r['lane_z']) else ''
        print(f'    {ROLE_JP[r["role"]]:>6}: 独立{r["marginal_pp"]:>+5.1f}pp {ci_s:>14}  '
              f'単純WR{r["simple_wr"]:>5.1f}%  差{r["gap"]:>+5.1f}pp '
              f'{arrow:>6}  {lz}  ({int(r["games"])}試合)')
    print()

# ─── PART 3 ───
print('=' * 82)
print('  PART 3: 実際に観測された配置パターンの勝率')
print('=' * 82)
print()
print('  ※ 実際にその5人配置でプレイした試合の勝率（最も信頼できるデータ）')
print()

for i, p in enumerate(pattern_stats[:10], 1):
    a = p['assignment']
    parts = [f'{ROLE_JP[r]}={a[r]}' for r in ROLES if r in a]
    print(f'  {i:>2}. [{p["wr"]:>5.1f}% / {p["n"]:>2}試合 / '
          f'{p["wins"]}勝{p["losses"]}敗]  {" / ".join(parts)}')

if not pattern_stats:
    print('  3試合以上の同一配置パターンなし')

# ─── PART 4 ───
print()
print('=' * 82)
print('  PART 4: 玉突きパターン — 主要な押し出し関係')
print('=' * 82)
print()

member_roles_data = ps_m[['matchId', 'summonerName', 'role']]
displacements = []

for member in MEMBERS:
    m_matches = member_roles_data[member_roles_data['summonerName'] == member]
    for role in ROLES:
        m_role_ids = m_matches[m_matches['role'] == role]['matchId'].values
        if len(m_role_ids) < MIN_GAMES:
            continue
        others = member_roles_data[
            (member_roles_data['matchId'].isin(m_role_ids))
            & (member_roles_data['summonerName'] != member)
        ]
        for other in MEMBERS:
            if other == member:
                continue
            od = others[others['summonerName'] == other]
            if len(od) < 3:
                continue
            local_dist = od['role'].value_counts(normalize=True)
            global_dist = (member_roles_data[member_roles_data['summonerName'] == other]
                           ['role'].value_counts(normalize=True))
            for or_ in ROLES:
                loc = local_dist.get(or_, 0) * 100
                glb = global_dist.get(or_, 0) * 100
                shift = loc - glb
                if abs(shift) < 15:
                    continue
                erow = reg_effects[(reg_effects['member'] == other)
                                   & (reg_effects['role'] == or_)]
                eff = erow.iloc[0]['marginal_pp'] if len(erow) > 0 else None
                displacements.append({
                    'trigger': member, 't_role': role,
                    'affected': other, 'a_role': or_,
                    'local': loc, 'global': glb,
                    'shift': shift, 'eff': eff, 'n': len(od),
                })

if displacements:
    disp_df = pd.DataFrame(displacements).sort_values(
        'shift', key=abs, ascending=False)
    for _, d in disp_df.head(12).iterrows():
        direction = '↑押込' if d['shift'] > 0 else '↓追出'
        eff_str = (f'独立効果{d["eff"]:+.1f}pp'
                   if d['eff'] is not None and not np.isnan(d['eff'])
                   else '')
        impact = ''
        if d['eff'] is not None and not np.isnan(d['eff']):
            if d['shift'] > 0 and d['eff'] < -2:
                impact = ' ⚠ 悪影響'
            elif d['shift'] > 0 and d['eff'] > 2:
                impact = ' ✓ 好影響'
            elif d['shift'] < 0 and d['eff'] > 2:
                impact = ' ⚠ 損失'
        print(f'  {d["trigger"]}→{ROLE_JP[d["t_role"]]} のとき '
              f'{d["affected"]}の{ROLE_JP[d["a_role"]]}率: '
              f'{d["global"]:.0f}%→{d["local"]:.0f}% '
              f'({d["shift"]:+.0f}%) {direction}  [{eff_str}]{impact}')

# ─── PART 5 ───
print()
print()
print('=' * 82)
print('  PART 5: 最適5人配置（総合スコアリング + 信頼区間 + 外挿検知）')
print('=' * 82)
print()
print('  スコアリング方式:')
print('    総合スコア = Σ(独立効果_i × 信頼度_i)')
print(f'    信頼度 = min(sqrt(試合数) / sqrt({MIN_GAMES_SCORE}), 1.0)')
print(f'    ※ 推奨の最低試合数: {MIN_GAMES_SCORE}試合')
print()

active = [m for m in MEMBERS if m in reg_effects['member'].unique()]
print(f'  対象: {", ".join(active)} ({len(active)}人)')
print()

best_combos = []
for combo in combinations(active, 5):
    for perm in permutations(range(5)):
        assignment = [(ROLES[perm[i]], combo[i]) for i in range(5)]
        total_score = 0
        details = []
        ok = True
        for rn, mn in assignment:
            s = get_score(mn, rn)
            if s is None:
                ok = False
                break
            total_score += s['score']
            details.append((rn, mn, s))
        if ok:
            best_combos.append((total_score, details))

best_combos.sort(key=lambda x: -x[0])

shown = set()
rank = 0
for total_score, details in best_combos:
    key = tuple(sorted((d[0], d[1]) for d in details))
    if key in shown:
        continue
    shown.add(key)
    rank += 1
    if rank > 5:
        break

    avg_wr = np.mean([d[2]['simple_wr'] for d in details])
    x_vec = np.zeros(len(valid_cols_list))
    for rn, mn, s in details:
        feat = f'{mn}@{rn}'
        if feat in valid_cols_list:
            x_vec[valid_cols_list.index(feat)] = 1
    pred_prob = logit.predict_proba(x_vec.reshape(1, -1))[0][1] * 100
    ci_l, ci_h = get_bootstrap_pred_ci(details)
    ci_str = f'[{ci_l:.1f}%, {ci_h:.1f}%]' if ci_l is not None else '[算出不可]'

    obs = find_observed_pattern(details)
    if obs:
        obs_str = f'実測 {obs["wr"]:.1f}% ({obs["n"]}試合)'
        extrap = ''
    else:
        obs_str = '未観測'
        extrap = ' ⚠ 外挿注意'

    print(f'  ─── 第{rank}位 (総合スコア: {total_score:+.1f}, '
          f'予測勝率: {pred_prob:.1f}% {ci_str}) ───')
    print(f'       観測: {obs_str}{extrap}')

    for rn, mn, s in sorted(details, key=lambda x: ROLES.index(x[0])):
        ci_m = f'[{s["ci_low"]:+.1f},{s["ci_high"]:+.1f}]'
        lz = ('序盤◎' if not np.isnan(s['lane_z']) and s['lane_z'] > 0.3
              else '序盤✕' if not np.isnan(s['lane_z']) and s['lane_z'] < -0.3
              else '')
        print(f'    {ROLE_JP[rn]:<6} → {mn:<14} '
              f'独立{s["marginal"]:>+5.1f}pp {ci_m:>14}  '
              f'WR{s["simple_wr"]:>5.1f}%  '
              f'信頼{s["reliability"]*100:>3.0f}%  ({s["n"]}試合) {lz}')
    print(f'    → 予測勝率: {pred_prob:.1f}% {ci_str}  '
          f'(単純平均WR: {avg_wr:.1f}%)')
    print()

# ─── PART 6 ───
print('=' * 82)
print('  PART 6: 単純勝率ベース vs 回帰ベース — 推奨の違い')
print('=' * 82)
print()

simple_combos = []
for combo in combinations(active, 5):
    for perm in permutations(range(5)):
        assignment = [(ROLES[perm[i]], combo[i]) for i in range(5)]
        total = 0
        det = []
        ok = True
        for rn, mn in assignment:
            md = ps_m[(ps_m['summonerName'] == mn) & (ps_m['role'] == rn)]
            n = len(md)
            if n < MIN_GAMES_SCORE:
                ok = False
                break
            wr = md['win'].mean() * 100
            total += wr * np.log(n + 1)
            det.append((rn, mn, wr, n))
        if ok:
            simple_combos.append((total, det))

simple_combos.sort(key=lambda x: -x[0])

if simple_combos and best_combos:
    s_top = {d[0]: (d[1], d[2], d[3]) for d in simple_combos[0][1]}
    r_top = {d[0]: (d[1], d[2]) for d in best_combos[0][1]}

    print(f'  {"ロール":>6}  {"単純勝率ベース":^28}  '
          f'{"回帰ベース":^28}  {"比較"}')
    print('  ' + '-' * 84)

    diffs = []
    for role in ROLES:
        sn, sw, snn = s_top.get(role, ('---', 0, 0))
        rn, rs = r_top.get(role, ('---', None))
        rpp = rs['marginal'] if rs else 0
        rnn = rs['n'] if rs else 0
        differs = sn != rn
        mark = '◀ 異なる' if differs else ''
        if differs:
            diffs.append(role)
        print(f'  {ROLE_JP[role]:>6}  {sn:<14} WR{sw:>5.1f}%({snn:>3})  '
              f'{rn:<14} {rpp:>+5.1f}pp({rnn:>3})  {mark}')

    print()
    if diffs:
        print('  → 配置が異なるポジションがあります。')
        print()
        for role in diffs:
            sn = s_top[role][0]
            rn = r_top[role][0]
            print(f'    {ROLE_JP[role]}: 単純勝率では{sn}だが、'
                  f'回帰分析では{rn}を推奨')
            se = reg_effects[(reg_effects['member'] == sn)
                             & (reg_effects['role'] == role)]
            if len(se) > 0 and se.iloc[0]['gap'] < -3:
                print(f'      理由: {sn}の{ROLE_JP[role]}勝率'
                      f'{se.iloc[0]["simple_wr"]:.1f}%は他メンバーの好配置の'
                      f'おかげで膨らんでいた(差{se.iloc[0]["gap"]:+.1f}pp)')
    else:
        print('  → 両手法で同じ配置が推奨されました。')

# ─── PART 7 ───
print()
print()
print('=' * 82)
print('  PART 7: 最終推奨配置 + 信頼度サマリー')
print('=' * 82)
print()

if best_combos:
    top_details = best_combos[0][1]

    x_vec = np.zeros(len(valid_cols_list))
    for rn, mn, s in top_details:
        feat = f'{mn}@{rn}'
        if feat in valid_cols_list:
            x_vec[valid_cols_list.index(feat)] = 1
    pred_prob = logit.predict_proba(x_vec.reshape(1, -1))[0][1] * 100
    ci_l, ci_h = get_bootstrap_pred_ci(top_details)
    avg_simple = np.mean([d[2]['simple_wr'] for d in top_details])
    obs = find_observed_pattern(top_details)

    print('  ★ 回帰分析に基づく最適配置:')
    print('    （これは回帰モデルの1つの視点です。')
    print('     複合判断には lane_composite_judgment.py を併用してください）')
    print()

    for rn, mn, s in sorted(top_details, key=lambda x: ROLES.index(x[0])):
        lz = s['lane_z']
        lz_s = ('序盤◎' if not np.isnan(lz) and lz > 0.3
                else '序盤✕' if not np.isnan(lz) and lz < -0.3
                else '序盤並' if not np.isnan(lz) else '')
        ci_m = f'[{s["ci_low"]:+.1f},{s["ci_high"]:+.1f}]'
        print(f'    {ROLE_JP[rn]:<6} → {mn}')
        print(f'           独立効果{s["marginal"]:>+5.1f}pp {ci_m} / '
              f'WR{s["simple_wr"]:.1f}% / {s["n"]}試合 / '
              f'信頼{s["reliability"]*100:.0f}% {lz_s}')

    print()
    ci_str = (f'[{ci_l:.1f}%, {ci_h:.1f}%]' if ci_l is not None else '')
    print(f'  予測勝率: {pred_prob:.1f}% {ci_str}')
    print(f'  単純勝率平均: {avg_simple:.1f}%')
    if obs:
        print(f'  実測データ: {obs["wr"]:.1f}% ({obs["n"]}試合)')
    else:
        print(f'  実測データ: この配置パターンは未観測 ⚠')

    # 信頼度サマリー
    print()
    print('  ─── 信頼度サマリー ───')
    print()
    print(f'    モデル全体:')
    print(f'      CV AUC = {cv_auc:.3f}  {auc_comment}')
    print(f'      CV 分類精度 = {cv_acc * 100:.1f}%')
    print()
    print(f'    予測の不確実性:')
    if ci_l is not None:
        ci_w = ci_h - ci_l
        ci_comment = ('非常に広い — 予測の信頼性が低い' if ci_w > 30
                      else '広い — 参考程度' if ci_w > 20
                      else '比較的安定')
        print(f'      予測勝率の95%CI幅 = {ci_w:.1f}pp  ({ci_comment})')
    print()
    print(f'    データ充足度:')
    min_n = min(d[2]['n'] for d in top_details)
    max_n = max(d[2]['n'] for d in top_details)
    print(f'      推奨メンバーの試合数: {min_n}〜{max_n}試合')
    if min_n < 30:
        print(f'      ⚠ 最少{min_n}試合のメンバーあり — 推定が不安定な可能性')

    # サブ
    assigned = {d[1] for d in top_details}
    reg_assign = {d[0]: d[1] for d in top_details}
    bench = [m for m in active if m not in assigned]
    if bench:
        print()
        print('  サブメンバー:')
        for m in bench:
            m_eff = reg_effects[(reg_effects['member'] == m)
                                & (reg_effects['games'] >= MIN_GAMES)]
            if len(m_eff) == 0:
                continue
            best = m_eff.sort_values('marginal_pp', ascending=False).iloc[0]
            replacing = reg_assign.get(best['role'], '?')
            ci_sub = f'[{best["ci_low"]:+.1f},{best["ci_high"]:+.1f}]'
            print(f'    {m}: ベスト={ROLE_JP[best["role"]]} '
                  f'(独立{best["marginal_pp"]:+.1f}pp {ci_sub}, '
                  f'WR{best["simple_wr"]:.1f}%, {int(best["games"])}試合)')
            print(f'         {replacing} と交代するパターン')

    # 玉突き注目
    print()
    print('  ─── 玉突き効果で評価が大きく変わったケース ───')
    print()
    big_gap = reg_effects[
        (reg_effects['games'] >= MIN_GAMES)
        & (reg_effects['gap'].abs() > 5)
    ].sort_values('gap', key=abs, ascending=False)
    for _, r in big_gap.head(6).iterrows():
        reason = ('他メンバーの悪配置のせいで単純WRが低く見えていた'
                  if r['gap'] > 0
                  else '他メンバーの好配置のおかげで単純WRが高く見えていた')
        print(f'    {r["member"]} {ROLE_JP[r["role"]]}: '
              f'WR{r["simple_wr"]:.1f}% → 独立{r["marginal_pp"]:+.1f}pp  '
              f'(差{r["gap"]:+.1f}pp: {reason})')

print()
print('=' * 82)
print('  分析完了')
print('=' * 82)
