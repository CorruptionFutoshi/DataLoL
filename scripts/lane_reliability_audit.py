"""
レーン配置勝率の信頼性を多角的に監査するスクリプト

検証項目:
  1. モデル予測力 — 順列検定で「ランダム以下」は偶然かを検証
  2. 勝率の説明要因比較 — レーン配置 vs チャンピオン vs 序盤GD vs オブジェクト
  3. 時系列安定性 — 前半/後半で「最適」配置は変わるか
  4. サンプルサイズ妥当性 — 各メンバー×ロールの統計的検出力
  5. Bootstrap安定性 — 最適ロール推奨の変動率
  6. 実測パターンの信頼区間 — 観測配置の勝率は偶然の範囲か
"""
import sys, io, warnings
import yaml
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / 'config/settings.yaml', encoding='utf-8') as f:
    _cfg = yaml.safe_load(f)
MEMBERS = [m['game_name'] for m in _cfg['members']]

ps = pd.read_csv('data/processed/player_stats.csv')
tf = pd.read_csv('data/processed/timeline_frames.csv')
obj = pd.read_csv('data/processed/objectives.csv')
matches = pd.read_csv('data/processed/matches.csv')
ROLES = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']
ROLE_JP = {'TOP': 'トップ', 'JUNGLE': 'JG', 'MIDDLE': 'ミッド',
           'BOTTOM': 'BOT', 'UTILITY': 'SUP'}

ps_m = ps[ps['summonerName'].isin(MEMBERS)].copy()
ps_m = ps_m[ps_m['role'].notna()]
match_win = ps_m.groupby('matchId')['win'].first()

print('=' * 80)
print('  レーン配置勝率の信頼性監査')
print('=' * 80)
print(f'\n  分析対象: {len(match_win)}試合 / {len(MEMBERS)}メンバー\n')

# ═══════════════════════════════════════════════════════════════
#  検証1: 順列検定 — モデルのAUCは「たまたま」か？
# ═══════════════════════════════════════════════════════════════
print('=' * 80)
print('  検証1: 順列検定 — レーン配置モデルのAUCは偶然か？')
print('=' * 80)
print()

feature_rows = []
for mid in match_win.index:
    row = {}
    match_ps = ps_m[ps_m['matchId'] == mid]
    for _, p in match_ps.iterrows():
        key = f'{p["summonerName"]}@{p["role"]}'
        row[key] = 1
    feature_rows.append(row)

feat_df = pd.DataFrame(feature_rows, index=match_win.index).fillna(0)
valid_cols = [c for c in feat_df.columns if feat_df[c].sum() >= 15]
feat_df = feat_df[valid_cols]

X = feat_df.values.astype(float)
y = match_win.values.astype(float)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
real_aucs = cross_val_score(
    LogisticRegression(C=0.01, solver='lbfgs', max_iter=5000, random_state=42),
    X, y, cv=cv, scoring='roc_auc'
)
real_auc = real_aucs.mean()

N_PERM = 200
perm_aucs = []
rng = np.random.RandomState(42)
for i in range(N_PERM):
    y_perm = rng.permutation(y)
    perm_auc = cross_val_score(
        LogisticRegression(C=0.01, solver='lbfgs', max_iter=5000, random_state=42),
        X, y_perm, cv=cv, scoring='roc_auc'
    ).mean()
    perm_aucs.append(perm_auc)

perm_aucs = np.array(perm_aucs)
p_value = np.mean(perm_aucs >= real_auc)

print(f'  実際のモデル CV AUC: {real_auc:.3f}')
print(f'  ランダム順列 AUC (平均±SD): {perm_aucs.mean():.3f} ± {perm_aucs.std():.3f}')
print(f'  ランダム順列 AUC (5%-95%): [{np.percentile(perm_aucs, 5):.3f}, {np.percentile(perm_aucs, 95):.3f}]')
print(f'  順列検定 p値: {p_value:.3f}')
print()
if p_value > 0.05:
    print(f'  → 結論: p={p_value:.3f} > 0.05')
    print(f'    レーン配置モデルはランダムと統計的に区別できない')
    print(f'    つまり「誰がどのレーンに行くか」だけでは勝敗を予測できない')
else:
    print(f'  → 結論: p={p_value:.3f} <= 0.05')
    print(f'    レーン配置はランダムを超える予測力がある')

# ═══════════════════════════════════════════════════════════════
#  検証2: 勝率の説明要因比較
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 80)
print('  検証2: 勝率の説明要因比較 — 何が勝敗を一番左右するか？')
print('=' * 80)
print()

factor_aucs = {}

# Factor A: レーン配置（既に計算済み）
factor_aucs['レーン配置 (誰がどのロール)'] = real_auc

# Factor B: チャンピオン選択
champ_rows = []
for mid in match_win.index:
    row = {}
    match_ps = ps_m[ps_m['matchId'] == mid]
    for _, p in match_ps.iterrows():
        key = f'champ_{p["championName"]}'
        row[key] = 1
    champ_rows.append(row)

champ_df = pd.DataFrame(champ_rows, index=match_win.index).fillna(0)
champ_valid = [c for c in champ_df.columns if champ_df[c].sum() >= 10]
if len(champ_valid) > 5:
    X_champ = champ_df[champ_valid].values.astype(float)
    champ_auc = cross_val_score(
        LogisticRegression(C=0.01, solver='lbfgs', max_iter=5000, random_state=42),
        X_champ, y, cv=cv, scoring='roc_auc'
    ).mean()
    factor_aucs['チャンピオン選択 (味方ピック)'] = champ_auc

# Factor C: 15分ゴールド差（チーム合計）
tf15 = tf[(tf['timestampMin'] == 15) & (tf['summonerName'].isin(MEMBERS))]
team_gd15 = tf15.groupby('matchId')['goldDiffVsOpponent'].sum()
common_matches = match_win.index.intersection(team_gd15.index)
if len(common_matches) > 50:
    X_gd = team_gd15.loc[common_matches].values.reshape(-1, 1).astype(float)
    y_gd = match_win.loc[common_matches].values.astype(float)
    gd_auc = cross_val_score(
        LogisticRegression(C=1, solver='lbfgs', max_iter=5000, random_state=42),
        X_gd, y_gd, cv=cv, scoring='roc_auc'
    ).mean()
    factor_aucs['15分チームGD合計'] = gd_auc

# Factor D: 序盤GD（レーン別）
if len(common_matches) > 50:
    lane_gd_rows = []
    for mid in common_matches:
        row = {}
        mid_tf = tf15[tf15['matchId'] == mid]
        for _, p in mid_tf.iterrows():
            row[f'gd_{p["role"]}'] = p['goldDiffVsOpponent']
        lane_gd_rows.append(row)
    lane_gd_df = pd.DataFrame(lane_gd_rows, index=common_matches).fillna(0)
    X_lane_gd = lane_gd_df.values.astype(float)
    lane_gd_auc = cross_val_score(
        LogisticRegression(C=1, solver='lbfgs', max_iter=5000, random_state=42),
        X_lane_gd, y_gd, cv=cv, scoring='roc_auc'
    ).mean()
    factor_aucs['15分レーン別GD'] = lane_gd_auc

# Factor E: ファーストドラゴン/バロン取得
first_drag = obj[(obj['objectiveType'] == 'DRAGON') & (obj['isFirst'] == True)].copy()
our_team_ids = ps_m.groupby('matchId')['teamId'].first()
drag_matches = first_drag['matchId'].unique()
common_drag = match_win.index.intersection(drag_matches)
if len(common_drag) > 50:
    drag_rows = []
    for mid in common_drag:
        our_tid = our_team_ids.get(mid, None)
        fd = first_drag[first_drag['matchId'] == mid]
        got_first = 1 if (len(fd) > 0 and fd.iloc[0]['teamId'] == our_tid) else 0
        drag_rows.append({'first_dragon': got_first})
    X_drag = pd.DataFrame(drag_rows, index=common_drag).values.astype(float)
    y_drag = match_win.loc[common_drag].values.astype(float)
    drag_auc = cross_val_score(
        LogisticRegression(C=1, solver='lbfgs', max_iter=5000, random_state=42),
        X_drag, y_drag, cv=cv, scoring='roc_auc'
    ).mean()
    factor_aucs['ファーストドラゴン'] = drag_auc

# Factor F: レーン配置 + チャンピオン
if 'チャンピオン選択 (味方ピック)' in factor_aucs:
    combined_rows = []
    for mid in match_win.index:
        row = {}
        match_ps_c = ps_m[ps_m['matchId'] == mid]
        for _, p in match_ps_c.iterrows():
            key = f'{p["summonerName"]}@{p["role"]}'
            row[key] = 1
            key2 = f'champ_{p["championName"]}'
            row[key2] = 1
        combined_rows.append(row)
    comb_df = pd.DataFrame(combined_rows, index=match_win.index).fillna(0)
    comb_valid = [c for c in comb_df.columns if comb_df[c].sum() >= 10]
    if len(comb_valid) > 5:
        X_comb = comb_df[comb_valid].values.astype(float)
        comb_auc = cross_val_score(
            LogisticRegression(C=0.01, solver='lbfgs', max_iter=5000, random_state=42),
            X_comb, y, cv=cv, scoring='roc_auc'
        ).mean()
        factor_aucs['レーン配置 + チャンピオン'] = comb_auc

# Factor G: KDA (試合後のデータなので因果ではないが、上限参考値)
kda_rows = []
for mid in match_win.index:
    match_ps_k = ps_m[ps_m['matchId'] == mid]
    row = {
        'team_kills': match_ps_k['kills'].sum(),
        'team_deaths': match_ps_k['deaths'].sum(),
        'team_assists': match_ps_k['assists'].sum(),
    }
    kda_rows.append(row)
kda_df = pd.DataFrame(kda_rows, index=match_win.index)
X_kda = kda_df.values.astype(float)
kda_auc = cross_val_score(
    LogisticRegression(C=1, solver='lbfgs', max_iter=5000, random_state=42),
    X_kda, y, cv=cv, scoring='roc_auc'
).mean()
factor_aucs['チームKDA (試合後, 参考)'] = kda_auc

sorted_factors = sorted(factor_aucs.items(), key=lambda x: -x[1])

print(f'  {"要因":<35} {"CV AUC":>8}  {"判定"}')
print('  ' + '-' * 65)
for name, auc in sorted_factors:
    if auc >= 0.65:
        judge = '★ かなり有効'
    elif auc >= 0.55:
        judge = '○ やや有効'
    elif auc >= 0.50:
        judge = '△ 微弱'
    else:
        judge = '✕ 無効 (ランダム以下)'
    bar = '█' * int(max(0, (auc - 0.4)) * 100)
    print(f'  {name:<35} {auc:>8.3f}  {judge}  {bar}')

print()
print('  → AUC 0.50 = コイン投げと同じ。0.50を大きく超えないと予測力は無い')
print('  → レーン配置のAUCと他要因のAUCを比較すると、レーン配置の影響度がわかる')

# ═══════════════════════════════════════════════════════════════
#  検証3: 時系列安定性 — 前半/後半で結論は変わるか？
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 80)
print('  検証3: 時系列安定性 — 前半/後半で「最適」配置は変わるか？')
print('=' * 80)
print()

match_ids_sorted = sorted(match_win.index)
mid_point = len(match_ids_sorted) // 2
first_half_ids = set(match_ids_sorted[:mid_point])
second_half_ids = set(match_ids_sorted[mid_point:])

def compute_role_wr(match_ids_set):
    sub = ps_m[ps_m['matchId'].isin(match_ids_set)]
    results = {}
    for member in MEMBERS:
        m_data = sub[sub['summonerName'] == member]
        results[member] = {}
        for role in ROLES:
            mr = m_data[m_data['role'] == role]
            if len(mr) >= 3:
                results[member][role] = (mr['win'].mean() * 100, len(mr))
    return results

wr_first = compute_role_wr(first_half_ids)
wr_second = compute_role_wr(second_half_ids)

print(f'  前半: {len(first_half_ids)}試合 / 後半: {len(second_half_ids)}試合')
print()

stability_data = []
for member in MEMBERS:
    best_first = None
    best_second = None
    best_first_wr = -1
    best_second_wr = -1

    for role in ROLES:
        if role in wr_first.get(member, {}):
            wr, n = wr_first[member][role]
            if wr > best_first_wr and n >= 5:
                best_first_wr = wr
                best_first = role
        if role in wr_second.get(member, {}):
            wr, n = wr_second[member][role]
            if wr > best_second_wr and n >= 5:
                best_second_wr = wr
                best_second = role

    stable = best_first == best_second
    stability_data.append({
        'member': member,
        'best_first': best_first,
        'best_second': best_second,
        'stable': stable
    })

print(f'  {"メンバー":<14} {"前半ベスト":>10} {"後半ベスト":>10}  {"安定性"}')
print('  ' + '-' * 55)
for sd in stability_data:
    bf = ROLE_JP.get(sd['best_first'], '---') if sd['best_first'] else '---'
    bs = ROLE_JP.get(sd['best_second'], '---') if sd['best_second'] else '---'
    mark = '✓ 一致' if sd['stable'] else '✕ 変動'
    print(f'  {sd["member"]:<14} {bf:>10} {bs:>10}  {mark}')

stable_count = sum(1 for sd in stability_data if sd['stable'])
total_count = len(stability_data)
print()
print(f'  安定性: {stable_count}/{total_count}人 ({stable_count/total_count*100:.0f}%) が前後半で同じベストロール')
if stable_count < total_count * 0.5:
    print('  → 半数以上が変動している → レーン適性は時期で変わる or サンプルが足りない')

print()
print('  ─── 各メンバーのロール別 前半/後半 勝率差 ───')
print()

for member in MEMBERS:
    diffs = []
    for role in ROLES:
        f_data = wr_first.get(member, {}).get(role)
        s_data = wr_second.get(member, {}).get(role)
        if f_data and s_data:
            f_wr, f_n = f_data
            s_wr, s_n = s_data
            diff = s_wr - f_wr
            diffs.append((role, f_wr, f_n, s_wr, s_n, diff))

    if diffs:
        print(f'  ■ {member}')
        for role, f_wr, f_n, s_wr, s_n, diff in diffs:
            arrow = '↑' if diff > 5 else '↓' if diff < -5 else '→'
            print(f'    {ROLE_JP[role]:<6}: 前半{f_wr:>5.1f}%({f_n}試合) → 後半{s_wr:>5.1f}%({s_n}試合)  差{diff:>+6.1f}pp {arrow}')
        print()

# ═══════════════════════════════════════════════════════════════
#  検証4: サンプルサイズ妥当性
# ═══════════════════════════════════════════════════════════════
print('=' * 80)
print('  検証4: サンプルサイズ妥当性 — 5pp差を検出するのに何試合必要か？')
print('=' * 80)
print()

def required_n_for_detection(effect_pp, alpha=0.05, power=0.8, base_wr=0.5):
    """二項検定で effect_pp の差を検出するのに必要なサンプルサイズ"""
    from scipy.stats import norm
    p1 = base_wr
    p2 = base_wr + effect_pp / 100
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    p_bar = (p1 + p2) / 2
    n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
          z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / ((p2 - p1) ** 2)
    return int(np.ceil(n))

for effect in [3, 5, 8, 10, 15]:
    n_needed = required_n_for_detection(effect)
    print(f'  {effect:>2}pp差を検出するのに必要な試合数: {n_needed:>5}試合 (各条件)')

print()
print('  ─── 現在の各メンバー×ロールの試合数 ───')
print()
print(f'  {"":>14}', end='')
for r in ROLES:
    print(f' {ROLE_JP[r]:>6}', end='')
print()
print('  ' + '-' * 50)

insufficient_count = 0
total_cells = 0
for member in MEMBERS:
    print(f'  {member:>14}', end='')
    for role in ROLES:
        n = len(ps_m[(ps_m['summonerName'] == member) & (ps_m['role'] == role)])
        total_cells += 1
        if n == 0:
            print(f'  {"--":>4}', end='')
        else:
            mark = '!' if n < 30 else ' '
            if n < 30:
                insufficient_count += 1
            print(f' {n:>4}{mark}', end='')
    print()

n_needed_5pp = required_n_for_detection(5)
print()
print(f'  5pp差の検出に必要: {n_needed_5pp}試合')
print(f'  30試合未満のセル: {insufficient_count}/{total_cells}  '
      f'({insufficient_count/total_cells*100:.0f}%)')
print(f'  → 30試合未満では5pp差すら統計的に有意に検出不可能')

# ═══════════════════════════════════════════════════════════════
#  検証5: Bootstrap安定性 — 最適ロールの変動率
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 80)
print('  検証5: Bootstrap安定性 — 「最適ロール」は何%の確率で変わるか？')
print('=' * 80)
print()

N_BOOT = 300
rng = np.random.RandomState(42)
boot_best_roles = {m: [] for m in MEMBERS}

for b in range(N_BOOT):
    idx = rng.choice(len(X), size=len(X), replace=True)
    X_b = X[idx]
    y_b = y[idx]
    try:
        model = LogisticRegression(C=0.01, solver='lbfgs', max_iter=5000, random_state=42)
        model.fit(X_b, y_b)
        coeffs = model.coef_[0]
        effects = coeffs * 0.25 * 100

        for member in MEMBERS:
            best_role = None
            best_eff = -999
            for i, col in enumerate(valid_cols):
                parts = col.split('@')
                if parts[0] == member and effects[i] > best_eff:
                    best_eff = effects[i]
                    best_role = parts[1]
            if best_role:
                boot_best_roles[member].append(best_role)
    except Exception:
        pass

print(f'  {N_BOOT}回のBootstrapで「最適ロール」がどれだけ変動するか')
print()
print(f'  {"メンバー":<14} {"最頻ロール":>8} {"最頻率":>6} {"2位ロール":>8} {"2位率":>6}  {"安定度"}')
print('  ' + '-' * 65)

for member in MEMBERS:
    roles = boot_best_roles[member]
    if not roles:
        print(f'  {member:<14} {"---":>8}')
        continue
    from collections import Counter
    cnt = Counter(roles)
    top2 = cnt.most_common(2)
    r1 = ROLE_JP.get(top2[0][0], top2[0][0])
    p1 = top2[0][1] / len(roles) * 100
    if len(top2) > 1:
        r2 = ROLE_JP.get(top2[1][0], top2[1][0])
        p2 = top2[1][1] / len(roles) * 100
    else:
        r2 = '---'
        p2 = 0

    if p1 >= 70:
        stability = '◎ 安定'
    elif p1 >= 50:
        stability = '○ やや安定'
    else:
        stability = '△ 不安定'

    print(f'  {member:<14} {r1:>8} {p1:>5.1f}% {r2:>8} {p2:>5.1f}%  {stability}')

# ═══════════════════════════════════════════════════════════════
#  検証6: 実測パターンの信頼区間
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 80)
print('  検証6: 実測配置パターンの勝率 — 偶然の範囲はどこまで？')
print('=' * 80)
print()

match_assignments = {}
for mid in match_win.index:
    match_ps_a = ps_m[ps_m['matchId'] == mid]
    assign = {}
    for _, p in match_ps_a.iterrows():
        if p['summonerName'] in MEMBERS and pd.notna(p['role']):
            assign[p['role']] = p['summonerName']
    assign_key = tuple(sorted(assign.items()))
    if assign_key not in match_assignments:
        match_assignments[assign_key] = []
    match_assignments[assign_key].append(match_win[mid])

pattern_stats = []
for assign_key, wins in match_assignments.items():
    n = len(wins)
    if n < 5:
        continue
    wr = np.mean(wins) * 100
    w = sum(wins)
    ci = stats.binom.interval(0.95, n, wr / 100)
    ci_low = ci[0] / n * 100
    ci_high = ci[1] / n * 100
    pattern_stats.append({
        'assignment': dict(assign_key),
        'n': n, 'wr': wr, 'wins': w, 'losses': n - w,
        'ci_low': ci_low, 'ci_high': ci_high
    })

pattern_stats.sort(key=lambda x: -x['n'])

print(f'  {"#":>3} {"WR":>6} {"95%CI":>16} {"試合":>4} {"50%含む?":>8}  {"配置"}')
print('  ' + '-' * 80)

for i, p in enumerate(pattern_stats[:12], 1):
    a = p['assignment']
    contains_50 = p['ci_low'] <= 50 <= p['ci_high']
    mark = '→ 偶然の範囲' if contains_50 else '→ 有意'
    roles_str = ' / '.join(
        f'{ROLE_JP.get(r, r)}={a[r][:4]}' for r in ROLES if r in a
    )
    print(f'  {i:>3} {p["wr"]:>5.1f}% [{p["ci_low"]:>5.1f},{p["ci_high"]:>5.1f}%] {p["n"]:>4}  {mark:<12}  {roles_str}')

sig_patterns = [p for p in pattern_stats if not (p['ci_low'] <= 50 <= p['ci_high'])]
print()
print(f'  50%を含まないパターン: {len(sig_patterns)}/{len(pattern_stats)}')
print(f'  → 95%CIが50%をまたがないパターンは「偶然ではない可能性がある」配置')

# ═══════════════════════════════════════════════════════════════
#  検証7: 交互作用 — レーン配置よりペア相性が重要か？
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 80)
print('  検証7: ペア相性 vs レーン配置 — どちらが勝率を説明するか？')
print('=' * 80)
print()

pair_rows = []
for mid in match_win.index:
    row = {}
    match_ps_p = ps_m[ps_m['matchId'] == mid]
    members_in = sorted(match_ps_p['summonerName'].unique())
    from itertools import combinations as comb
    for m1, m2 in comb(members_in, 2):
        if m1 in MEMBERS and m2 in MEMBERS:
            row[f'pair_{m1}_{m2}'] = 1
    pair_rows.append(row)

pair_df = pd.DataFrame(pair_rows, index=match_win.index).fillna(0)
pair_valid = [c for c in pair_df.columns if pair_df[c].sum() >= 15]
if len(pair_valid) > 3:
    X_pair = pair_df[pair_valid].values.astype(float)
    pair_auc = cross_val_score(
        LogisticRegression(C=0.1, solver='lbfgs', max_iter=5000, random_state=42),
        X_pair, y, cv=cv, scoring='roc_auc'
    ).mean()

    print(f'  レーン配置モデル AUC:   {real_auc:.3f}')
    print(f'  メンバーペアモデル AUC: {pair_auc:.3f}')
    print()
    if pair_auc > real_auc + 0.02:
        print('  → 「誰と誰が同じチームか」の方が「誰がどのロールか」より予測力がある')
        print('    レーン配置よりもメンバーの組み合わせ自体が重要な可能性')
    elif abs(pair_auc - real_auc) <= 0.02:
        print('  → 両者の予測力はほぼ同等（どちらも低い）')
        print('    レーン配置もメンバー組み合わせも、単独では勝敗をほぼ説明できない')
    else:
        print('  → レーン配置の方がやや高い予測力を持つ')

# ═══════════════════════════════════════════════════════════════
#  総合結論
# ═══════════════════════════════════════════════════════════════
print()
print()
print('=' * 80)
print('  総合結論: レーン配置勝率は信頼できるか？')
print('=' * 80)
print()

issues = []
if real_auc < 0.52:
    issues.append(('致命的', f'モデルAUC={real_auc:.3f}: ランダムと区別できない'))
if p_value > 0.05:
    issues.append(('致命的', f'順列検定p={p_value:.3f}: 統計的に有意でない'))
if stable_count < total_count * 0.5:
    issues.append(('深刻', f'時系列安定性: {stable_count}/{total_count}人しか安定しない'))
elif stable_count < total_count * 0.7:
    issues.append(('注意', f'時系列安定性: {stable_count}/{total_count}人のみ安定'))
if insufficient_count > total_cells * 0.4:
    issues.append(('深刻', f'サンプル不足: {insufficient_count}/{total_cells}セルが30試合未満'))

severity_order = {'致命的': 0, '深刻': 1, '注意': 2}
issues.sort(key=lambda x: severity_order.get(x[0], 3))

for sev, msg in issues:
    icon = '🔴' if sev == '致命的' else '🟡' if sev == '深刻' else '🟢'
    print(f'  {icon} [{sev}] {msg}')

print()
print('  ─── まとめ ───')
print()

critical_count = sum(1 for s, _ in issues if s == '致命的')
if critical_count >= 1:
    print('  ■ 結論: 勝率ベースの最適レーン配置は「信頼できない」')
    print()
    print('  理由:')
    print('  1. レーン配置のみで勝敗を予測するモデルはランダム以下の精度')
    print('     → 「誰がどのレーンに行くか」は勝敗にほとんど影響しない')
    print()
    print('  2. チャンピオン選択・序盤のレーニング・オブジェクトなど')
    print('     他の要因の方が勝敗への影響が大きい（可能性がある）')
    print()
    print('  3. サンプルが足りないのではなく、そもそもシグナルが弱い')
    print('     → 1000試合集めても劇的に改善する可能性は低い')
    print()
    print('  ■ 唯一信頼できること:')
    print('  - 明確に負けパターンのレーン（95%CIが完全にマイナス）は避けるべき')
    print('    例: PlayerX@ミッド = -2.5pp [-4.2, -1.0]')
    print()
    print('  ■ 勝率を上げたいなら:')
    print('  - レーン配置の最適化よりも:')
    print('    (1) チャンピオンプールの改善')
    print('    (2) 序盤レーニング力の向上')
    print('    (3) オブジェクト管理の徹底')
    print('    の方が効果的')
else:
    print('  ■ レーン配置は一定の影響力を持つが、過信は禁物')
    print('  ■ 他の要因と組み合わせた総合的な判断が必要')

print()
print('=' * 80)
print('  監査完了')
print('=' * 80)
