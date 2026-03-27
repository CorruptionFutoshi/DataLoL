"""序盤ビハインド時のメンタル耐性ランキング（ロール補正版）
同ロールの平均的プレイヤーと比較し、「普通以上に崩れる度合い」を測定する。"""
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

EARLY_MIN = 10
BEHIND_THRESHOLD = -300
METRICS = ['kda', 'deaths', 'kp', 'dmg_share', 'visionScore', 'cs']
METRIC_JP = {'kda': 'KDA', 'deaths': 'デス', 'kp': 'KP',
             'dmg_share': 'ダメ割合', 'visionScore': 'ビジョン', 'cs': 'CS'}

# ═══════════════════════════════════════════════════════════════
#  Step 1: 全プレイヤーにメトリクス・序盤分類を付与
# ═══════════════════════════════════════════════════════════════
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
ps_all['is_member'] = ps_all['summonerName'].isin(MEMBERS)


# ═══════════════════════════════════════════════════════════════
#  Step 2: ロール別ベースライン — 全プレイヤーの平均的な低下幅
# ═══════════════════════════════════════════════════════════════
def calc_degradation(df_ahead, df_behind):
    """序盤勝ち→序盤負けの変化量を計算"""
    if len(df_ahead) < 3 or len(df_behind) < 3:
        return None
    rec = {}
    for m in METRICS:
        val_a = df_ahead[m].mean()
        val_b = df_behind[m].mean()
        rec[f'{m}_ahead'] = val_a
        rec[f'{m}_behind'] = val_b
        if m == 'deaths':
            rec[f'{m}_change_pct'] = ((val_b - val_a) / val_a * 100) if val_a > 0 else 0
        elif m in ('kp', 'dmg_share'):
            rec[f'{m}_change_pp'] = (val_a - val_b) * 100
        else:
            rec[f'{m}_change_pct'] = ((val_a - val_b) / val_a * 100) if val_a > 0 else 0
    rec['wr_ahead'] = df_ahead['win'].mean() * 100
    rec['wr_behind'] = df_behind['win'].mean() * 100
    rec['wr_drop'] = rec['wr_ahead'] - rec['wr_behind']
    return rec


role_baselines = {}
for role in ROLES:
    role_data = ps_all[ps_all['role'] == role]
    bl = calc_degradation(role_data[role_data['ahead']], role_data[role_data['behind']])
    if bl:
        bl['n_ahead'] = len(role_data[role_data['ahead']])
        bl['n_behind'] = len(role_data[role_data['behind']])
        role_baselines[role] = bl


# ═══════════════════════════════════════════════════════════════
#  Step 3: メンバーごとにロール加重平均の「超過低下」を計算
# ═══════════════════════════════════════════════════════════════
member_records = []

for name in MEMBERS:
    player = ps_all[ps_all['summonerName'] == name]
    player_behind = player[player['behind']]
    player_ahead = player[player['ahead']]

    if len(player_behind) < 3 or len(player_ahead) < 3:
        continue

    # Per-role comparison vs baseline
    role_diffs = []
    role_details = []

    for role in ROLES:
        pr = player[player['role'] == role]
        pr_behind = pr[pr['behind']]
        pr_ahead = pr[pr['ahead']]
        n_role = len(pr)

        if n_role < 3 or role not in role_baselines:
            continue
        if len(pr_behind) < 2 or len(pr_ahead) < 2:
            continue

        bl = role_baselines[role]
        player_deg = calc_degradation(pr_ahead, pr_behind)
        if player_deg is None:
            continue

        excess = {}
        for m in METRICS:
            if m == 'deaths':
                p_change = player_deg[f'{m}_change_pct']
                bl_change = bl[f'{m}_change_pct']
                excess[m] = p_change - bl_change
            elif m in ('kp', 'dmg_share'):
                p_change = player_deg[f'{m}_change_pp']
                bl_change = bl[f'{m}_change_pp']
                excess[m] = p_change - bl_change
            else:
                p_change = player_deg[f'{m}_change_pct']
                bl_change = bl[f'{m}_change_pct']
                excess[m] = p_change - bl_change

        excess['wr_drop'] = player_deg['wr_drop'] - bl['wr_drop']

        role_diffs.append({
            'role': role,
            'n': n_role,
            'n_behind': len(pr_behind),
            'n_ahead': len(pr_ahead),
            **{f'excess_{k}': v for k, v in excess.items()},
            **{f'player_{k}': v for k, v in player_deg.items()},
            **{f'baseline_{k}': v for k, v in bl.items()},
        })

    if not role_diffs:
        continue

    rd_df = pd.DataFrame(role_diffs)
    total_n = rd_df['n'].sum()
    weights = rd_df['n'] / total_n

    rec = {'name': name, 'total': len(player),
           'n_behind': len(player_behind), 'n_ahead': len(player_ahead),
           'behind_rate': len(player_behind) / len(player) * 100}

    # Weighted average of excess degradation across roles
    for m in METRICS:
        rec[f'excess_{m}'] = (rd_df[f'excess_{m}'] * weights).sum()
    rec['excess_wr_drop'] = (rd_df['excess_wr_drop'] * weights).sum()

    # Overall raw stats (for display)
    overall = calc_degradation(player_ahead, player_behind)
    rec.update({f'raw_{k}': v for k, v in overall.items()})

    # Main role
    main_role_row = rd_df.sort_values('n', ascending=False).iloc[0]
    rec['main_role'] = main_role_row['role']

    # Role breakdown for detail display
    rec['_role_details'] = role_diffs

    member_records.append(rec)

mdf = pd.DataFrame(member_records)


# ═══════════════════════════════════════════════════════════════
#  Step 4: 総合ティルトスコア（超過低下ベース）
# ═══════════════════════════════════════════════════════════════
def norm(series, higher_is_worse=True):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    n = (series - mn) / (mx - mn) * 100
    return n if higher_is_worse else (100 - n)


mdf['s_kda'] = norm(mdf['excess_kda'], higher_is_worse=True)
mdf['s_deaths'] = norm(mdf['excess_deaths'], higher_is_worse=True)
mdf['s_kp'] = norm(mdf['excess_kp'], higher_is_worse=True)
mdf['s_dmg'] = norm(mdf['excess_dmg_share'], higher_is_worse=True)
mdf['s_vision'] = norm(mdf['excess_visionScore'], higher_is_worse=True)
mdf['s_cs'] = norm(mdf['excess_cs'], higher_is_worse=True)
mdf['s_wr'] = norm(mdf['excess_wr_drop'], higher_is_worse=True)

W = {'s_kda': 0.25, 's_deaths': 0.25, 's_kp': 0.15,
     's_dmg': 0.10, 's_vision': 0.05, 's_cs': 0.05, 's_wr': 0.15}
mdf['tilt_score'] = sum(mdf[c] * w for c, w in W.items())
mdf = mdf.sort_values('tilt_score', ascending=False)


# ═══════════════════════════════════════════════════════════════
#  Output: ロール別ベースライン
# ═══════════════════════════════════════════════════════════════
print()
print('=' * 78)
print('  ロール別 序盤ビハインド時の平均的パフォーマンス低下（全プレイヤー）')
print('=' * 78)
print()
print(f'  ※ 全プレイヤー（チームメンバー以外含む）の{EARLY_MIN}分ゴールド差 {BEHIND_THRESHOLD}G を')
print(f'    基準とした、序盤勝ち→序盤負け時の変化量')
print()
print(f'  {"ロール":<10} {"KDA低下":>10} {"デス増加":>10} {"KP低下":>10} {"ダメ割↓":>10} {"ビジョン↓":>10} {"WR差":>10} {"n(ahead)":>10} {"n(behind)":>10}')
print(f'  {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
for role in ROLES:
    if role not in role_baselines:
        continue
    bl = role_baselines[role]
    print(f'  {ROLE_JP[role]:<10}'
          f' {bl["kda_change_pct"]:>+8.0f}%'
          f' {bl["deaths_change_pct"]:>+8.0f}%'
          f' {bl["kp_change_pp"]:>+7.1f}pp'
          f' {bl["dmg_share_change_pp"]:>+7.1f}pp'
          f' {bl["visionScore_change_pct"]:>+8.0f}%'
          f' {bl["wr_drop"]:>+7.1f}pp'
          f' {bl["n_ahead"]:>10}'
          f' {bl["n_behind"]:>10}')

print()
print('  → これが「普通の低下」。メンバーの低下がこれより大きければ"メンタル崩壊"。')


# ═══════════════════════════════════════════════════════════════
#  Output: メンバーランキング
# ═══════════════════════════════════════════════════════════════
print()
print()
print('=' * 78)
print('  メンタルブーム度ランキング（ロール補正版）')
print('  〜 同ロールの平均プレイヤーと比較し、序盤負け時に"余計に"崩れる度合い 〜')
print('=' * 78)
print()
print(f'  判定基準: {EARLY_MIN}分時点のゴールド差 {BEHIND_THRESHOLD:+}G で 序盤勝ち/負けを分類')
print(f'  比較対象: 同ロールの全プレイヤー（対戦相手含む）の平均低下幅')
print(f'  超過低下: メンバーの低下 − ロール平均の低下（正=平均より崩れる、負=平均より耐える）')
print()

medals = ['💀 1位', '😰 2位', '😥 3位', '😐 4位', '🙂 5位', '😎 6位', '🧘 7位']

for rank, (_, row) in enumerate(mdf.iterrows()):
    medal = medals[rank] if rank < len(medals) else f'   {rank+1}位'
    main_r = ROLE_JP.get(row['main_role'], row['main_role'])

    print()
    print('━' * 78)
    print(f'  {medal}  {row["name"]}（主:{main_r}）  ティルトスコア: {row["tilt_score"]:.1f} / 100')
    print('━' * 78)
    print(f'    全{int(row["total"])}試合 | 序盤負け {int(row["n_behind"])}試合({row["behind_rate"]:.0f}%) | 序盤勝ち {int(row["n_ahead"])}試合')
    print()

    # Actual performance: ahead vs behind
    print(f'    【実際のパフォーマンス変化】')
    print(f'      {"指標":<12} {"序盤勝ち":>10} {"序盤負け":>10} {"変化":>10}  {"ロール平均":>10} {"超過低下":>10}')
    print(f'      {"-"*12} {"-"*10} {"-"*10} {"-"*10}  {"-"*10} {"-"*10}')

    # KDA
    kda_a = row['raw_kda_ahead']
    kda_b = row['raw_kda_behind']
    kda_chg = row['raw_kda_change_pct']
    print(f'      {"KDA":<12} {kda_a:>10.2f} {kda_b:>10.2f} {kda_chg:>+8.0f}%  '
          f'{"(平均低下)":>10} {row["excess_kda"]:>+8.1f}pp')

    # Deaths
    d_a = row['raw_deaths_ahead']
    d_b = row['raw_deaths_behind']
    d_chg = row['raw_deaths_change_pct']
    print(f'      {"デス":<12} {d_a:>10.1f} {d_b:>10.1f} {d_chg:>+8.0f}%  '
          f'{"(平均増加)":>10} {row["excess_deaths"]:>+8.1f}pp')

    # KP
    kp_a = row['raw_kp_ahead']
    kp_b = row['raw_kp_behind']
    kp_chg = row['raw_kp_change_pp']
    print(f'      {"KP":<12} {kp_a:>9.0%}  {kp_b:>9.0%}  {kp_chg:>+7.1f}pp  '
          f'{"(平均低下)":>10} {row["excess_kp"]:>+7.1f}pp')

    # Damage share
    ds_a = row['raw_dmg_share_ahead']
    ds_b = row['raw_dmg_share_behind']
    ds_chg = row['raw_dmg_share_change_pp']
    print(f'      {"ダメ割合":<12} {ds_a:>9.0%}  {ds_b:>9.0%}  {ds_chg:>+7.1f}pp  '
          f'{"(平均低下)":>10} {row["excess_dmg_share"]:>+7.1f}pp')

    # Vision
    v_a = row['raw_visionScore_ahead']
    v_b = row['raw_visionScore_behind']
    v_chg = row['raw_visionScore_change_pct']
    print(f'      {"ビジョン":<12} {v_a:>10.1f} {v_b:>10.1f} {v_chg:>+8.0f}%  '
          f'{"(平均低下)":>10} {row["excess_visionScore"]:>+8.1f}pp')

    # CS
    cs_a = row['raw_cs_ahead']
    cs_b = row['raw_cs_behind']
    cs_chg = row['raw_cs_change_pct']
    print(f'      {"CS":<12} {cs_a:>10.1f} {cs_b:>10.1f} {cs_chg:>+8.0f}%  '
          f'{"(平均低下)":>10} {row["excess_cs"]:>+8.1f}pp')

    # Tilt symptoms based on excess
    print()
    symptoms = []
    ex = row
    if ex['excess_deaths'] > 10:
        symptoms.append(f'デスが同ロール平均より+{ex["excess_deaths"]:.0f}pp余計に増加 → ビハインド時に無理しすぎ')
    if ex['excess_kda'] > 10:
        symptoms.append(f'KDAが同ロール平均より+{ex["excess_kda"]:.0f}pp余計に低下 → 全体的に崩れやすい')
    if ex['excess_kp'] > 3:
        symptoms.append(f'KPが同ロール平均より+{ex["excess_kp"]:.1f}pp余計に低下 → チーム戦への参加意欲が平均以上に落ちる')
    if ex['excess_dmg_share'] > 2:
        symptoms.append(f'ダメ割合が同ロール平均より+{ex["excess_dmg_share"]:.1f}pp余計に低下 → 存在感がなくなる')
    if ex['excess_visionScore'] > 5:
        symptoms.append(f'ビジョンが同ロール平均より+{ex["excess_visionScore"]:.0f}pp余計に低下 → 視界管理を放棄')

    # Positive traits
    good = []
    if ex['excess_deaths'] < -5:
        good.append(f'デス増加が同ロール平均より{abs(ex["excess_deaths"]):.0f}pp少ない → ビハインドでも冷静')
    if ex['excess_kda'] < -5:
        good.append(f'KDA低下が同ロール平均より{abs(ex["excess_kda"]):.0f}pp少ない → 崩れにくい')
    if ex['excess_kp'] < -2:
        good.append(f'KP低下が同ロール平均より{abs(ex["excess_kp"]):.1f}pp少ない → ビハインドでもチーム戦に参加')
    if ex['excess_wr_drop'] < -5:
        good.append(f'勝率低下が同ロール平均より{abs(ex["excess_wr_drop"]):.0f}pp少ない → 逆転力あり')

    if symptoms:
        print(f'    【ティルト症状（ロール平均超過分）】')
        for s in symptoms:
            print(f'      ⚠ {s}')
    if good:
        print(f'    【メンタル耐性（ロール平均との比較）】')
        for g in good:
            print(f'      ✅ {g}')
    if not symptoms and not good:
        print(f'    → ロール平均とほぼ同程度の低下。特筆すべきティルト傾向なし。')

    # Role breakdown
    details = row['_role_details']
    if len(details) > 1:
        print()
        print(f'    【ロール別 超過低下の内訳】')
        print(f'      {"ロール":<10} {"試合数":>6} {"KDA超過":>10} {"デス超過":>10} {"KP超過":>10}')
        print(f'      {"-"*10} {"-"*6} {"-"*10} {"-"*10} {"-"*10}')
        for d in sorted(details, key=lambda x: x['n'], reverse=True):
            rj = ROLE_JP.get(d['role'], d['role'])
            print(f'      {rj:<10} {d["n"]:>6}'
                  f' {d["excess_kda"]:>+8.1f}pp'
                  f' {d["excess_deaths"]:>+8.1f}pp'
                  f' {d["excess_kp"]:>+8.1f}pp')


# ═══════════════════════════════════════════════════════════════
#  Summary table
# ═══════════════════════════════════════════════════════════════
print()
print()
print('=' * 78)
print('  超過低下 サマリー（正の値 = 平均より崩れる / 負の値 = 平均より耐える）')
print('=' * 78)
print()
print(f'  {"メンバー":<14} {"KDA超過":>8} {"デス超過":>8} {"KP超過":>8} {"ダメ割超過":>10} {"WR超過":>8} {"スコア":>8}')
print(f'  {"-"*14} {"-"*8} {"-"*8} {"-"*8} {"-"*10} {"-"*8} {"-"*8}')
for _, row in mdf.iterrows():
    print(f'  {row["name"]:<14}'
          f' {row["excess_kda"]:>+6.1f}pp'
          f' {row["excess_deaths"]:>+6.1f}pp'
          f' {row["excess_kp"]:>+6.1f}pp'
          f' {row["excess_dmg_share"]:>+8.1f}pp'
          f' {row["excess_wr_drop"]:>+6.1f}pp'
          f' {row["tilt_score"]:>7.1f}')


# ═══════════════════════════════════════════════════════════════
#  Conclusions
# ═══════════════════════════════════════════════════════════════
print()
print()
print('=' * 78)
print('  まとめ & アドバイス')
print('=' * 78)
print()

top = mdf.iloc[0]
bot = mdf.iloc[-1]

print(f'  🔥 最もメンタルが崩れやすい: {top["name"]}')
print(f'     同ロールの平均プレイヤーと比べて:')
if top['excess_kda'] > 0:
    print(f'     → KDA低下が平均より +{top["excess_kda"]:.1f}pp 大きい')
if top['excess_deaths'] > 0:
    print(f'     → デス増加が平均より +{top["excess_deaths"]:.1f}pp 大きい')
if top['excess_kp'] > 0:
    print(f'     → KP低下が平均より +{top["excess_kp"]:.1f}pp 大きい')
print(f'     ✏ 序盤負けた時こそファームと安全なプレイを意識。味方はフォローを。')
print()

print(f'  🧘 最もメンタルが安定: {bot["name"]}')
print(f'     同ロールの平均プレイヤーと比べて:')
if bot['excess_kda'] < 0:
    print(f'     → KDA低下が平均より {bot["excess_kda"]:.1f}pp 小さい（耐える）')
if bot['excess_deaths'] < 0:
    print(f'     → デス増加が平均より {bot["excess_deaths"]:.1f}pp 小さい（冷静）')
print(f'     ✏ チームのメンタル支柱。苦しい展開で頼りになる。')
print()
