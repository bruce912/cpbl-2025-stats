import os
import pandas as pd
from flask import Flask, jsonify, send_from_directory, request

app = Flask(__name__, static_folder='static', static_url_path='')
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


# ─────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────
def load_data():
    games   = pd.read_csv(os.path.join(DATA_DIR, 'cpbl_2025_games.csv'))
    batting = pd.read_csv(os.path.join(DATA_DIR, 'cpbl_2025_batting.csv'))
    pitching= pd.read_csv(os.path.join(DATA_DIR, 'cpbl_2025_pitching.csv'))

    games['GameDate']    = pd.to_datetime(games['GameDate'], errors='coerce')
    games['GameDateStr'] = games['GameDate'].dt.strftime('%Y-%m-%d').fillna('')

    ginfo = games[['GameSno','GameDate','GameDateStr',
                   'VisitingTeamName','HomeTeamName','FieldAbbe',
                   'VisitingTotalScore','HomeTotalScore',
                   'WinningPitcherName','LosePitcherName','CloserPitcherName',
                   'GameDuringTime','AudienceCntBackend','GameStatus']].copy()

    def enrich(df):
        merged = df.merge(ginfo, on='GameSno', how='left')
        merged['TeamName'] = merged.apply(
            lambda r: r['VisitingTeamName'] if r['VisitingHomeType'] == 1 else r['HomeTeamName'], axis=1)
        merged['OpponentName'] = merged.apply(
            lambda r: r['HomeTeamName'] if r['VisitingHomeType'] == 1 else r['VisitingTeamName'], axis=1)
        return merged

    return games, enrich(batting), enrich(pitching)


games_df, batting_df, pitching_df = load_data()
TEAMS = sorted(batting_df['TeamName'].dropna().unique().tolist())


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def apply_filters(df, team, opponent, date_from, date_to):
    if team:
        df = df[df['TeamName'] == team]
    if opponent:
        df = df[df['OpponentName'] == opponent]
    if date_from:
        df = df[df['GameDate'] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df['GameDate'] <= pd.to_datetime(date_to)]
    return df


def fmt_ip(x):
    full   = int(x)
    thirds = min(max(round((x - full) * 3), 0), 2)
    return f"{full}.{thirds}"


def fmt_duration(s):
    try:
        s = str(int(float(s))).zfill(6)
        return f"{int(s[:2])}:{s[2:4]}"
    except Exception:
        return ''


def n(x):
    """safe int"""
    try: return int(x)
    except: return 0


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/meta')
def meta():
    return jsonify({'teams': TEAMS})


@app.route('/api/standings')
def standings():
    done = games_df[games_df['GameStatus'].isin([2, 3, 7, 8])].copy()
    rows = {t: {'team': t, 'W': 0, 'L': 0, 'T': 0} for t in TEAMS}

    for _, g in done.iterrows():
        vt, ht = g['VisitingTeamName'], g['HomeTeamName']
        vs, hs = g['VisitingTotalScore'], g['HomeTotalScore']
        if pd.isna(vs) or pd.isna(hs):
            continue
        vs, hs = int(vs), int(hs)
        if vs > hs:
            if vt in rows: rows[vt]['W'] += 1
            if ht in rows: rows[ht]['L'] += 1
        elif hs > vs:
            if ht in rows: rows[ht]['W'] += 1
            if vt in rows: rows[vt]['L'] += 1
        else:
            if vt in rows: rows[vt]['T'] += 1
            if ht in rows: rows[ht]['T'] += 1

    result = list(rows.values())
    for r in result:
        wl = r['W'] + r['L']
        r['G']  = r['W'] + r['L'] + r['T']
        r['WP'] = round(r['W'] / wl, 3) if wl > 0 else 0.0

    result.sort(key=lambda x: (-x['WP'], -x['W']))

    lw = result[0]['W'] if result else 0
    ll = result[0]['L'] if result else 0
    for i, r in enumerate(result):
        gb = ((lw - r['W']) + (r['L'] - ll)) / 2
        r['GB']   = '-' if i == 0 else (int(gb) if gb == int(gb) else gb)
        r['Rank'] = i + 1

    return jsonify(result)


@app.route('/api/batting')
def batting_stats():
    df = apply_filters(batting_df.copy(),
        request.args.get('team',''), request.args.get('opponent',''),
        request.args.get('date_from',''), request.args.get('date_to',''))

    # fill numeric columns
    for col in ['PlateAppearances','HitCnt','HittingCnt','TwoBaseHitCnt','ThreeBaseHitCnt',
                'HomeRunCnt','RunBattedINCnt','ScoreCnt','BasesONBallsCnt','HitBYPitchCnt',
                'StrikeOutCnt','StealBaseOKCnt','SacrificeFlyCnt','TotalBases']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # recalculate TotalBases where 0
    mask = df['TotalBases'] == 0
    df.loc[mask, 'TotalBases'] = (
        df.loc[mask,'HittingCnt'] +
        df.loc[mask,'TwoBaseHitCnt'] +
        df.loc[mask,'ThreeBaseHitCnt'] * 2 +
        df.loc[mask,'HomeRunCnt'] * 3
    )

    agg = df.groupby(['HitterName','TeamName']).agg(
        G   =('GameSno','nunique'),
        PA  =('PlateAppearances','sum'),
        AB  =('HitCnt','sum'),
        H   =('HittingCnt','sum'),
        D   =('TwoBaseHitCnt','sum'),
        T   =('ThreeBaseHitCnt','sum'),
        HR  =('HomeRunCnt','sum'),
        RBI =('RunBattedINCnt','sum'),
        R   =('ScoreCnt','sum'),
        BB  =('BasesONBallsCnt','sum'),
        HBP =('HitBYPitchCnt','sum'),
        K   =('StrikeOutCnt','sum'),
        SB  =('StealBaseOKCnt','sum'),
        SF  =('SacrificeFlyCnt','sum'),
        TB  =('TotalBases','sum'),
    ).reset_index()

    def sdiv(a, b): return round(a / b, 3) if b > 0 else 0.0

    agg['AVG'] = agg.apply(lambda r: sdiv(r['H'], r['AB']), axis=1)
    agg['OBP'] = agg.apply(
        lambda r: sdiv(r['H'] + r['BB'] + r['HBP'], r['AB'] + r['BB'] + r['HBP'] + r['SF']), axis=1)
    agg['SLG'] = agg.apply(lambda r: sdiv(r['TB'], r['AB']), axis=1)
    agg['OPS'] = (agg['OBP'] + agg['SLG']).round(3)

    int_cols = ['G','PA','AB','H','D','T','HR','RBI','R','BB','HBP','K','SB']
    agg[int_cols] = agg[int_cols].astype(int)
    agg = agg.drop(columns=['SF','TB'])
    agg = agg.sort_values('PA', ascending=False)
    return jsonify(agg.to_dict('records'))


@app.route('/api/pitching')
def pitching_stats():
    df = apply_filters(pitching_df.copy(),
        request.args.get('team',''), request.args.get('opponent',''),
        request.args.get('date_from',''), request.args.get('date_to',''))

    for col in ['InningPitchedCnt','InningPitchedDiv3Cnt','PlateAppearances','PitchCnt',
                'HittingCnt','HomeRunCnt','BasesONBallsCnt','HitBYPitchCnt','StrikeOutCnt',
                'RunCnt','EarnedRunCnt','WildPitchCnt','BalkCnt','ReliefPointCnt']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['IP_dec']  = df['InningPitchedCnt'] + df['InningPitchedDiv3Cnt'] / 3.0
    df['is_win']  = df['GameResult'].astype(str).str.strip().str.upper() == 'W'
    df['is_loss'] = df['GameResult'].astype(str).str.strip().str.upper() == 'L'
    df['is_save'] = pd.to_numeric(df['IsSaveOK'], errors='coerce').fillna(0) > 0
    df['is_hold'] = df['ReliefPointCnt'] > 0

    agg = df.groupby(['PitcherName','TeamName']).agg(
        G    =('GameSno','nunique'),
        W    =('is_win','sum'),
        L    =('is_loss','sum'),
        S    =('is_save','sum'),
        HLD  =('is_hold','sum'),
        IP_r =('IP_dec','sum'),
        BF   =('PlateAppearances','sum'),
        H    =('HittingCnt','sum'),
        HR   =('HomeRunCnt','sum'),
        BB   =('BasesONBallsCnt','sum'),
        HBP  =('HitBYPitchCnt','sum'),
        K    =('StrikeOutCnt','sum'),
        R    =('RunCnt','sum'),
        ER   =('EarnedRunCnt','sum'),
        P    =('PitchCnt','sum'),
        WILD =('WildPitchCnt','sum'),
        BLK  =('BalkCnt','sum'),
    ).reset_index()

    agg['IP']   = agg['IP_r'].apply(fmt_ip)
    agg['ERA']  = agg.apply(lambda r: round(r['ER']*9/r['IP_r'],2) if r['IP_r']>0 else 0.0, axis=1)
    agg['WHIP'] = agg.apply(lambda r: round((r['H']+r['BB'])/r['IP_r'],2) if r['IP_r']>0 else 0.0, axis=1)

    int_cols = ['G','W','L','S','HLD','BF','H','HR','BB','HBP','K','R','ER','P','WILD','BLK']
    agg[int_cols] = agg[int_cols].astype(int)
    agg = agg.sort_values('IP_r', ascending=False).drop(columns=['IP_r'])
    return jsonify(agg.to_dict('records'))


@app.route('/api/games')
def games_api():
    df = games_df.copy()
    team     = request.args.get('team','')
    opponent = request.args.get('opponent','')
    date_from= request.args.get('date_from','')
    date_to  = request.args.get('date_to','')

    if team and opponent:
        df = df[
            ((df['VisitingTeamName']==team)&(df['HomeTeamName']==opponent)) |
            ((df['HomeTeamName']==team)&(df['VisitingTeamName']==opponent))
        ]
    elif team:
        df = df[(df['VisitingTeamName']==team)|(df['HomeTeamName']==team)]
    elif opponent:
        df = df[(df['VisitingTeamName']==opponent)|(df['HomeTeamName']==opponent)]

    if date_from:
        df = df[df['GameDate'] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df['GameDate'] <= pd.to_datetime(date_to)]

    df = df.sort_values('GameDate', ascending=False)
    df['Duration'] = df['GameDuringTime'].apply(fmt_duration)
    df['Audience'] = df['AudienceCntBackend'].fillna('').astype(str).str.replace(r'\.0$','',regex=True)

    cols = ['GameSno','GameDateStr','VisitingTeamName','VisitingTotalScore',
            'HomeTeamName','HomeTotalScore','FieldAbbe','Duration',
            'WinningPitcherName','LosePitcherName','CloserPitcherName','Audience']
    result = df[cols].fillna('').to_dict('records')
    for r in result:
        try: r['VisitingTotalScore'] = int(float(r['VisitingTotalScore']))
        except: r['VisitingTotalScore'] = '-'
        try: r['HomeTotalScore'] = int(float(r['HomeTotalScore']))
        except: r['HomeTotalScore'] = '-'
    return jsonify(result)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=False, host='0.0.0.0', port=port)
