import pandas as pd
from config import airport

def parse_clock_minutes(series, column):
    """Convert CSV ETA/ETD to minutes since midnight; reject ambiguous or invalid values."""
    raw = series.astype('string').str.strip()
    numeric = pd.to_numeric(raw.str.replace(',', '.', regex=False), errors='coerce')
    if numeric.notna().all():
        if numeric.between(0, 1, inclusive='left').all():
            minutes = numeric.astype(float) * 1440
        else:
            raise ValueError(
                f'{column} contains numeric values whose unit is ambiguous: '
                f'{raw.head(10).tolist()}. Expected HH:MM[:SS] or Excel day fractions [0,1).'
            )
    else:
        clock = raw.str.extract(r'(\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?)\s*$')[0]
        clock = clock.where(~clock.str.fullmatch(r'\d{1,2}:\d{2}'), clock + ':00')
        duration = pd.to_timedelta(clock.str.replace(',', '.', regex=False), errors='coerce')
        minutes = duration.dt.total_seconds() / 60
    if minutes.isna().any():
        bad = raw[minutes.isna()].head(10).tolist()
        raise ValueError(f'Cannot parse {column} values: {bad}. Check CSV time format.')
    if not minutes.between(0, 1440).all():
        raise ValueError(f'{column} outside the expected 00:00-24:00 time range.')
    print(f'{column}: raw sample={raw.head(5).tolist()}, '
          f'min/max={minutes.min():.1f}/{minutes.max():.1f} minutes, '
          f'unique times={minutes.nunique()}')
    if column == 'ETA' and minutes.nunique() == 1:
        raise ValueError(
            f'All {len(minutes)} arrivals have the same ETA ({minutes.iloc[0]:.1f} min). '
            f'Raw examples: {raw.head(10).tolist()}. The input ETA data/format '
            'must be corrected before any GDP/GHP calculations.'
        )
    return minutes.astype(float)

def readArrivals(filename):  #selecionar aviones con destino LIRF
    df = pd.read_csv(filename, sep=';', decimal=',')
    df['Distance'] = df['Distance'] * 1.852  # NM -> km
    for col in ['ADES', 'ETA', 'ETD']:
        if col not in df.columns:
            raise ValueError(f'Missing {col} column. Available columns: {list(df.columns)}')
    arr = df[df['ADES'].astype(str).str.strip().str.upper() == airport.upper()].copy()  #pasar ETA y ETD a min
    if arr.empty:
        raise ValueError(f'No arrivals found for {airport}. ADES counts: '
                         f'{df["ADES"].value_counts().head(10).to_dict()}')
    arr['ETA_min'] = parse_clock_minutes(arr['ETA'], 'ETA')
    arr['ETD_min'] = parse_clock_minutes(arr['ETD'], 'ETD')
    arr.loc[arr['ETD_min'] > arr['ETA_min'], 'ETD_min'] -= 1440
    arr['NonECAC'] = arr['ECAC SI/No'].astype(str).str.strip() == 'Fuera de ECAC'
    arr['Seats'] = pd.to_numeric(arr['Available seats'], errors='coerce')
    if 'Compañía' not in arr.columns:
        arr['Compañía'] = arr['ARCID'].str[:3]
    arr = arr.sort_values('ETA_min').reset_index(drop=True)
    arr.index = arr.index + 1
    return arr
